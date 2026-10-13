#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Gated DeltaNet Stepwise Recurrent Kernel (Decode Mode)
// 1 Thread Block per Head (num_heads = 16)
// 64 Threads per Block (each thread processes 1 row of S [64x64])
// Uses Shared Memory Tiling [64][65] with +1 bank-conflict-free padding
// Memory complexity per step: O(1)
extern "C" __global__ void __launch_bounds__(64, 4) rada_deltanet_step_kernel(
    float* __restrict__ S,                 // [num_heads, 64, 64] recurrent state matrix
    const float* __restrict__ q,           // [num_heads, 64] query vector
    const float* __restrict__ k,           // [num_heads, 64] key vector
    const float* __restrict__ v,           // [num_heads, 64] value vector
    const float* __restrict__ alpha,       // [num_heads] decay / forget gate
    const float* __restrict__ beta,        // [num_heads] write rate / learning rate
    float* __restrict__ y_out,             // [num_heads, 64] retrieved output vector
    int num_heads,
    int head_dim                           // 64
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int i = threadIdx.x; // row index in [0, 63]

    // Shared memory for key, query, and bank-conflict-free tiled S matrix
    __shared__ float s_k[64];
    __shared__ float s_q[64];
    __shared__ float s_S[64][65]; // +1 padding eliminates 32-way shared memory bank conflicts

    // Cooperative load of key, query, and head state row into shared memory
    s_k[i] = k[h * head_dim + i];
    s_q[i] = q[h * head_dim + i];
    float v_i = v[h * head_dim + i];
    float a_h = alpha[h];
    float b_h = beta[h];

    // In-SRAM L2 normalization of key and query across 64 elements to ensure contractive recurrence stability
    float k_sq = s_k[i] * s_k[i];
    float q_sq = s_q[i] * s_q[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
    }
    __shared__ float s_k_sum[2];
    __shared__ float s_q_sum[2];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) {
        s_k_sum[wid] = k_sq;
        s_q_sum[wid] = q_sq;
    }
    __syncthreads();
    float inv_k_norm = rsqrtf(s_k_sum[0] + s_k_sum[1] + 1e-10f);
    float inv_q_norm = rsqrtf(s_q_sum[0] + s_q_sum[1] + 1e-10f);
    s_k[i] = s_k[i] * inv_k_norm;
    s_q[i] = s_q[i] * inv_q_norm;

    float* S_h = S + (size_t)h * (head_dim * head_dim);

    // 100% Coalesced cooperative 128-byte global memory load into shared memory
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < 4096; idx += blockDim.x) {
        int r = idx / 64;
        int c = idx % 64;
        s_S[r][c] = S_h[idx];
    }

    __syncthreads();

    // 1. Predict retrieved value in SRAM: v_pred[i] = sum_j (s_S[i][j] * s_k[j])
    float v_pred_i = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < 64; ++j) {
        v_pred_i += s_S[i][j] * s_k[j];
    }

    // 2. Compute delta error
    float v_err_i = v_i - v_pred_i;

    // 3. Update recurrent state matrix in SRAM and compute retrieved output:
    //    s_S[i][j] = a_h * s_S[i][j] + b_h * v_err_i * s_k[j]
    //    y_out[i] = sum_j (s_S[i][j] * s_q[j])
    float y_i = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < 64; ++j) {
        float s_val = s_S[i][j];
        float s_new = a_h * s_val + b_h * v_err_i * s_k[j];
        s_S[i][j] = s_new;
        y_i += s_new * s_q[j];
    }

    // 4. Write updated state back to global memory (coalesced 128-byte writes)
    __syncthreads();
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < 4096; idx += blockDim.x) {
        int r = idx / 64;
        int c = idx % 64;
        S_h[idx] = s_S[r][c];
    }

    y_out[h * head_dim + i] = y_i;
}

// Causal 1D Depthwise Convolution Step Kernel (k=4)
// Maintains a 3-element history per channel, executes SiLU activation, adds residual
__global__ void __launch_bounds__(256, 4) rada_deltanet_conv1d_step_kernel(
    float* __restrict__ conv_buf,          // [hidden_size, 3] circular history buffer
    const float* __restrict__ x_in,        // [hidden_size] input token vector
    const float* __restrict__ conv_weight, // [hidden_size, 4] depthwise conv weights
    const float* __restrict__ conv_bias,   // [hidden_size] conv bias
    float* __restrict__ x_out,             // [hidden_size] output vector (x_in + silu(conv))
    int hidden_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden_size) return;

    float* buf_c = conv_buf + (size_t)c * 3;
    const float* w_c = conv_weight + (size_t)c * 4;

    float h0 = buf_c[0];
    float h1 = buf_c[1];
    float h2 = buf_c[2];
    float curr = x_in[c];

    // Compute convolution dot product: w0*h0 + w1*h1 + w2*h2 + w3*curr + bias
    float conv_val = h0 * w_c[0] + h1 * w_c[1] + h2 * w_c[2] + curr * w_c[3] + conv_bias[c];

    // SiLU activation: x / (1 + exp(-x))
    float sig = 1.0f / (1.0f + expf(-conv_val));
    float silu_val = conv_val * sig;

    // Shift history buffer: [h1, h2, curr]
    buf_c[0] = h1;
    buf_c[1] = h2;
    buf_c[2] = curr;

    // Output with residual connection
    x_out[c] = curr + silu_val;
}

extern "C" void launch_rada_deltanet_step(
    cudaStream_t stream,
    float* d_S,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    const float* d_alpha,
    const float* d_beta,
    float* d_y_out,
    int num_heads,
    int head_dim
) {
    if (!d_S || !d_q || !d_k || !d_v || !d_alpha || !d_beta || !d_y_out || num_heads <= 0 || head_dim != 64) return;
    rada_deltanet_step_kernel<<<num_heads, head_dim, 0, stream>>>(
        d_S, d_q, d_k, d_v, d_alpha, d_beta, d_y_out, num_heads, head_dim
    );
}

extern "C" void launch_rada_deltanet_conv1d_step(
    cudaStream_t stream,
    float* d_conv_buf,
    const float* d_x_in,
    const float* d_conv_weight,
    const float* d_conv_bias,
    float* d_x_out,
    int hidden_size,
    int kernel_size
) {
    if (hidden_size <= 0 || kernel_size != 4 || !d_conv_buf || !d_x_in || !d_conv_weight || !d_conv_bias || !d_x_out) return;
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    rada_deltanet_conv1d_step_kernel<<<blocks, threads, 0, stream>>>(
        d_conv_buf, d_x_in, d_conv_weight, d_conv_bias, d_x_out, hidden_size
    );
}
