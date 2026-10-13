#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32
#define TILE_WORDS 32 // 32 int4s = 512 int8 elements = 512 bytes of shared memory (tiles K=1536 and K=6144 perfectly)

// Fast branchless 2-bit ternary unpacking: {00: 0, 01: +1, 10: -1, 11: 0}
__device__ __forceinline__ int32_t unpack_ternary_byte(uint32_t b) {
    int32_t p = (b & 0x55);        // positive flags
    int32_t n = ((b >> 1) & 0x55); // negative flags
    int32_t v0 = (p & 1) - (n & 1);
    int32_t v1 = ((p >> 2) & 1) - ((n >> 2) & 1);
    int32_t v2 = ((p >> 4) & 1) - ((n >> 4) & 1);
    int32_t v3 = ((p >> 6) & 1) - ((n >> 6) & 1);
    return static_cast<int32_t>(
        (static_cast<uint32_t>(v0 & 0xFF)) |
        (static_cast<uint32_t>(v1 & 0xFF) << 8) |
        (static_cast<uint32_t>(v2 & 0xFF) << 16) |
        (static_cast<uint32_t>(v3 & 0xFF) << 24)
    );
}

__device__ __forceinline__ int32_t unpack_and_dp4a(uint32_t byte_val, int32_t x_int8x4, int32_t acc) {
    int32_t w_signed = unpack_ternary_byte(byte_val);
    return __dp4a(x_int8x4, w_signed, acc);
}

// Bit-Parallel Ternary GEMV kernel with Shared Memory Tiling for X:
// 4 warps per block (128 threads), 1 warp per output row, vectorized 128-bit loads, DP4A dot-product.
// Vector X is cooperatively loaded into shared memory once, saving 4x memory bandwidth across warps!
__global__ void __launch_bounds__(128, 4) rada_ternary_gemv_kernel(
    const int8_t* __restrict__ X,           // [K] INT8 quantized activations
    const uint32_t* __restrict__ W_packed,  // [N, K / 16] packed 2-bit ternary weights
    float* __restrict__ Y,                  // [N] FP32 output activations
    const float scale_x,
    const float* __restrict__ scale_w,      // [N] per-row weight scales
    int K, int N
) {
    __shared__ int4 s_X[TILE_WORDS];

    int warp_id = threadIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * WARP_SIZE + lane; // 0..127
    int row = blockIdx.x * blockDim.y + warp_id;

    int32_t local_acc = 0;
    int K_words = K / 16;
    const uint32_t* row_w = W_packed + (size_t)row * K_words;

    int k_step = TILE_WORDS;
    int k_aligned_limit = (K_words / k_step) * k_step;

    for (int tile_k = 0; tile_k < k_aligned_limit; tile_k += k_step) {
        // 1. Cooperative load of activation tile into Shared Memory (32 words)
        if (tid < TILE_WORDS) {
            int k_word = tile_k + tid;
            s_X[tid] = *reinterpret_cast<const int4*>(X + k_word * 16);
        }
        __syncthreads();

        // 2. Each active warp consumes the shared tile for its assigned row
        if (row < N) {
            uint32_t w32 = row_w[tile_k + lane];
            int4 x4 = s_X[lane];

            local_acc = unpack_and_dp4a(w32 & 0xFF,         x4.x, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 8) & 0xFF,  x4.y, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 16) & 0xFF, x4.z, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 24) & 0xFF, x4.w, local_acc);
        }
        __syncthreads();
    }

    if (row >= N) return;

    // 3. Residual tail words
    for (int k_idx = k_aligned_limit + lane; k_idx < K_words; k_idx += WARP_SIZE) {
        uint32_t w32 = row_w[k_idx];
        int base_k = k_idx * 16;
        int4 x4 = *reinterpret_cast<const int4*>(X + base_k);

        local_acc = unpack_and_dp4a(w32 & 0xFF,         x4.x, local_acc);
        local_acc = unpack_and_dp4a((w32 >> 8) & 0xFF,  x4.y, local_acc);
        local_acc = unpack_and_dp4a((w32 >> 16) & 0xFF, x4.z, local_acc);
        local_acc = unpack_and_dp4a((w32 >> 24) & 0xFF, x4.w, local_acc);
    }

    // 4. Warp-level butterfly reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_acc += __shfl_down_sync(0xFFFFFFFF, local_acc, offset);
    }

    if (lane == 0) {
        Y[row] = static_cast<float>(local_acc) * (scale_x * scale_w[row]);
    }
}

// Batched LM Head Kernel for MTP Speculative Drafting (M=2)
// Evaluates candidate 1 (t+1) and candidate 2 (t+2) in a single pass over LM Head weights (saves 201 MB DRAM read)
__global__ void __launch_bounds__(128, 4) rada_batched_lm_head_kernel(
    const float* __restrict__ h_candidates, // [2, hidden_size]
    const half* __restrict__ lm_head_w,     // [vocab_size, hidden_size]
    float* __restrict__ logits_out,         // [2, vocab_size]
    int hidden_size,
    int vocab_size,
    float logit_softcap
) {
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;
    int vocab_idx = blockIdx.x * blockDim.y + warp_id;
    if (vocab_idx >= vocab_size) return;

    const half* row_w = lm_head_w + (size_t)vocab_idx * hidden_size;
    const float* h0 = h_candidates;
    const float* h1 = h_candidates + hidden_size;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int i = lane; i < hidden_size; i += WARP_SIZE) {
        float w_val = __half2float(row_w[i]);
        acc0 += w_val * h0[i];
        acc1 += w_val * h1[i];
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (lane == 0) {
        if (logit_softcap > 0.0f) {
            acc0 = logit_softcap * tanhf(acc0 / logit_softcap);
            acc1 = logit_softcap * tanhf(acc1 / logit_softcap);
        }
        logits_out[vocab_idx] = acc0;
        logits_out[(size_t)vocab_size + vocab_idx] = acc1;
    }
}

extern "C" void launch_rada_ternary_gemv(
    cudaStream_t stream,
    const int8_t* d_X,
    const uint32_t* d_W_packed,
    const float* d_scale_w,
    float scale_x,
    float* d_Y,
    int M, int K, int N
) {
    if (K <= 0 || N <= 0 || (K % 16 != 0) || !d_X || !d_W_packed || !d_Y || !d_scale_w) return;
    if ((reinterpret_cast<uintptr_t>(d_X) & 0xF) != 0) return;
    dim3 block(WARP_SIZE, 4);
    dim3 grid((N + block.y - 1) / block.y);
    rada_ternary_gemv_kernel<<<grid, block, 0, stream>>>(d_X, d_W_packed, d_Y, scale_x, d_scale_w, K, N);
}

extern "C" void launch_rada_batched_lm_head(
    cudaStream_t stream,
    const float* d_h_candidates,
    const half* d_lm_head_w,
    float* d_logits_out,
    int hidden_size,
    int vocab_size,
    float logit_softcap
) {
    if (hidden_size <= 0 || vocab_size <= 0 || !d_h_candidates || !d_lm_head_w || !d_logits_out) return;
    dim3 block(WARP_SIZE, 4);
    dim3 grid((vocab_size + block.y - 1) / block.y);
    rada_batched_lm_head_kernel<<<grid, block, 0, stream>>>(d_h_candidates, d_lm_head_w, d_logits_out, hidden_size, vocab_size, logit_softcap);
}
