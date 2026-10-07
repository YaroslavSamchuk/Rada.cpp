#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32

// Unpack 4 ternary 2-bit weights into signed 8-bit integers and compute 4-element dot product with INT8 activations using DP4A
__device__ __forceinline__ int32_t unpack_and_dp4a(uint32_t w_2bit_4elem, int32_t x_int8_4elem, int32_t acc) {
    int32_t w_signed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t bits = (w_2bit_4elem >> (i * 2)) & 0x3;
        int8_t val = (bits == 1) ? 1 : ((bits == 2) ? -1 : 0);
        w_signed |= (((int32_t)val & 0xFF) << (i * 8));
    }
    return __dp4a(x_int8_4elem, w_signed, acc);
}

// Bit-Parallel Ternary GEMV kernel: Multiplies INT8 activation vector X by 1.58-bit packed ternary weight matrix W
__global__ void __launch_bounds__(128, 4) rada_ternary_gemv_kernel(
    const int8_t* __restrict__ X,           // [K] INT8 quantized activations
    const uint32_t* __restrict__ W_packed,  // [N, K / 16] packed 2-bit ternary weights
    float* __restrict__ Y,                  // [N] FP32/FP16 output activations
    const float scale_x,
    const float* __restrict__ scale_w,      // [N] per-row weight scales
    int K, int N
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x;
    int32_t local_acc = 0;
    int K_words = K / 16;
    const uint32_t* row_w = W_packed + row * K_words;

    // Vectorized 128-bit loads (uint4 = 4 uint32 = 64 weights) for maximum memory bus saturation
    for (int k_idx = lane * 4; k_idx < K_words; k_idx += WARP_SIZE * 4) {
        uint4 packed_w4 = *reinterpret_cast<const uint4*>(row_w + k_idx);

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            uint32_t w32 = (sub == 0) ? packed_w4.x : ((sub == 1) ? packed_w4.y : ((sub == 2) ? packed_w4.z : packed_w4.w));
            int base_k = (k_idx + sub) * 16;
            int4 x4 = *reinterpret_cast<const int4*>(X + base_k);

            local_acc = unpack_and_dp4a(w32 & 0xFF, x4.x, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 8) & 0xFF, x4.y, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 16) & 0xFF, x4.z, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 24) & 0xFF, x4.w, local_acc);
        }
    }

    // Warp-level reduction without shared memory
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_acc += __shfl_down_sync(0xFFFFFFFF, local_acc, offset);
    }

    // Write result
    if (lane == 0) {
        Y[row] = (float)local_acc * (scale_x * scale_w[row]);
    }
}
