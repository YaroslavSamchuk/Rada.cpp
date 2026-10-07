#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32

// Fast branchless 2-bit ternary unpacking: {00: 0, 01: +1, 10: -1, 11: 0}
// Uses (bit0 - bit1) identity: zero branching, minimal instruction overhead
__device__ __forceinline__ int32_t unpack_ternary_byte(uint32_t b) {
    int32_t p = (b & 0x55);        // positive flags
    int32_t n = ((b >> 1) & 0x55); // negative flags
    int32_t v0 = (p & 1) - (n & 1);
    int32_t v1 = ((p >> 2) & 1) - ((n >> 2) & 1);
    int32_t v2 = ((p >> 4) & 1) - ((n >> 4) & 1);
    int32_t v3 = ((p >> 6) & 1) - ((n >> 6) & 1);
    return (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24);
}

__device__ __forceinline__ int32_t unpack_and_dp4a(uint32_t byte_val, int32_t x_int8x4, int32_t acc) {
    int32_t w_signed = unpack_ternary_byte(byte_val);
    return __dp4a(x_int8x4, w_signed, acc);
}

// Bit-Parallel Ternary GEMV kernel:
// 1 warp per output row, vectorized 128-bit coalesced memory transfers, DP4A dot-product
__global__ void __launch_bounds__(128, 4) rada_ternary_gemv_kernel(
    const int8_t* __restrict__ X,           // [K] INT8 quantized activations
    const uint32_t* __restrict__ W_packed,  // [N, K / 16] packed 2-bit ternary weights
    float* __restrict__ Y,                  // [N] FP32 output activations
    const float scale_x,
    const float* __restrict__ scale_w,      // [N] per-row weight scales
    int K, int N
) {
    // 2D thread block: threadIdx.x is lane (0..31), threadIdx.y is warp inside block (0..3)
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;
    int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= N) return;

    int32_t local_acc = 0;
    int K_words = K / 16;
    const uint32_t* row_w = W_packed + (size_t)row * K_words;

    // Vectorized 128-bit loads (uint4 = 4 uint32 = 64 ternary weights)
    // Coalesced loop: all 32 threads in the warp advance together by 128 words
    int k_step = WARP_SIZE * 4;
    int k_aligned_limit = (K_words / k_step) * k_step;

    for (int k_idx = lane * 4; k_idx < k_aligned_limit; k_idx += k_step) {
        uint4 packed_w4 = *reinterpret_cast<const uint4*>(row_w + k_idx);

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            uint32_t w32 = (sub == 0) ? packed_w4.x : ((sub == 1) ? packed_w4.y : ((sub == 2) ? packed_w4.z : packed_w4.w));
            int base_k = (k_idx + sub) * 16;
            int4 x4 = *reinterpret_cast<const int4*>(X + base_k);

            local_acc = unpack_and_dp4a(w32 & 0xFF,         x4.x, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 8) & 0xFF,  x4.y, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 16) & 0xFF, x4.z, local_acc);
            local_acc = unpack_and_dp4a((w32 >> 24) & 0xFF, x4.w, local_acc);
        }
    }

    // Residual cleanup loop for tail words when K is not a multiple of 2048
    for (int k_idx = k_aligned_limit + lane; k_idx < K_words; k_idx += WARP_SIZE) {
        uint32_t w32 = row_w[k_idx];
        int base_k = k_idx * 16;
        int4 x4 = *reinterpret_cast<const int4*>(X + base_k);

        local_acc = unpack_and_dp4a(w32 & 0xFF,         x4.x, local_acc);
        local_acc = unpack_and_dp4a((w32 >> 8) & 0xFF,  x4.y, local_acc);
        local_acc = unpack_and_dp4a((w32 >> 16) & 0xFF, x4.z, local_acc);
        local_acc = unpack_and_dp4a((w32 >> 24) & 0xFF, x4.w, local_acc);
    }

    // Warp-level butterfly reduction without shared memory
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_acc += __shfl_down_sync(0xFFFFFFFF, local_acc, offset);
    }

    // Lane 0 writes normalized result
    if (lane == 0) {
        Y[row] = static_cast<float>(local_acc) * (scale_x * scale_w[row]);
    }
}
