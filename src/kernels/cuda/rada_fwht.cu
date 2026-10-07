#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define HADAMARD_B 512
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define FWHT_NORM_SCALE 0.04419417382415922f // 1.0 / sqrt(512)

// Ultra-fast single-warp in-register Fast Walsh-Hadamard Transform (FWHT B=512)
// Completely eliminates Shared Memory, bank conflicts, and syncthreads barriers (< 40 ns latency)
__device__ __forceinline__ void warp_fwht_512(float* __restrict__ vals, int lane) {
    // Stage 0-4: Strides 1, 2, 4, 8, 16 across threads via warp shuffle XOR
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        int mask = 1 << s;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float partner = __shfl_xor_sync(0xFFFFFFFF, vals[k], mask);
            vals[k] = ((lane & mask) == 0) ? (vals[k] + partner) : (partner - vals[k]);
        }
    }

    // Stage 5-8: Strides 32, 64, 128, 256 within thread registers (zero communication)
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int k_stride = 1 << m;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            if ((k & k_stride) == 0) {
                float u = vals[k];
                float v = vals[k + k_stride];
                vals[k]            = u + v;
                vals[k + k_stride] = u - v;
            }
        }
    }
}

// Global kernel: 4 warps (128 threads) per block, each warp transforming one 512-element vector
__global__ void __launch_bounds__(128, 8) rada_fwht_512_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int total_blocks,
    bool normalize
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int chunk_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (chunk_id >= total_blocks) return;

    const half* in_ptr = input + chunk_id * HADAMARD_B;
    half* out_ptr = output + chunk_id * HADAMARD_B;

    float vals[16];

    // Vectorized coalesced loads: 16 elements per thread across 32 lanes
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        vals[k] = __half2float(in_ptr[k * WARP_SIZE + lane]);
    }

    // Execute 9-stage butterfly transform entirely in registers
    warp_fwht_512(vals, lane);

    float scale = normalize ? FWHT_NORM_SCALE : 1.0f;

    // Vectorized coalesced stores
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        out_ptr[k * WARP_SIZE + lane] = __float2half(vals[k] * scale);
    }
}
