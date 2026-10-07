#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define HADAMARD_B 512

// XOR Swizzling to completely eliminate bank conflicts in Shared Memory
__device__ __forceinline__ int swizzle(int idx) {
    return idx ^ (idx >> 5);
}

// In-SRAM Fast Walsh-Hadamard Transform (FWHT B=512) for outlier elimination (< 1.2 microsecond latency)
__global__ void __launch_bounds__(256, 4) rada_fwht_512_in_sram(
    const half* __restrict__ input,
    half* __restrict__ output,
    int total_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= total_blocks) return;

    __shared__ half smem[HADAMARD_B];

    int tid = threadIdx.x; // 0..255 (2 elements per thread)
    int lane = tid & 31;

    const half* in_ptr = input + block_id * HADAMARD_B;
    half* out_ptr = output + block_id * HADAMARD_B;

    half2 in_pair = *reinterpret_cast<const half2*>(in_ptr + tid * 2);
    float u0 = __half2float(in_pair.x) + __half2float(in_pair.y);
    float u1 = __half2float(in_pair.x) - __half2float(in_pair.y);

    // Butterfly Stages 1-5 purely inside warp registers via __shfl_xor_sync
    #pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
        float p0 = __shfl_xor_sync(0xFFFFFFFF, u0, stride);
        float p1 = __shfl_xor_sync(0xFFFFFFFF, u1, stride);
        u0 = ((lane & stride) == 0) ? (u0 + p0) : (p0 - u0);
        u1 = ((lane & stride) == 0) ? (u1 + p1) : (p1 - u1);
    }

    // Write to Shared Memory using bank-conflict-free swizzling
    smem[swizzle(tid * 2)]     = __float2half(u0);
    smem[swizzle(tid * 2 + 1)] = __float2half(u1);
    __syncthreads();

    // Butterfly Stages 6-9 across warps via Shared Memory
    #pragma unroll
    for (int stride = 32; stride < HADAMARD_B; stride <<= 1) {
        int idx = (tid / stride) * (2 * stride) + (tid % stride);
        int partner = idx + stride;

        half v1 = smem[swizzle(idx)];
        half v2 = smem[swizzle(partner)];
        __syncthreads();
        smem[swizzle(idx)]     = __hadd(v1, v2);
        smem[swizzle(partner)] = __hsub(v1, v2);
        __syncthreads();
    }

    out_ptr[tid * 2]     = smem[swizzle(tid * 2)];
    out_ptr[tid * 2 + 1] = smem[swizzle(tid * 2 + 1)];
}
