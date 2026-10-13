#include <cuda_runtime.h>
#include <cstdint>
#include "rada_kernels.h"

#define HOPFIELD_CODEBOOK_SIZE 512
#define HOPFIELD_WORDS_PER_SLOT 8 // 128 val_dim * 2 bits / 32 bits = 8 uint32 words per slot

// Speculative Asynchronous L2 Prefetch Kernel (SLH-Core v2: 128 val_dim = 8 uint32s per slot):
// Dispatched during Stage 2 computation to pull Top-32 Hopfield memory vectors (16 KB)
// from DRAM directly into the GPU L2 cache ahead of execution, completely hiding latency
__global__ void __launch_bounds__(64, 4) rada_hopfield_prefetch_l2(
    const HopfieldSlotRef* __restrict__ active_slots, // [512 slots: 16 groups * 32 active pairs]
    const uint32_t* __restrict__ V_packed,           // [536M ternary weights packed into 107 MB]
    int num_slots
) {
    int slot_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_id >= num_slots) return;

    HopfieldSlotRef slot = active_slots[slot_id];
    if (slot.group_idx >= 16 || slot.u_idx >= HOPFIELD_CODEBOOK_SIZE || slot.v_idx >= HOPFIELD_CODEBOOK_SIZE) return;
    size_t offset = (((size_t)slot.group_idx * HOPFIELD_CODEBOOK_SIZE + slot.u_idx) * HOPFIELD_CODEBOOK_SIZE + slot.v_idx) * HOPFIELD_WORDS_PER_SLOT;

    const uint4* src_ptr0 = reinterpret_cast<const uint4*>(V_packed + offset);
    const uint4* src_ptr1 = reinterpret_cast<const uint4*>(V_packed + offset + 4);

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(src_ptr0));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(src_ptr1));
    #else
    asm volatile("" : : "l"(src_ptr0) : "memory");
    asm volatile("" : : "l"(src_ptr1) : "memory");
    #endif
}

// Staging copy kernel: gathers prefetched Top-32 slots into contiguous workspace buffer
__global__ void __launch_bounds__(64, 4) rada_hopfield_gather_active_slots(
    const HopfieldSlotRef* __restrict__ active_slots,
    const uint32_t* __restrict__ V_packed,
    uint32_t* __restrict__ staging_buffer,
    int num_slots
) {
    int slot_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_id >= num_slots) return;

    HopfieldSlotRef slot = active_slots[slot_id];
    if (slot.group_idx >= 16 || slot.u_idx >= HOPFIELD_CODEBOOK_SIZE || slot.v_idx >= HOPFIELD_CODEBOOK_SIZE) return;
    size_t offset = (((size_t)slot.group_idx * HOPFIELD_CODEBOOK_SIZE + slot.u_idx) * HOPFIELD_CODEBOOK_SIZE + slot.v_idx) * HOPFIELD_WORDS_PER_SLOT;

    const uint4* src_ptr0 = reinterpret_cast<const uint4*>(V_packed + offset);
    const uint4* src_ptr1 = reinterpret_cast<const uint4*>(V_packed + offset + 4);

    uint4* dst_ptr0 = reinterpret_cast<uint4*>(staging_buffer + (size_t)slot_id * HOPFIELD_WORDS_PER_SLOT);
    uint4* dst_ptr1 = reinterpret_cast<uint4*>(staging_buffer + (size_t)slot_id * HOPFIELD_WORDS_PER_SLOT + 4);

    // Vectorized 2 x 128-bit transfers
    *dst_ptr0 = *src_ptr0;
    *dst_ptr1 = *src_ptr1;
}

extern "C" void launch_rada_hopfield_prefetch(
    cudaStream_t stream,
    const HopfieldSlotRef* d_active_slots,
    const uint32_t* d_V_packed,
    int num_slots
) {
    if (num_slots <= 0 || !d_active_slots || !d_V_packed) return;
    if ((reinterpret_cast<uintptr_t>(d_V_packed) & 0xF) != 0) return;
    int threads = 64;
    int blocks = (num_slots + threads - 1) / threads;
    rada_hopfield_prefetch_l2<<<blocks, threads, 0, stream>>>(d_active_slots, d_V_packed, num_slots);
}

extern "C" void launch_rada_hopfield_gather(
    cudaStream_t stream,
    const HopfieldSlotRef* d_active_slots,
    const uint32_t* d_V_packed,
    uint32_t* d_staging_buffer,
    int num_slots
) {
    if (num_slots <= 0 || !d_active_slots || !d_V_packed || !d_staging_buffer) return;
    if ((reinterpret_cast<uintptr_t>(d_staging_buffer) & 0xF) != 0 || (reinterpret_cast<uintptr_t>(d_V_packed) & 0xF) != 0) return;
    int threads = 64;
    int blocks = (num_slots + threads - 1) / threads;
    rada_hopfield_gather_active_slots<<<blocks, threads, 0, stream>>>(d_active_slots, d_V_packed, d_staging_buffer, num_slots);
}
