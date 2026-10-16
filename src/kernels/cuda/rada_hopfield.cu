#include <cuda_runtime.h>
#include <cstdint>
#include "rada_kernels.h"

#define HOPFIELD_CODEBOOK_SIZE 1024
#define HOPFIELD_WORDS_PER_SLOT 2 // 32 val_dim * 2 bits / 32 bits = 2 uint32 words per slot (8 bytes)

// Speculative Asynchronous L2 Prefetch Kernel (SLH-Core v2: 32 val_dim = 2 uint32s per slot):
// Dispatched during Stage 2 computation to pull Top-32 Hopfield memory vectors (4 KB)
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

    const uint2* src_ptr = reinterpret_cast<const uint2*>(V_packed + offset);

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(src_ptr));
    #else
    asm volatile("" : : "l"(src_ptr) : "memory");
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

    const uint2* src_ptr = reinterpret_cast<const uint2*>(V_packed + offset);
    uint2* dst_ptr = reinterpret_cast<uint2*>(staging_buffer + (size_t)slot_id * HOPFIELD_WORDS_PER_SLOT);

    // Vectorized 64-bit transfer (single atomic cycle)
    *dst_ptr = *src_ptr;
}

extern "C" void launch_rada_hopfield_prefetch(
    cudaStream_t stream,
    const HopfieldSlotRef* d_active_slots,
    const uint32_t* d_V_packed,
    int num_slots
) {
    if (num_slots <= 0 || !d_active_slots || !d_V_packed) return;
    if ((reinterpret_cast<uintptr_t>(d_V_packed) & 0x7) != 0) return;
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
    if ((reinterpret_cast<uintptr_t>(d_staging_buffer) & 0x7) != 0 || (reinterpret_cast<uintptr_t>(d_V_packed) & 0x7) != 0) return;
    int threads = 64;
    int blocks = (num_slots + threads - 1) / threads;
    rada_hopfield_gather_active_slots<<<blocks, threads, 0, stream>>>(d_active_slots, d_V_packed, d_staging_buffer, num_slots);
}
