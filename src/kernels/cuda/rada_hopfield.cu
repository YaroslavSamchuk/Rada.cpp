#include <cuda_runtime.h>
#include <cstdint>

struct HopfieldSlotRef {
    uint32_t group_idx;
    uint32_t u_idx;
    uint32_t v_idx;
};

// Speculative Asynchronous L2 Prefetch Kernel:
// Dispatched during Stage 2 computation to pull Top-32 Hopfield memory vectors (16 KB)
// from DRAM directly into the GPU L2 cache ahead of Stage 3 execution, completely hiding latency
__global__ void __launch_bounds__(64, 4) rada_hopfield_prefetch_l2(
    const HopfieldSlotRef* __restrict__ active_slots, // [1024 slots: 32 minigroups * 32 active pairs]
    const uint32_t* __restrict__ V_packed,           // [536M ternary weights packed into 105 MB]
    int num_slots
) {
    int slot_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_id >= num_slots) return;

    HopfieldSlotRef slot = active_slots[slot_id];
    size_t offset = (((size_t)slot.group_idx * 512 + slot.u_idx) * 512 + slot.v_idx) * 4;

    const uint4* src_ptr = reinterpret_cast<const uint4*>(V_packed + offset);

    // Issue cache line prefetch hints to memory subsystem
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(src_ptr));
    #else
    // Volatile read hint for Turing SM 7.5
    asm volatile("" : : "r"(src_ptr) : "memory");
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
    size_t offset = (((size_t)slot.group_idx * 512 + slot.u_idx) * 512 + slot.v_idx) * 4;

    const uint4* src_ptr = reinterpret_cast<const uint4*>(V_packed + offset);
    uint4* dst_ptr = reinterpret_cast<uint4*>(staging_buffer + (size_t)slot_id * 4);

    // 128-bit vectorized transfer
    *dst_ptr = *src_ptr;
}
