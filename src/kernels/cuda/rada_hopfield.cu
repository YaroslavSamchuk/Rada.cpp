#include <cuda_runtime.h>
#include <cstdint>

struct HopfieldSlotRef {
    uint32_t group_idx;
    uint32_t u_idx;
    uint32_t v_idx;
};

// Asynchronous DMA prefetching of active Top-32 Hopfield memory slots (16 KB) directly into L2 cache / SRAM
__global__ void __launch_bounds__(64, 4) rada_hopfield_prefetch_l2(
    const HopfieldSlotRef* __restrict__ active_slots, // [1024 slots total: 32 minigroups * 32 active pairs]
    const uint32_t* __restrict__ V_packed,           // [536M ternary weights packed into 105 MB]
    uint32_t* __restrict__ sram_staging_buffer       // [16 KB SRAM buffer]
) {
    int slot_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_id >= 1024) return;

    HopfieldSlotRef slot = active_slots[slot_id];
    size_t offset = (((size_t)slot.group_idx * 512 + slot.u_idx) * 512 + slot.v_idx) * 4;

    const uint4* src_ptr = reinterpret_cast<const uint4*>(V_packed + offset);
    uint4* dst_ptr = reinterpret_cast<uint4*>(sram_staging_buffer + slot_id * 4);

    #if __CUDA_ARCH__ >= 800
    asm volatile("prefetch.global.L2 [%0];" :: "l"(src_ptr));
    #endif

    *dst_ptr = *src_ptr;
}
