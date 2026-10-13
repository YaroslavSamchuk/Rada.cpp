#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#else
// Forward declarations for host C++ compilation
typedef void* cudaStream_t;
#ifndef CUDA_FP16_H
struct half { uint16_t x; };
#define CUDA_FP16_H
#endif
#endif

struct HopfieldSlotRef {
    uint32_t group_idx;
    uint32_t u_idx;
    uint32_t v_idx;
};

extern "C" {

// 1. BitNet b1.58 Ternary Matrix-Vector Multiplication Kernel (DP4A accelerated)
void launch_rada_ternary_gemv(
    cudaStream_t stream,
    const int8_t* d_X,
    const uint32_t* d_W_packed,
    const float* d_scale_w,
    float scale_x,
    float* d_Y,
    int M, int K, int N
);

// 2. Fast Walsh-Hadamard Transform 512-Block Kernel (In-Register Warp-Shuffle)
void launch_rada_fwht_512(
    cudaStream_t stream,
    const half* d_input,
    half* d_output,
    int total_blocks,
    bool normalize
);

// 3. Speculative Latent-Hopfield Core L2 Prefetch & Gather Kernels
void launch_rada_hopfield_prefetch(
    cudaStream_t stream,
    const HopfieldSlotRef* d_active_slots,
    const uint32_t* d_V_packed,
    int num_slots
);

void launch_rada_hopfield_gather(
    cudaStream_t stream,
    const HopfieldSlotRef* d_active_slots,
    const uint32_t* d_V_packed,
    uint32_t* d_staging_buffer,
    int num_slots
);

// 4. Gated DeltaNet Stepwise Decode Recurrent Attention Kernel
void launch_rada_deltanet_step(
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
);

// 5. Causal 1D Depthwise Convolution Step Kernel (k=4)
void launch_rada_deltanet_conv1d_step(
    cudaStream_t stream,
    float* d_conv_buf,          // [hidden_size, 3] circular buffer
    const float* d_x_in,        // [hidden_size] input token vector
    const float* d_conv_weight, // [hidden_size, 4] depthwise weight
    const float* d_conv_bias,   // [hidden_size] bias
    float* d_x_out,             // [hidden_size] output vector
    int hidden_size,
    int kernel_size
);

// 6. Batched LM Head Evaluation for MTP Speculative Drafting (M=2)
void launch_rada_batched_lm_head(
    cudaStream_t stream,
    const float* d_h_candidates, // [2, hidden_size] (primary h_t+1 and draft h_t+2)
    const half* d_lm_head_w,     // [vocab_size, hidden_size] tied embeddings
    float* d_logits_out,         // [2, vocab_size]
    int hidden_size,
    int vocab_size,
    float logit_softcap
);

} // extern "C"
