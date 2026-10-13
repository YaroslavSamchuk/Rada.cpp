#pragma once

#include <cstdint>
#include <cstddef>

namespace rada {

// Magic bytes: "RADA" in little-endian ('R'=0x52, 'A'=0x41, 'D'=0x44, 'A'=0x41)
constexpr uint32_t RADA_MAGIC = 0x41444152;
constexpr uint32_t RADA_VERSION = 1;
constexpr size_t RADA_ALIGNMENT = 64; // 64-byte alignment for AVX-512 and CUDA coalesced access

// Canonical Hetman-2.0B Special Token IDs matching SentencePiece / ByteLevel BPE
constexpr uint32_t RADA_TOKEN_UNK         = 0;
constexpr uint32_t RADA_TOKEN_BOS         = 1; // <s>
constexpr uint32_t RADA_TOKEN_EOS         = 2; // </s>
constexpr uint32_t RADA_TOKEN_PAD         = 3; // <pad>
constexpr uint32_t RADA_TOKEN_IM_START    = 4; // <|im_start|>
constexpr uint32_t RADA_TOKEN_IM_END      = 5; // <|im_end|>
constexpr uint32_t RADA_TOKEN_THINK       = 6; // <think>
constexpr uint32_t RADA_TOKEN_THINK_END   = 7; // </think>
constexpr uint32_t RADA_TOKEN_THOUGHT_SUMMARY     = 8;  // <thought_summary>
constexpr uint32_t RADA_TOKEN_THOUGHT_SUMMARY_END = 9;  // </thought_summary>
constexpr uint32_t RADA_TOKEN_TOOL_CALL           = 10; // <|call|>
constexpr uint32_t RADA_TOKEN_TOOL_RETURN         = 11; // <|return|>
constexpr uint32_t RADA_TOKEN_EXTRA_0             = 12; // <|extra_0|>
constexpr uint32_t RADA_TOKEN_EXTRA_1             = 13; // <|extra_1|>
constexpr uint32_t RADA_TOKEN_EXTRA_2             = 14; // <|extra_2|>
constexpr uint32_t RADA_TOKEN_EXTRA_3             = 15; // <|extra_3|>


enum class TensorDtype : uint32_t {
    BF16 = 0,
    FP16 = 1,
    FP32 = 2,
    INT8 = 3,
    TERNARY_2BIT = 4 // Packed 2-bit ternary: 00 = 0, 01 = +1, 10 = -1 (16 weights per uint32_t)
};

#pragma pack(push, 1)

struct RadaHeader {
    uint32_t magic;                  // RADA_MAGIC (0x41444152)
    uint32_t version;                // RADA_VERSION (1)
    uint32_t vocab_size;             // 65536
    uint32_t hidden_size;            // 1536
    uint32_t intermediate_size;      // 6144
    uint32_t num_layers;             // 34
    uint32_t num_deltanet_layers;    // 28
    uint32_t num_global_layers;      // 6
    uint32_t num_heads;              // 16
    uint32_t num_kv_heads;           // 2
    uint32_t head_dim;               // 64
    uint32_t global_head_dim;        // 128
    uint32_t global_heads;           // 12
    uint32_t hopfield_groups;        // 16 (SLH-Core v2)
    uint32_t hopfield_codebook_size; // 512
    uint32_t hopfield_val_dim;       // 128 (SLH-Core v2)
    uint32_t max_seq_len;            // 262144 (256k context)
    uint32_t num_tensors;            // Total count of tensors in model
    uint64_t tensor_table_offset;    // Byte offset to array of RadaTensorDescriptor
    uint64_t tensor_data_offset;     // Byte offset to aligned tensor data payload
    uint64_t vocab_offset;           // Byte offset to embedded vocabulary / token table (or 0 if external)
    uint64_t vocab_size_bytes;       // Byte size of embedded vocabulary data
    uint64_t merges_offset;          // Byte offset to BPE merges table (or 0 if none)
    uint64_t merges_size_bytes;      // Byte size of BPE merges table
    uint8_t  reserved[8];            // Padding to align header to 128 bytes
};

struct RadaTensorDescriptor {
    char     name[64];               // Null-terminated tensor identifier (e.g., "stem.0.attn.q_proj.weight")
    uint32_t ndims;                  // Number of dimensions (1 to 4)
    uint32_t dims[4];                // Dimensions [dim0, dim1, dim2, dim3]
    uint32_t dtype;                  // TensorDtype enum
    float    scale_gamma;            // Scaling factor γ for 1.58-bit ternary dequantization
    uint8_t  reserved_align[4];      // Natural 8-byte alignment pad for uint64_t fields
    uint64_t offset;                 // Absolute file offset (aligned to 64 bytes)
    uint64_t size_bytes;             // Payload byte size
    uint32_t checksum;               // CRC32 checksum of tensor data
    uint8_t  padding[12];            // Padding to ensure exactly 128 bytes per descriptor
};

#pragma pack(pop)

static_assert(sizeof(RadaHeader) == 128, "RadaHeader must be exactly 128 bytes");
static_assert(sizeof(RadaTensorDescriptor) == 128, "RadaTensorDescriptor must be exactly 128 bytes");

} // namespace rada
