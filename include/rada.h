#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct rada_engine_t rada_engine_t;

typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t total_layers;
    uint32_t deltanet_layers;
    uint32_t global_layers;
    uint32_t max_seq_len;
} rada_config_t;

// Lifecycle
rada_engine_t* rada_create(const char* model_path, int gpu_id);
void rada_destroy(rada_engine_t* engine);

// Tokenizer & Generation
int rada_tokenize(rada_engine_t* engine, const char* text, uint32_t* tokens, int max_tokens);
const char* rada_token_to_str(rada_engine_t* engine, uint32_t token_id);

typedef bool (*rada_stream_callback_t)(const char* token_str, void* user_data);

bool rada_generate_stream(
    rada_engine_t* engine,
    const char* prompt,
    float temperature,
    float top_p,
    int max_new_tokens,
    rada_stream_callback_t callback,
    void* user_data
);

// Performance Telemetry
double rada_get_last_speed_tok_s(rada_engine_t* engine);
size_t rada_get_vram_usage_bytes(rada_engine_t* engine);

#ifdef __cplusplus
}
#endif
