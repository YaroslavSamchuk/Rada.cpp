#include "rada.h"
#include "rada_engine.hpp"
#include <string>
#include <exception>

struct rada_engine_t {
    rada::RadaEngine engine;
};

extern "C" {

rada_engine_t* rada_create(const char* model_path, int gpu_id) {
    if (!model_path) return nullptr;
    try {
        auto* wrapper = new rada_engine_t();
        if (!wrapper->engine.load_model(model_path, gpu_id)) {
            delete wrapper;
            return nullptr;
        }
        return wrapper;
    } catch (...) {
        return nullptr;
    }
}

void rada_destroy(rada_engine_t* engine) {
    if (!engine) return;
    try {
        delete engine;
    } catch (...) {
        // Do not propagate exceptions across C ABI boundary
    }
}

int rada_tokenize(rada_engine_t* /*engine*/, const char* text, uint32_t* tokens, int max_tokens) {
    if (!text || !tokens || max_tokens <= 0) return 0;
    // Basic whitespace tokenization stub
    int count = 0;
    const char* p = text;
    while (*p && count < max_tokens) {
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p) {
            tokens[count++] = static_cast<uint32_t>(count + 100);
            while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
        }
    }
    return count;
}

const char* rada_token_to_str(rada_engine_t* /*engine*/, uint32_t /*token_id*/) {
    static thread_local std::string s_token = " ";
    return s_token.c_str();
}

bool rada_generate_stream(
    rada_engine_t* engine,
    const char* prompt,
    float temperature,
    float top_p,
    int max_new_tokens,
    rada_stream_callback_t callback,
    void* user_data
) {
    if (!engine || !prompt || !callback) return false;
    try {
        rada::GenerationParams params;
        params.temperature = temperature;
        params.top_p = top_p;
        params.max_tokens = max_new_tokens;
        params.stream = true;

        engine->engine.generate_stream(prompt, params, [callback, user_data](const std::string& chunk) -> bool {
            return callback(chunk.c_str(), user_data);
        });
        return true;
    } catch (...) {
        return false;
    }
}

double rada_get_last_speed_tok_s(rada_engine_t* engine) {
    if (!engine) return 0.0;
    try {
        return engine->engine.get_stats().last_speed_tok_s;
    } catch (...) {
        return 0.0;
    }
}

size_t rada_get_vram_usage_bytes(rada_engine_t* engine) {
    if (!engine) return 0;
    try {
        return engine->engine.get_stats().vram_usage_bytes;
    } catch (...) {
        return 0;
    }
}

} // extern "C"
