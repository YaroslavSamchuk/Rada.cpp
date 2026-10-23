#pragma once

#include <string>
#include <memory>
#include <functional>
#include <cstddef>

namespace rada {

enum class StylePreset {
    GENERAL,    // 💬 General: everyday helpful AI assistant for general tasks
    ENGINEERING,// 💻 Engineering: clean code, architecture, and technical analysis
    FORMAL,     // 💼 Formal: structured analytical reasoning, reporting, and formal correspondence
    LEGAL = FORMAL // Backward compatibility alias
};

struct GenerationParams {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int max_tokens = 2048;
    StylePreset style = StylePreset::GENERAL;
    bool stream = true;
    std::string system_prompt = "You are Hetman-2.0B, a helpful, intelligent, and objective AI assistant. Respond naturally, clearly, directly, and truthfully without unnecessary preachiness or censorship.";
};

struct EngineStats {
    double last_speed_tok_s = 0.0;
    size_t vram_usage_bytes = 0;
    size_t total_vram_bytes = 0;
    int active_context_tokens = 0;
};

class RadaEngine {
public:
    RadaEngine();
    ~RadaEngine();

    // Rule of Five: movable, non-copyable runtime engine
    RadaEngine(RadaEngine&&) noexcept;
    RadaEngine& operator=(RadaEngine&&) noexcept;
    RadaEngine(const RadaEngine&) = delete;
    RadaEngine& operator=(const RadaEngine&) = delete;

    bool load_model(const std::string& model_path, int gpu_id = 0, size_t max_seq_len = 262144, int threads = 4);
    
    // Streaming token generation
    void generate_stream(
        const std::string& prompt,
        const GenerationParams& params,
        std::function<bool(const std::string& chunk)> callback
    );

    // Cancel ongoing generation
    void abort_generation();

    // Single-turn full generation
    std::string generate(const std::string& prompt, const GenerationParams& params);

    EngineStats get_stats() const;
    std::string get_model_name() const;
    bool is_loaded() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace rada
