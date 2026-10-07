#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

namespace rada {

enum class StylePreset {
    KOZAK,      // ⚔️ Козацький: колоритний, побратимський, соборний дух
    LEGAL,      // ⚖️ Діловий / Державний: строгий юридичний аналіз законів України
    ENGINEERING // 💻 Інженерний: точний код, математика, мінімум слів
};

struct GenerationParams {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int max_tokens = 2048;
    StylePreset style = StylePreset::KOZAK;
    bool stream = true;
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

    bool load_model(const std::string& model_path, int gpu_id = 0, size_t max_seq_len = 262144);
    
    // Потокова генерація токенів
    void generate_stream(
        const std::string& prompt,
        const GenerationParams& params,
        std::function<bool(const std::string& chunk)> callback
    );

    // Одноразова повна генерація
    std::string generate(const std::string& prompt, const GenerationParams& params);

    EngineStats get_stats() const;
    std::string get_model_name() const;
    bool is_loaded() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace rada
