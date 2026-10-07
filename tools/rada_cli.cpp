#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "rada_engine.hpp"

void print_banner() {
    std::cout << "\033[1;33m"
              << "=======================================================================\n"
              << "     RADA.CPP: HIGH-PERFORMANCE INFERENCE RUNTIME FOR HETMAN-2.0B      \n"
              << "      (Cossack Rada | Pure C++20/CUDA | RTX 2060+ >= 6 GB VRAM)        \n"
              << "=======================================================================\n"
              << "\033[0m";
}

void print_help() {
    std::cout << "Usage: rada_cli [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>      Path to model file (.rada / .gguf) [default: hetman-2.0b.rada]\n"
              << "  -c, --ctx <num>         Context window size (up to 262144) [default: 262144]\n"
              << "  -t, --threads <num>     Number of CPU fallback threads [default: 4]\n"
              << "  -g, --gpu <id>          Target GPU device ID [default: 0]\n"
              << "  --temp <num>            Sampling temperature (0.0 - 1.5) [default: 0.7]\n"
              << "  --top-p <num>           Top-P nucleus sampling (0.0 - 1.0) [default: 0.9]\n"
              << "  -s, --style <style>     Response style preset: kozak | legal | tech [default: kozak]\n"
              << "  -p, --prompt <text>     Single-turn prompt (if omitted, launches interactive REPL)\n"
              << "  -h, --help              Display this help message\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/hetman-2.0b-ternary.rada";
    size_t ctx_len = 262144;
    int threads = 4;
    int gpu_id = 0;
    float temp = 0.7f;
    float top_p = 0.9f;
    std::string style_str = "kozak";
    std::string single_prompt = "";

    // Parse command-line flags
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) {
            ctx_len = std::stoull(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if ((arg == "-s" || arg == "--style") && i + 1 < argc) {
            style_str = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_banner();
            print_help();
            return 0;
        }
    }

    rada::StylePreset current_style = rada::StylePreset::KOZAK;
    if (style_str == "legal") current_style = rada::StylePreset::LEGAL;
    else if (style_str == "tech") current_style = rada::StylePreset::ENGINEERING;

    print_banner();

    std::cout << "\033[1;36m[RADA]\033[0m Loading model: " << model_path << "\n";
    std::cout << "\033[1;36m[RADA]\033[0m Context: " << ctx_len << " tokens | GPU: #" << gpu_id << " | Threads: " << threads << "\n";
    std::cout << "\033[1;36m[RADA]\033[0m Style: " << style_str << " | Temp: " << temp << " | Top-P: " << top_p << "\n\n";

    rada::RadaEngine engine;
    if (!engine.load_model(model_path, gpu_id, ctx_len)) {
        std::cerr << "\033[1;31m[ERROR]\033[0m Failed to initialize model at " << model_path << "\n";
        return 1;
    }

    rada::GenerationParams gen_params;
    gen_params.temperature = temp;
    gen_params.top_p = top_p;
    gen_params.style = current_style;

    // Single prompt mode
    if (!single_prompt.empty()) {
        std::cout << "\033[1;32mUser:\033[0m " << single_prompt << "\n";
        std::cout << "\033[1;33mHetman:\033[0m ";
        engine.generate_stream(single_prompt, gen_params, [](const std::string& chunk) {
            std::cout << chunk << std::flush;
            return true;
        });
        std::cout << "\n\n";
        auto stats = engine.get_stats();
        std::cout << "\033[1;30m[Speed: " << std::fixed << std::setprecision(1) << stats.last_speed_tok_s 
                  << " tok/s | VRAM: " << (stats.vram_usage_bytes / (1024 * 1024)) << " MB]\033[0m\n";
        return 0;
    }

    // Interactive REPL chat session
    std::cout << "\033[1;32m[Interactive Session Online]\033[0m Enter your prompt below (type 'exit' or 'quit' to terminate):\n";
    std::cout << "Style: " << style_str << " (switch styles by typing: /kozak, /legal, /tech)\n\n";

    std::string line;
    while (true) {
        std::cout << "\033[1;32mUser > \033[0m";
        if (!std::getline(std::cin, line)) break;
        if (line == "exit" || line == "quit") {
            std::cout << "\033[1;33m[RADA]\033[0m Concluding session. Farewell!\n";
            break;
        }
        if (line.empty()) continue;

        if (line == "/kozak") {
            gen_params.style = rada::StylePreset::KOZAK;
            std::cout << "\033[1;35m[RADA] Switched to Cossack / Deliberative Style\033[0m\n\n";
            continue;
        } else if (line == "/legal") {
            gen_params.style = rada::StylePreset::LEGAL;
            std::cout << "\033[1;35m[RADA] Switched to Legal / Statutory Style\033[0m\n\n";
            continue;
        } else if (line == "/tech") {
            gen_params.style = rada::StylePreset::ENGINEERING;
            std::cout << "\033[1;35m[RADA] Switched to Engineering / Technical Style\033[0m\n\n";
            continue;
        }

        std::cout << "\033[1;33mHetman > \033[0m";
        engine.generate_stream(line, gen_params, [](const std::string& chunk) {
            std::cout << chunk << std::flush;
            return true;
        });
        std::cout << "\n";
        auto stats = engine.get_stats();
        std::cout << "\033[1;30m(" << std::fixed << std::setprecision(1) << stats.last_speed_tok_s 
                  << " tok/s | VRAM: " << (stats.vram_usage_bytes / (1024 * 1024)) << " MB)\033[0m\n\n";
    }

    return 0;
}
