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
              << "  -n, --max-tokens <num>  Maximum generated tokens [default: 2048]\n"
              << "  --temp <num>            Sampling temperature (0.0 - 1.5) [default: 0.7]\n"
              << "  --top-p <num>           Top-P nucleus sampling (0.0 - 1.0) [default: 0.9]\n"
              << "  -s, --style <style>     Response style preset: general | formal | tech [default: general]\n"
              << "  -p, --prompt <text>     Single-turn prompt (if omitted, launches interactive REPL)\n"
              << "  -h, --help              Display this help message\n";
}

#include <atomic>

#ifdef _WIN32
#include <windows.h>
static UINT g_orig_cp = CP_UTF8;
static UINT g_orig_output_cp = CP_UTF8;
static DWORD g_orig_mode = 0;
static bool g_mode_saved = false;
static rada::RadaEngine* g_engine = nullptr;
static std::atomic<bool> g_generating{false};

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT) {
        if (g_generating.load() && g_engine) {
            g_engine->abort_generation();
            std::cout << "\n\033[1;31m[Перервано користувачем]\033[0m\n";
            return TRUE; // Handled, gracefully return to REPL
        }
    }
    return FALSE;
}

void enable_windows_vt_mode() {
    g_orig_output_cp = GetConsoleOutputCP();
    g_orig_cp = GetConsoleCP();
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;
    if (GetConsoleMode(hOut, &g_orig_mode)) {
        g_mode_saved = true;
        DWORD dwMode = g_orig_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
}

void restore_windows_console() {
    SetConsoleOutputCP(g_orig_output_cp);
    SetConsoleCP(g_orig_cp);
    if (g_mode_saved) {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            SetConsoleMode(hOut, g_orig_mode);
        }
    }
}
#else
static rada::RadaEngine* g_engine = nullptr;
static std::atomic<bool> g_generating{false};
void enable_windows_vt_mode() {}
void restore_windows_console() {}
#endif

class CliDeliberationStreamer {
private:
    bool in_think_block = false;
    std::string tag_buffer;
    std::chrono::steady_clock::time_point think_start;

    static size_t get_complete_utf8_len(const std::string& str) {
        if (str.empty()) return 0;
        size_t len = str.length();
        size_t back = 0;
        while (back < 4 && back < len) {
            unsigned char b = static_cast<unsigned char>(str[len - 1 - back]);
            if ((b & 0x80) == 0) {
                return len; // ASCII character at end
            }
            if ((b & 0xC0) == 0xC0) { // Lead byte
                size_t expected = 0;
                if ((b & 0xE0) == 0xC0) expected = 2;
                else if ((b & 0xF0) == 0xE0) expected = 3;
                else if ((b & 0xF8) == 0xF0) expected = 4;
                else return len;
                size_t have = back + 1;
                if (have < expected) {
                    return len - have; // Incomplete UTF-8 sequence at end
                } else {
                    return len;
                }
            }
            back++;
        }
        return len;
    }

public:
    bool on_chunk(const std::string& chunk) {
        tag_buffer += chunk;
        
        while (!tag_buffer.empty()) {
            if (!in_think_block) {
                size_t pos = tag_buffer.find("<think>");
                if (pos != std::string::npos) {
                    if (pos > 0) {
                        std::cout << tag_buffer.substr(0, pos) << std::flush;
                    }
                    in_think_block = true;
                    think_start = std::chrono::steady_clock::now();
                    std::cout << "\n\033[38;5;244m╭── 🧠 [Міркування над завданням] ────────────\033[0m\n\033[38;5;242m│ \033[0m";
                    tag_buffer = tag_buffer.substr(pos + 7);
                } else {
                    if (tag_buffer.length() < 7 && std::string("<think>").rfind(tag_buffer, 0) == 0) {
                        break;
                    }
                    size_t valid_len = get_complete_utf8_len(tag_buffer);
                    if (valid_len == 0 && !tag_buffer.empty()) {
                        break;
                    }
                    std::cout << tag_buffer.substr(0, valid_len) << std::flush;
                    tag_buffer = tag_buffer.substr(valid_len);
                    if (!tag_buffer.empty()) {
                        break;
                    }
                }
            } else {
                size_t pos = tag_buffer.find("</think>");
                if (pos != std::string::npos) {
                    std::string think_text = tag_buffer.substr(0, pos);
                    for (char c : think_text) {
                        std::cout << c;
                        if (c == '\n') std::cout << "\033[38;5;242m│ \033[0m";
                    }
                    in_think_block = false;
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - think_start
                    ).count();
                    std::cout << "\n\033[38;5;244m╰── ⚡ Завершено за " << duration << " мс ────────────────────\033[0m\n\n\033[0m";
                    tag_buffer = tag_buffer.substr(pos + 8);
                } else {
                    if (tag_buffer.length() < 8 && std::string("</think>").rfind(tag_buffer, 0) == 0) {
                        break;
                    }
                    size_t valid_len = get_complete_utf8_len(tag_buffer);
                    if (valid_len == 0 && !tag_buffer.empty()) {
                        break;
                    }
                    std::string to_print = tag_buffer.substr(0, valid_len);
                    for (char c : to_print) {
                        std::cout << c;
                        if (c == '\n') std::cout << "\033[38;5;242m│ \033[0m";
                    }
                    std::cout.flush();
                    tag_buffer = tag_buffer.substr(valid_len);
                    if (!tag_buffer.empty()) {
                        break;
                    }
                }
            }
        }
        return true;
    }

    void flush() {
        if (!tag_buffer.empty()) {
            std::cout << tag_buffer << std::flush;
            tag_buffer.clear();
        }
        if (in_think_block) {
            std::cout << "\n\033[38;5;244m╰───────────────────────────────────────────\033[0m\n\033[0m";
            in_think_block = false;
        }
    }
};

int main(int argc, char* argv[]) {
    enable_windows_vt_mode();

    std::string model_path = "models/hetman-2.0b-ternary.rada";
    size_t ctx_len = 262144;
    int threads = 4;
    int gpu_id = 0;
    int max_tokens = 2048;
    float temp = 0.7f;
    float top_p = 0.9f;
    std::string style_str = "general";
    std::string single_prompt = "";

    // Parse command-line flags with robust exception handling
    try {
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
            } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
                max_tokens = std::stoi(argv[++i]);
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
    } catch (const std::exception& e) {
        std::cerr << "\033[1;31m[ERROR]\033[0m Invalid command-line argument: " << e.what() << "\n";
        print_help();
        return 1;
    }

    rada::StylePreset current_style = rada::StylePreset::GENERAL;
    if (style_str == "formal" || style_str == "legal" || style_str == "business") current_style = rada::StylePreset::FORMAL;
    else if (style_str == "tech" || style_str == "code" || style_str == "engineering") current_style = rada::StylePreset::ENGINEERING;
    else if (style_str == "kozak" || style_str == "general" || style_str == "chat") current_style = rada::StylePreset::GENERAL;

    // Automatic fallback if default path not found
    std::ifstream test_f(model_path, std::ios::binary);
    if (!test_f.good()) {
        std::ifstream f_mini("hetman_mini.rada", std::ios::binary);
        if (f_mini.good()) model_path = "hetman_mini.rada";
        else {
            std::ifstream f_root("hetman-2.0b.rada", std::ios::binary);
            if (f_root.good()) model_path = "hetman-2.0b.rada";
        }
    }

    print_banner();

    std::cout << "\033[1;36m[RADA]\033[0m Loading model: " << model_path << "\n";
    std::cout << "\033[1;36m[RADA]\033[0m Context: " << ctx_len << " tokens | GPU: #" << gpu_id << " | Threads: " << threads << "\n";
    std::cout << "\033[1;36m[RADA]\033[0m Style: " << style_str << " | Max Tokens: " << max_tokens << " | Temp: " << temp << " | Top-P: " << top_p << "\n\n";

    rada::RadaEngine engine;
    g_engine = &engine;
    if (!engine.load_model(model_path, gpu_id, ctx_len, threads)) {
        std::cerr << "\033[1;31m[ERROR]\033[0m Failed to initialize model at " << model_path << "\n";
        restore_windows_console();
        return 1;
    }

    rada::GenerationParams gen_params;
    gen_params.temperature = temp;
    gen_params.top_p = top_p;
    gen_params.max_tokens = max_tokens;
    gen_params.style = current_style;

    // Single prompt mode
    if (!single_prompt.empty()) {
        std::cout << "\033[1;32mUser:\033[0m " << single_prompt << "\n";
        std::cout << "\033[1;33mHetman:\033[0m ";
        CliDeliberationStreamer streamer;
        g_generating.store(true);
        engine.generate_stream(single_prompt, gen_params, [&](const std::string& chunk) {
            return streamer.on_chunk(chunk);
        });
        g_generating.store(false);
        streamer.flush();
        std::cout << "\n\n";
        auto stats = engine.get_stats();
        std::cout << "\033[1;30m[Speed: " << std::fixed << std::setprecision(1) << stats.last_speed_tok_s 
                  << " tok/s | VRAM: " << (stats.vram_usage_bytes / (1024 * 1024)) << " MB]\033[0m\n";
        restore_windows_console();
        return 0;
    }

    // Interactive REPL chat session
    std::cout << "\033[1;32m[Interactive Session Online]\033[0m Enter your prompt below (type 'exit' or 'quit' to terminate):\n";
    std::cout << "Style: " << style_str << " (modes: /general, /tech, /formal)\n\n";

    std::string line;
    while (true) {
        std::cout << "\033[1;32mUser > \033[0m";
        if (!std::getline(std::cin, line)) break;
        if (line == "exit" || line == "quit") {
            std::cout << "\033[1;33m[Hetman]\033[0m До зустрічі!\n";
            break;
        }
        if (line.empty()) continue;

        if (line == "/general" || line == "/chat" || line == "/kozak") {
            gen_params.style = rada::StylePreset::GENERAL;
            std::cout << "\033[1;35m[Hetman] Звичайний режим асистента\033[0m\n\n";
            continue;
        } else if (line == "/formal" || line == "/business" || line == "/legal") {
            gen_params.style = rada::StylePreset::FORMAL;
            std::cout << "\033[1;35m[Hetman] Діловий / аналітичний режим\033[0m\n\n";
            continue;
        } else if (line == "/tech" || line == "/code") {
            gen_params.style = rada::StylePreset::ENGINEERING;
            std::cout << "\033[1;35m[Hetman] Режим програмування та інженерії\033[0m\n\n";
            continue;
        }

        std::cout << "\033[1;33mHetman > \033[0m";
        CliDeliberationStreamer repl_streamer;
        g_generating.store(true);
        engine.generate_stream(line, gen_params, [&](const std::string& chunk) {
            return repl_streamer.on_chunk(chunk);
        });
        g_generating.store(false);
        repl_streamer.flush();
        std::cout << "\n";
        auto stats = engine.get_stats();
        std::cout << "\033[1;30m(" << std::fixed << std::setprecision(1) << stats.last_speed_tok_s 
                  << " tok/s | VRAM: " << (stats.vram_usage_bytes / (1024 * 1024)) << " MB)\033[0m\n\n";
    }

    restore_windows_console();
    return 0;
}
