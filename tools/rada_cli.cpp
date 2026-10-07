#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "rada_engine.hpp"

void print_banner() {
    std::cout << "\033[1;33m"
              << "=======================================================================\n"
              << "        RADA.CPP: СУВЕРЕННИЙ РУШІЙ ДЛЯ МОДЕЛІ «ГЕТЬМАН-2.0B»          \n"
              << "      (Козацька Рада | Pure C++20/CUDA | RTX 2060+ >= 6 GB VRAM)      \n"
              << "=======================================================================\n"
              << "\033[0m";
}

void print_help() {
    std::cout << "Використання: rada_cli [опції]\n\n"
              << "Опції:\n"
              << "  -m, --model <шлях>      Шлях до файлу моделі (.rada / .gguf) [за замовч: hetman-2.0b.rada]\n"
              << "  -c, --ctx <число>       Розмір контекстного вікна (до 262144) [за замовч: 262144]\n"
              << "  -t, --threads <число>   Кількість потоків CPU для резервного бекенду [за замовч: 4]\n"
              << "  -g, --gpu <id>          Ідентифікатор відеокарти GPU [за замовч: 0]\n"
              << "  --temp <число>          Температура генерації (0.0 - 1.5) [за замовч: 0.7]\n"
              << "  --top-p <число>         Top-P семплінг (0.0 - 1.0) [за замовч: 0.9]\n"
              << "  -s, --style <стиль>     Стиль відповіді: kozak | legal | tech [за замовч: kozak]\n"
              << "  -p, --prompt <текст>    Одноразовий запит (якщо не вказано - запускається чат)\n"
              << "  -h, --help              Показати цю довідку\n";
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

    std::cout << "\033[1;36m[RADA]\033[0m Завантаження моделі: " << model_path << "\n";
    std::cout << "\033[1;36m[RADA]\033[0m Контекст: " << ctx_len << " токенів | GPU: #" << gpu_id << " | Потоки: " << threads << "\n";
    std::cout << "\033[1;36m[RADA]\033[0m Стиль: " << style_str << " | Temp: " << temp << " | Top-P: " << top_p << "\n\n";

    rada::RadaEngine engine;
    if (!engine.load_model(model_path, gpu_id, ctx_len)) {
        std::cerr << "\033[1;31m[ПОМИЛКА]\033[0m Не вдалося ініціалізувати модель " << model_path << "\n";
        return 1;
    }

    rada::GenerationParams gen_params;
    gen_params.temperature = temp;
    gen_params.top_p = top_p;
    gen_params.style = current_style;

    // Single prompt mode
    if (!single_prompt.empty()) {
        std::cout << "\033[1;32mКористувач:\033[0m " << single_prompt << "\n";
        std::cout << "\033[1;33mГетьман:\033[0m ";
        engine.generate_stream(single_prompt, gen_params, [](const std::string& chunk) {
            std::cout << chunk << std::flush;
            return true;
        });
        std::cout << "\n\n";
        auto stats = engine.get_stats();
        std::cout << "\033[1;30m[Швидкість: " << std::fixed << std::setprecision(1) << stats.last_speed_tok_s 
                  << " токен/с | VRAM: " << (stats.vram_usage_bytes / (1024 * 1024)) << " MB]\033[0m\n";
        return 0;
    }

    // Interactive REPL chat mode
    std::cout << "\033[1;32m[Рада зібрана]\033[0m Введіть ваше запитання або команду (введіть 'exit' для виходу):\n\n";
    std::string user_input;
    while (true) {
        std::cout << "\033[1;32mВи > \033[0m";
        if (!std::getline(std::cin, user_input) || user_input == "exit" || user_input == "вихід") {
            std::cout << "\n\033[1;33mРада завершила роботу. Бувайте здорові, побратиме!\033[0m\n";
            break;
        }
        if (user_input.empty()) continue;

        // Quick style switch command: /style <kozak|legal|tech>
        if (user_input.rfind("/style ", 0) == 0) {
            std::string s = user_input.substr(7);
            if (s == "kozak") { gen_params.style = rada::StylePreset::KOZAK; std::cout << "\033[1;35m[Стиль перемкнено на: ⚔️ Козацький]\033[0m\n"; }
            else if (s == "legal") { gen_params.style = rada::StylePreset::LEGAL; std::cout << "\033[1;35m[Стиль перемкнено на: ⚖️ Діловий / Державний]\033[0m\n"; }
            else if (s == "tech") { gen_params.style = rada::StylePreset::ENGINEERING; std::cout << "\033[1;35m[Стиль перемкнено на: 💻 Інженерний]\033[0m\n"; }
            else { std::cout << "Невідомий стиль. Оберіть: kozak, legal, tech\n"; }
            continue;
        }

        std::cout << "\033[1;33mГетьман > \033[0m";
        engine.generate_stream(user_input, gen_params, [](const std::string& chunk) {
            std::cout << chunk << std::flush;
            return true;
        });
        std::cout << "\n";

        auto stats = engine.get_stats();
        std::cout << "\033[1;30m[Швидкість: " << std::fixed << std::setprecision(1) << stats.last_speed_tok_s 
                  << " tok/s | VRAM: ~2.0 GB / 6.0 GB]\033[0m\n\n";
    }

    return 0;
}
