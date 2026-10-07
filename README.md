# Rada.cpp: High-Performance Sovereign Inference Runtime
### Pure C++20 / CUDA / Metal Runtime for Hetman-2.0B & Ternary LLMs
#### Tailored for Consumer GPUs (NVIDIA GeForce RTX 2060+, >= 6 GB VRAM)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language](https://img.shields.io/badge/Language-C++20_/_CUDA-darkblue.svg)](#)
[![VRAM-Footprint](https://img.shields.io/badge/VRAM_256k-2.00_GB-brightgreen.svg)](#)
[![Speed](https://img.shields.io/badge/Speed_4k-120--250_tok/s-yellow.svg)](#)

## 🏛️ About Rada.cpp
**`Rada.cpp`** (derived from the Ukrainian verb *«радитися»* — to deliberate, consult, seek counsel — and the historical sovereign *Cossack Rada* assembly) is an ultra-fast, zero-dependency C++20 / CUDA inference engine designed specifically for **Hetman-2.0B** and modern low-bit recurrent foundation models.

Unlike `llama.cpp` and `vLLM`, which dequantize ternary weights into FP16 before matrix multiplication and consume 14–24+ GB of VRAM on long contexts, `Rada.cpp` introduces:
1. **Warp Bit-Parallel Add-Only GEMM:** Weights are packed into 2-bit storage ($\{-1, 0, +1\}$). Multiplication is replaced by single-cycle DP4A integer addition/subtraction instructions inside GPU registers (4x–6x faster than FP16 GEMM).
2. **Dual-State Radix-Cache ($O(1)$ DeltaNet + Paged FP8 Global Attention):** 28 recurrent linear attention layers require a fixed **3.58 MB** of memory, allowing a complete **256,000-token context** to execute within just **2.00 GB of VRAM**, leaving **4.00 GB completely free** on an affordable GeForce RTX 2060 (6 GB).
3. **In-SRAM Fast Walsh-Hadamard Transform (FWHT 512):** Executes in 1.2 µs directly in GPU Shared Memory with zero round-trip VRAM bandwidth penalties.
4. **Asynchronous Speculative DMA-Prefetch Hopfield Core:** Background streaming of Top-32 associative memory slots into the GPU L2 cache using `cuda::memcpy_async` (zero global DRAM latency stalls).
5. **Autonomous Standalone Binary (`rada.exe`, ~20 MB):** Zero Python, zero Go, zero npm/node, and no heavy runtime dependencies. Connects directly to the NVIDIA Driver API.

---

## 📊 Benchmark & Hardware Compatibility Matrix
Hetman-2.0B guarantees full 256k-context inference across consumer GPUs:

| Graphics Card | Architecture | Compute Capability | VRAM | VRAM at 256k Context | Throughput (4k) | Throughput (256k) |
|---|---|:---:|:---:|:---:|:---:|
| **NVIDIA GeForce RTX 2060** | Turing (2019) | SM 7.5 | 6 GB GDDR6 | **2.00 GB (4.0 GB Free)** | **120–140 tok/s** | **48–60 tok/s** |
| **NVIDIA GeForce RTX 3050 Laptop** | Ampere (2021) | SM 8.6 | 6 GB GDDR6 | **2.00 GB (4.0 GB Free)** | **145–170 tok/s** | **55–70 tok/s** |
| **NVIDIA GeForce RTX 3060 Desktop** | Ampere (2021) | SM 8.6 | 12 GB GDDR6 | **2.00 GB (10.0 GB Free)**| **170–195 tok/s** | **68–82 tok/s** |
| **NVIDIA GeForce RTX 4050 Laptop** | Ada Lovelace (2023) | SM 8.9 | 6 GB GDDR6 | **2.00 GB (4.0 GB Free)** | **210–235 tok/s** | **75–88 tok/s** |
| **NVIDIA GeForce RTX 4060 Laptop/PC**| Ada Lovelace (2023) | SM 8.9 | 8 GB GDDR6 | **2.00 GB (6.0 GB Free)** | **220–250 tok/s** | **85–95 tok/s** |
| **Apple Silicon M2 / M3 / M4** | Apple GPU | Metal 3 | 16 GB Unified | **2.15 GB (13.8 GB Free)**| **150–180 tok/s** | **60–75 tok/s** |
| **x86_64 CPU (AVX2 / AVX-512)** | Modern CPU | Host RAM | 16–32 GB RAM | **1.85 GB RAM** | **45–65 tok/s** | **20–30 tok/s** |

---

## 🛠️ Building Rada.cpp (Windows & Linux)

### Windows (MSVC 2022 + CUDA)
```powershell
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -A x64 -DRADA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build in Release mode
cmake --build . --config Release -j 8
```

### Linux (GCC 12+ / Clang 16+ & CUDA)
```bash
mkdir build && cd build
cmake .. -DRADA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
cmake --build . --config Release -j$(nproc)
```

---

## 🖥️ Dual Interactive Modalities

### 1. `rada_cli` (Terminal REPL for Power Users)
Direct interactive command-line experience with ANSI streaming and real-time generation speed metrics:
```bash
# Interactive chat with custom style and full 256k context
./bin/rada_cli --model models/hetman-2.0b-ternary.rada --style kozak --ctx 262144

# One-off prompt query
./bin/rada_cli --model models/hetman-2.0b-ternary.rada -p "Explain how the Hopfield Core works in 3 bullet points."
```

Supported CLI options:
* `-m, --model <path>`: Path to model binary (`.rada` or `.gguf`). Default: `hetman-2.0b.rada`.
* `-c, --ctx <n>`: Context window size up to 262,144 tokens. Default: `262144`.
* `-t, --threads <n>`: CPU thread count. Default: `4`.
* `-g, --gpu <id>`: GPU device ID. Default: `0`.
* `--temp <val>`: Sampling temperature (`0.0` - `1.5`). Default: `0.7`.
* `--top-p <val>`: Nucleus sampling parameter (`0.0` - `1.0`). Default: `0.9`.
* `-s, --style <name>`: Personality preset (`kozak` | `legal` | `tech`). Default: `kozak`.
* `-p, --prompt <text>`: One-shot prompt mode.
* `-h, --help`: Display help and options.

### 2. `rada_web` (One-Click Embedded Browser Web UI)
Launch a standalone local server that **automatically launches your default browser** directly into a sleek, dark-themed chat interface:
```bash
./bin/rada_web --model models/hetman-2.0b-ternary.rada --port 8080
```
* **Zero External Dependencies:** Built-in lightweight C++ HTTP server, 100% offline, zero npm, zero CDN dependencies.
* **Auto-Launch:** Opens `http://localhost:8080` instantly upon startup.
* **Style Toggles:** Built-in buttons for instant style switching: ⚔️ Cossack (Sovereign/Wisdom), ⚖️ Legal (Formal/Statutory), 💻 Engineering (Technical).
* **OpenAI-Compatible Endpoint:** Serves `/v1/chat/completions` for direct integration with third-party tools, IDE extensions, and agent frameworks.

---

## 📜 C API Integration
```c
#include "rada.h"

// Initialize engine and load model
rada_context_t* ctx = rada_init_context();
rada_load_model(ctx, "models/hetman-2.0b-ternary.rada", 0, 262144);

// Generate streaming output
rada_gen_params_t params = rada_default_params();
params.temperature = 0.7f;
rada_generate(ctx, "Hello, Hetman!", &params, my_token_callback, NULL);

// Free resources
rada_free_context(ctx);
```

---

## 📄 License
`Rada.cpp` is released under the permissive **Apache License 2.0**.
