cmake_minimum_required(VERSION 3.22)
project(rada_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(RADA_CUDA "Enable NVIDIA CUDA Acceleration" ON)
option(RADA_BUILD_CLI "Build interactive terminal CLI chat (rada_cli)" ON)
option(RADA_BUILD_WEB "Build minimalist Web UI & Server with auto-browser opening (rada_web)" ON)

if (RADA_CUDA)
    enable_language(CUDA)
    # Support NVIDIA Turing (75), Ampere (86), Ada Lovelace (89)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
    add_compile_definitions(RADA_USE_CUDA)
endif()

include_directories(include)

set(RADA_SOURCES
    src/rada_engine.cpp
    src/kernels/cuda/rada_ternary_gemm.cu
    src/kernels/cuda/rada_fwht.cu
    src/kernels/cuda/rada_hopfield.cu
)

add_library(rada_core STATIC ${RADA_SOURCES})

# 1. Terminal Interactive Chat Executable
if (RADA_BUILD_CLI)
    add_executable(rada_cli tools/rada_cli.cpp)
    target_link_libraries(rada_cli PRIVATE rada_core)
endif()

# 2. Minimalist Web UI & Server Executable (Auto-opens browser)
if (RADA_BUILD_WEB)
    add_executable(rada_web tools/rada_web.cpp)
    target_link_libraries(rada_web PRIVATE rada_core)
    if (WIN32)
        target_link_libraries(rada_web PRIVATE ws2_32 shell32)
    endif()
endif()
