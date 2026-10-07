cmake_minimum_required(VERSION 3.22)
project(rada_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(RADA_CUDA "Enable NVIDIA CUDA Acceleration" ON)
option(RADA_BUILD_CLI "Build interactive terminal CLI chat (rada_cli)" ON)
option(RADA_BUILD_WEB "Build minimalist Web UI & Server with auto-browser opening (rada_web)" ON)
option(RADA_BUILD_SHARED "Build shared library for Python / Rust FFI" OFF)

# MSVC UTF-8 encoding support for Ukrainian strings and emoji
if (MSVC)
    add_compile_options(/utf-8 /permissive- /W4)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

set(RADA_SOURCES
    src/rada_engine.cpp
    src/rada_c_api.cpp
)

if (RADA_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Support NVIDIA Turing (75), Ampere (86), Ada Lovelace (89)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    endif()
    
    add_compile_definitions(RADA_USE_CUDA)
    list(APPEND RADA_SOURCES
        src/kernels/cuda/rada_ternary_gemm.cu
        src/kernels/cuda/rada_fwht.cu
        src/kernels/cuda/rada_hopfield.cu
    )
endif()

if (RADA_BUILD_SHARED)
    add_library(rada_core SHARED ${RADA_SOURCES})
    target_compile_definitions(rada_core PRIVATE RADA_BUILD_SHARED)
else()
    add_library(rada_core STATIC ${RADA_SOURCES})
endif()

target_include_directories(rada_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (RADA_CUDA)
    target_compile_options(rada_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
endif()

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

