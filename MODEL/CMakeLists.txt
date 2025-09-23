cmake_minimum_required(VERSION 3.18)
project(ArielHybridModel LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures for RTX 3080 (86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# Find required packages for Linux
find_package(PkgConfig REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/eigen3
)

# Add the main executable
add_executable(ariel_trainer
    ariel_trainer.cpp
    nebula_kernels.cu
)

# Add the FIXED executable with multi-GPU support
add_executable(fixed_ariel_trainer
    fixed_ariel_trainer.cpp
    nebula_kernels.cu
)

# Add target include directories
target_include_directories(ariel_trainer PRIVATE /usr/include/eigen3)
target_include_directories(fixed_ariel_trainer PRIVATE /usr/include/eigen3)

# Set target properties
set_target_properties(ariel_trainer PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(fixed_ariel_trainer PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link libraries for Linux
target_link_libraries(ariel_trainer
    PRIVATE
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
    ${CMAKE_CURRENT_SOURCE_DIR}/libcnpy.so
    openblas
    z
)

target_link_libraries(fixed_ariel_trainer
    PRIVATE
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
    ${CMAKE_CURRENT_SOURCE_DIR}/libcnpy.so
    openblas
    z
)

# Compiler-specific options
target_compile_options(ariel_trainer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-fPIC
        --extended-lambda
        --expt-relaxed-constexpr
    >
)

target_compile_options(fixed_ariel_trainer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-fPIC
        --extended-lambda
        --expt-relaxed-constexpr
    >
)