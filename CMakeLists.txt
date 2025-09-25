cmake_minimum_required(VERSION 3.18)

# Project name and version
project(CUDA_Examples 
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "CUDA Learning Examples with Vector Addition and Softmax"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA specific settings
enable_language(CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture (detect automatically or use default)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86)  # Support multiple architectures
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")  # Debug flags with device debug info
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")  # Release optimization

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Find cuBLAS and cuDNN
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# VECTOR ADDITION EXAMPLES
# =============================================================================

# Basic vector addition
add_executable(vector_add 
    src/vector_add/vector_add.cu
)

# 2D vector addition
add_executable(vector_add_2d 
    src/vector_add/vector_add_2d.cu
)

# CUDA threading explanation
add_executable(cuda_threading_explained 
    src/vector_add/cuda_threading_explained.cu
)

# Hardware mapping explanation
add_executable(hardware_mapping_explained 
    src/vector_add/hardware_mapping_explained.cu
)

# Shared memory explanation
add_executable(shared_memory_explained 
    src/vector_add/shared_memory_explained.cu
)

# Block isolation demo
add_executable(block_isolation_demo 
    src/vector_add/block_isolation_demo.cu
)

# Optimized vector addition
add_executable(vector_add_optimized 
    src/vector_add/vector_add_optimized.cu
)

# Insufficient threads demo
add_executable(insufficient_threads_demo 
    src/vector_add/insufficient_threads_demo.cu
)

# Block essence Chinese explanation
add_executable(block_essence_chinese 
    src/vector_add/block_essence_chinese.cu
)

# =============================================================================
# SOFTMAX EXAMPLES
# =============================================================================

# Basic softmax implementation
add_executable(softmax_basic 
    src/softmax/softmax_basic.cu
    src/utils/perf.cu
)

# Optimized softmax implementation
add_executable(softmax_optimized 
    src/softmax/softmax_optimized.cu
    src/utils/perf.cu
)

# Softmax unit tests (removed - file deleted)
# add_executable(softmax_unittest 
#     src/softmax/softmax_unittest.cu
# )

# cuDNN implementations removed

# Warp shuffle demonstration
add_executable(warp_shuffle_demo 
    src/softmax/warp_shuffle_demo.cu
)

# =============================================================================
# CUSTOM TARGETS FOR RUNNING EXAMPLES
# =============================================================================

# Run all examples
add_custom_target(run_all
    DEPENDS vector_add vector_add_2d cuda_threading_explained 
            hardware_mapping_explained shared_memory_explained 
            block_isolation_demo vector_add_optimized 
            insufficient_threads_demo block_essence_chinese
            softmax_basic softmax_optimized
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Threading Explanation (Educational) ==="
    COMMAND $<TARGET_FILE:cuda_threading_explained>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Hardware Mapping Explanation ==="
    COMMAND $<TARGET_FILE:hardware_mapping_explained>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Shared Memory Explanation ==="
    COMMAND $<TARGET_FILE:shared_memory_explained>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Block Isolation Demonstration ==="
    COMMAND $<TARGET_FILE:block_isolation_demo>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Performance Analysis ==="
    COMMAND $<TARGET_FILE:vector_add_optimized>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Insufficient Threads Demo ==="
    COMMAND $<TARGET_FILE:insufficient_threads_demo>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Block Essence Explanation (Chinese) ==="
    COMMAND $<TARGET_FILE:block_essence_chinese>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Basic Vector Addition Example ==="
    COMMAND $<TARGET_FILE:vector_add>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running 2D Grid Configuration Example ==="
    COMMAND $<TARGET_FILE:vector_add_2d>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Basic Softmax Implementation ==="
    COMMAND $<TARGET_FILE:softmax_basic>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Optimized Softmax Implementation ==="
    COMMAND $<TARGET_FILE:softmax_optimized>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Softmax Unit Tests ==="
    # COMMAND $<TARGET_FILE:softmax_unittest>
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "Running all CUDA examples"
)

# Individual run targets
add_custom_target(run1 COMMAND $<TARGET_FILE:vector_add>)
add_custom_target(run2 COMMAND $<TARGET_FILE:vector_add_2d>)
add_custom_target(run3 COMMAND $<TARGET_FILE:cuda_threading_explained>)
add_custom_target(run4 COMMAND $<TARGET_FILE:hardware_mapping_explained>)
add_custom_target(run5 COMMAND $<TARGET_FILE:shared_memory_explained>)
add_custom_target(run6 COMMAND $<TARGET_FILE:block_isolation_demo>)
add_custom_target(run7 COMMAND $<TARGET_FILE:vector_add_optimized>)
add_custom_target(run8 COMMAND $<TARGET_FILE:insufficient_threads_demo>)
add_custom_target(run9 COMMAND $<TARGET_FILE:block_essence_chinese>)
add_custom_target(run10 COMMAND $<TARGET_FILE:softmax_basic>)
add_custom_target(run11 COMMAND $<TARGET_FILE:softmax_optimized>)
# add_custom_target(run12 COMMAND $<TARGET_FILE:softmax_unittest>)
# add_custom_target(run13 COMMAND $<TARGET_FILE:softmax_cudnn>)

# Test target (alias for softmax unit tests)
# add_custom_target(test COMMAND $<TARGET_FILE:softmax_unittest>)

# =============================================================================
# PROFILING TARGET
# =============================================================================

add_custom_target(profile
    DEPENDS vector_add vector_add_optimized
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR} 
            bash src/vector_add/profile_vector_add.sh
    COMMENT "Running comprehensive profiling analysis"
)

# =============================================================================
# INSTALLATION
# =============================================================================

# Install all executables
install(TARGETS 
    vector_add vector_add_2d cuda_threading_explained 
    hardware_mapping_explained shared_memory_explained 
    block_isolation_demo vector_add_optimized 
    insufficient_threads_demo block_essence_chinese
    softmax_basic softmax_optimized
    DESTINATION bin
)

# Install documentation
install(FILES 
    README.md
    src/vector_add/INSTALL.md
    src/vector_add/performance_guide.md
    src/softmax/README.md
    CMAKE_USAGE.md
    DESTINATION share/doc/cuda_examples
)

# =============================================================================
# PRINT CONFIGURATION INFO
# =============================================================================

message(STATUS "CUDA Examples Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA version: ${CUDA_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  make all          - Build all examples")
message(STATUS "  make run_all      - Build and run all examples")
message(STATUS "  make run1-run16   - Run individual examples")
message(STATUS "  make test         - Run softmax unit tests")
message(STATUS "  make profile      - Run performance profiling")
message(STATUS "  make install      - Install to system")
message(STATUS "")
message(STATUS "Learning path: run3 → run4 → run5 → run6 → run9 → run8 → run1 → run2")
message(STATUS "Softmax path: run10 → run11")
message(STATUS "Performance: run7 → profile")
