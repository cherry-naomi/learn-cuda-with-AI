# CUDA Examples Makefile

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_50 -std=c++11

# Target executables
TARGET1 = vector_add
TARGET2 = vector_add_2d  
TARGET3 = cuda_threading_explained
TARGET4 = hardware_mapping_explained
TARGET5 = shared_memory_explained
TARGET6 = block_isolation_demo
TARGET7 = vector_add_optimized
TARGET8 = insufficient_threads_demo
TARGET9 = block_essence_chinese
TARGET10 = softmax_basic
TARGET11 = softmax_optimized
TARGET12 = softmax_unittest

# Source files
SOURCES1 = src/vector_add/vector_add.cu
SOURCES2 = src/vector_add/vector_add_2d.cu
SOURCES3 = src/vector_add/cuda_threading_explained.cu
SOURCES4 = src/vector_add/hardware_mapping_explained.cu
SOURCES5 = src/vector_add/shared_memory_explained.cu
SOURCES6 = src/vector_add/block_isolation_demo.cu
SOURCES7 = src/vector_add/vector_add_optimized.cu
SOURCES8 = src/vector_add/insufficient_threads_demo.cu
SOURCES9 = src/vector_add/block_essence_chinese.cu
SOURCES10 = src/softmax/softmax_basic.cu
SOURCES11 = src/softmax/softmax_optimized.cu
SOURCES12 = src/softmax/softmax_unittest.cu

# Default target - build all examples
all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) $(TARGET9) $(TARGET10) $(TARGET11) $(TARGET12)

# Build the basic vector addition example
$(TARGET1): $(SOURCES1)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET1) $(SOURCES1)

# Build the 2D grid configuration example  
$(TARGET2): $(SOURCES2)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET2) $(SOURCES2)

# Build the threading explanation example
$(TARGET3): $(SOURCES3)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET3) $(SOURCES3)

# Build the hardware mapping explanation
$(TARGET4): $(SOURCES4)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET4) $(SOURCES4)

# Build the shared memory explanation
$(TARGET5): $(SOURCES5)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET5) $(SOURCES5)

# Build the block isolation demonstration
$(TARGET6): $(SOURCES6)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET6) $(SOURCES6)

# Build the optimized vector addition
$(TARGET7): $(SOURCES7)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET7) $(SOURCES7)

# Build the insufficient threads demonstration
$(TARGET8): $(SOURCES8)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET8) $(SOURCES8)

# Build the block essence Chinese explanation
$(TARGET9): $(SOURCES9)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET9) $(SOURCES9)

# Build the basic softmax implementation
$(TARGET10): $(SOURCES10)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET10) $(SOURCES10)

# Build the optimized softmax implementation
$(TARGET11): $(SOURCES11)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET11) $(SOURCES11)

# Build the softmax unit tests
$(TARGET12): $(SOURCES12)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET12) $(SOURCES12)

# Clean up generated files
clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) $(TARGET9) $(TARGET10) $(TARGET11) $(TARGET12)

# Run the basic example
run1: $(TARGET1)
	./$(TARGET1)

# Run the 2D configuration example
run2: $(TARGET2) 
	./$(TARGET2)

# Run the threading explanation (RECOMMENDED FOR LEARNING!)
run3: $(TARGET3)
	./$(TARGET3)

# Run the hardware mapping explanation
run4: $(TARGET4)
	./$(TARGET4)

# Run the shared memory explanation
run5: $(TARGET5)
	./$(TARGET5)

# Run the block isolation demonstration
run6: $(TARGET6)
	./$(TARGET6)

# Run the optimized vector addition benchmarks
run7: $(TARGET7)
	./$(TARGET7)

# Run the insufficient threads demonstration
run8: $(TARGET8)
	./$(TARGET8)

# Run the block essence Chinese explanation
run9: $(TARGET9)
	./$(TARGET9)

# Run the basic softmax implementation
run10: $(TARGET10)
	./$(TARGET10)

# Run the optimized softmax implementation
run11: $(TARGET11)
	./$(TARGET11)

# Run the softmax unit tests
test: $(TARGET12)
	./$(TARGET12)

# Alias for unit tests
run12: test

# Run profiling analysis
profile: $(TARGET1) $(TARGET7)
	./src/vector_add/profile_vector_add.sh

# Run all examples
run: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) $(TARGET9) $(TARGET10) $(TARGET11) $(TARGET12)
	@echo "=== Running Threading Explanation (Educational) ==="
	./$(TARGET3)
	@echo ""
	@echo "=== Running Hardware Mapping Explanation ==="
	./$(TARGET4)
	@echo ""
	@echo "=== Running Shared Memory Explanation ==="
	./$(TARGET5)
	@echo ""
	@echo "=== Running Block Isolation Demonstration ==="
	./$(TARGET6)
	@echo ""
	@echo "=== Running Performance Analysis ==="
	./$(TARGET7)
	@echo ""
	@echo "=== Running Insufficient Threads Demo ==="
	./$(TARGET8)
	@echo ""
	@echo "=== Running Block Essence Explanation (Chinese) ==="
	./$(TARGET9)
	@echo ""
	@echo "=== Running Basic Vector Addition Example ==="
	./$(TARGET1)
	@echo ""
	@echo "=== Running 2D Grid Configuration Example ==="
	./$(TARGET2)
	@echo ""
	@echo "=== Running Basic Softmax Implementation ==="
	./$(TARGET10)
	@echo ""
	@echo "=== Running Optimized Softmax Implementation ==="
	./$(TARGET11)
	@echo ""
	@echo "=== Running Softmax Unit Tests ==="
	./$(TARGET12)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build all examples"
	@echo "  $(TARGET1)    - Build basic vector addition"
	@echo "  $(TARGET2) - Build 2D configuration example"  
	@echo "  $(TARGET3) - Build threading explanation"
	@echo "  $(TARGET4) - Build hardware mapping explanation"
	@echo "  $(TARGET5) - Build shared memory explanation"
	@echo "  $(TARGET6) - Build block isolation demo"
	@echo "  $(TARGET7) - Build optimized vector addition"
	@echo "  $(TARGET8) - Build insufficient threads demo"
	@echo "  $(TARGET9) - Build block essence explanation (Chinese)"
	@echo "  $(TARGET10) - Build basic softmax implementation"
	@echo "  $(TARGET11) - Build optimized softmax implementation"
	@echo "  $(TARGET12) - Build softmax unit tests"
	@echo "  clean   - Remove generated files"
	@echo "  run     - Build and run all examples"
	@echo "  run1    - Run basic example"
	@echo "  run2    - Run 2D example"
	@echo "  run3    - Run threading explanation"
	@echo "  run4    - Run hardware mapping"
	@echo "  run5    - Run shared memory (BLOCKS!)"
	@echo "  run6    - Run block isolation (ISOLATION!)"
	@echo "  run7    - Run performance analysis (OPTIMIZE!)"
	@echo "  run8    - Run insufficient threads (COVERAGE!)"
	@echo "  run9    - Run block essence (ESSENCE! 中文)"
	@echo "  run10   - Run basic softmax (SOFTMAX BASICS!)"
	@echo "  run11   - Run optimized softmax (ADVANCED SOFTMAX!)"
	@echo "  test    - Run softmax unit tests (TRANSFORMER TESTS!)"
	@echo "  run12   - Alias for test"
	@echo "  profile - Run comprehensive profiling"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Learning: run3 → run4 → run5 → run6 → run9 → run8 → run1 → run2"
	@echo "Softmax: run10 → run11 → test"
	@echo "Performance: run7 → profile"

.PHONY: all clean run run1 run2 run3 run4 run5 run6 run7 run8 run9 run10 run11 run12 test profile help
