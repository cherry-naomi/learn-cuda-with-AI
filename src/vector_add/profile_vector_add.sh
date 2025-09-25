#!/bin/bash

# CUDA Vector Add Performance Profiling Script

echo "=== CUDA VECTOR ADD PROFILING GUIDE ==="
echo ""

# Check if profiling tools are available
echo "Checking available profiling tools..."

if command -v nvprof &> /dev/null; then
    NVPROF_AVAILABLE=true
    echo "✅ nvprof found"
else
    NVPROF_AVAILABLE=false
    echo "❌ nvprof not found"
fi

if command -v ncu &> /dev/null; then
    NCU_AVAILABLE=true
    echo "✅ ncu (Nsight Compute) found"
else
    NCU_AVAILABLE=false
    echo "❌ ncu (Nsight Compute) not found"
fi

if command -v nsys &> /dev/null; then
    NSYS_AVAILABLE=true
    echo "✅ nsys (Nsight Systems) found"
else
    NSYS_AVAILABLE=false
    echo "❌ nsys (Nsight Systems) not found"
fi

echo ""

# Build the programs if they don't exist
echo "Building vector addition examples..."
make vector_add vector_add_optimized 2>/dev/null || echo "Build failed - check if CUDA is installed"

echo ""
echo "=== BASIC PERFORMANCE MEASUREMENT ==="
echo ""

# Run basic timing
if [ -f "./vector_add" ]; then
    echo "Running basic vector_add timing:"
    ./vector_add
    echo ""
fi

if [ -f "./vector_add_optimized" ]; then
    echo "Running optimized vector_add performance analysis:"
    ./vector_add_optimized
    echo ""
fi

echo "=== PROFILING WITH NVPROF (Legacy) ==="
echo ""

if [ "$NVPROF_AVAILABLE" = true ] && [ -f "./vector_add" ]; then
    echo "1. Basic nvprof profiling:"
    echo "Command: nvprof ./vector_add"
    nvprof ./vector_add
    echo ""
    
    echo "2. Detailed metrics:"
    echo "Command: nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./vector_add"
    nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./vector_add 2>/dev/null || echo "Some metrics may not be available on this GPU"
    echo ""
    
    echo "3. Memory bandwidth analysis:"
    echo "Command: nvprof --metrics dram_read_throughput,dram_write_throughput ./vector_add"
    nvprof --metrics dram_read_throughput,dram_write_throughput ./vector_add 2>/dev/null || echo "Memory metrics may not be available"
    echo ""
else
    echo "nvprof not available. Install CUDA Toolkit for profiling support."
    echo ""
fi

echo "=== PROFILING WITH NSIGHT COMPUTE (Modern) ==="
echo ""

if [ "$NCU_AVAILABLE" = true ] && [ -f "./vector_add" ]; then
    echo "1. Basic performance analysis:"
    echo "Command: ncu --set full ./vector_add"
    echo "Running simplified version..."
    ncu --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed ./vector_add 2>/dev/null || echo "ncu failed - may need newer GPU/driver"
    echo ""
    
    echo "2. Memory analysis:"
    echo "Command: ncu --set memory ./vector_add"
    echo "(This would show detailed memory metrics)"
    echo ""
    
    echo "3. Occupancy analysis:"
    echo "Command: ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./vector_add"
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./vector_add 2>/dev/null || echo "Occupancy metric not available"
    echo ""
else
    echo "ncu not available. Install latest CUDA Toolkit for Nsight Compute."
    echo ""
fi

echo "=== PROFILING WITH NSIGHT SYSTEMS (Timeline) ==="
echo ""

if [ "$NSYS_AVAILABLE" = true ] && [ -f "./vector_add" ]; then
    echo "Creating timeline profile:"
    echo "Command: nsys profile --output=vector_add_timeline ./vector_add"
    nsys profile --output=vector_add_timeline --force-overwrite=true ./vector_add 2>/dev/null || echo "nsys profiling failed"
    
    if [ -f "vector_add_timeline.nsys-rep" ]; then
        echo "✅ Timeline profile created: vector_add_timeline.nsys-rep"
        echo "Open with: nsys-ui vector_add_timeline.nsys-rep"
    fi
    echo ""
else
    echo "nsys not available. Install CUDA Toolkit for Nsight Systems."
    echo ""
fi

echo "=== MANUAL PERFORMANCE ANALYSIS ==="
echo ""

echo "Key metrics to analyze:"
echo ""
echo "1. KERNEL EXECUTION TIME"
echo "   - Look for kernel duration in microseconds"
echo "   - Compare with theoretical minimum"
echo ""
echo "2. MEMORY BANDWIDTH UTILIZATION"
echo "   - Vector add is memory-bound"
echo "   - Target: >80% of peak memory bandwidth"
echo "   - Formula: (3 * N * sizeof(float)) / time"
echo ""
echo "3. OCCUPANCY"
echo "   - Target: >50% occupancy"
echo "   - Balance threads per block vs resource usage"
echo ""
echo "4. MEMORY EFFICIENCY"
echo "   - Global load/store efficiency should be >90%"
echo "   - Indicates coalesced memory access"
echo ""

echo "=== OPTIMIZATION RECOMMENDATIONS ==="
echo ""

echo "For MEMORY-BOUND kernels like vector add:"
echo ""
echo "✅ DO:"
echo "   - Use 256-512 threads per block"
echo "   - Ensure coalesced memory access"
echo "   - Use vectorized loads (float4) if possible"
echo "   - Consider grid-stride loops for large data"
echo ""
echo "❌ AVOID:"
echo "   - Complex computations (this is memory-bound)"
echo "   - Shared memory unless you have reuse"
echo "   - Too small block sizes (<128 threads)"
echo "   - Divergent branching"
echo ""

echo "=== PROFILING COMMANDS REFERENCE ==="
echo ""

echo "NVPROF (Legacy):"
echo "  nvprof ./program                           # Basic profiling"
echo "  nvprof --metrics achieved_occupancy        # Occupancy"
echo "  nvprof --metrics gld_efficiency            # Memory efficiency"
echo "  nvprof --metrics dram_read_throughput      # Memory bandwidth"
echo ""

echo "NCU (Nsight Compute):"
echo "  ncu ./program                              # Interactive analysis"
echo "  ncu --set full ./program                   # Full metric set"
echo "  ncu --set memory ./program                 # Memory focus"
echo "  ncu --metrics metric_name ./program        # Specific metrics"
echo ""

echo "NSYS (Nsight Systems):"
echo "  nsys profile ./program                     # Timeline profiling"
echo "  nsys profile --gpu-metrics-device=all     # GPU metrics"
echo "  nsys-ui profile.nsys-rep                   # Open GUI"
echo ""

echo "=== GPU-SPECIFIC OPTIMIZATION ==="

# Get GPU info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Your GPU configuration:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    
    # Try to get compute capability
    if [ -f "./vector_add_optimized" ]; then
        ./vector_add_optimized 2>/dev/null | grep "Compute Capability" | head -1
    fi
    echo ""
fi

echo "Optimization tips based on compute capability:"
echo "  - SM 3.x (Kepler): Focus on occupancy"
echo "  - SM 5.x (Maxwell): Good memory bandwidth"
echo "  - SM 6.x (Pascal): Optimize for memory coalescing"
echo "  - SM 7.x (Volta/Turing): Use Tensor cores if applicable"
echo "  - SM 8.x (Ampere): Excellent memory bandwidth"
echo ""

echo "=== DONE ==="
echo ""
echo "For detailed analysis:"
echo "1. Run ./vector_add_optimized for comprehensive benchmarks"
echo "2. Use ncu for detailed kernel analysis"
echo "3. Use nsys for timeline and CPU-GPU interaction"
echo ""
echo "Happy profiling! 🚀"
