#!/bin/bash

# Remote CUDA Profiling Script
# Generates profiles on server that can be viewed on client

echo "=== REMOTE CUDA PROFILING WORKFLOW ==="
echo "Server: $(hostname)"
echo "Date: $(date)"
echo ""

# Check if programs exist
if [ ! -f "./vector_add" ]; then
    echo "Building vector_add..."
    make vector_add || { echo "Build failed!"; exit 1; }
fi

if [ ! -f "./vector_add_optimized" ]; then
    echo "Building vector_add_optimized..."
    make vector_add_optimized || { echo "Build failed!"; exit 1; }
fi

echo "=== STEP 1: Generate Nsight Systems Timeline ==="
echo "Command: nsys profile --output=vector_add_timeline --force-overwrite=true ./vector_add"

if command -v nsys &> /dev/null; then
    nsys profile --output=vector_add_timeline --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --gpu-metrics-device=all \
        --delay=1 \
        ./vector_add
    
    if [ -f "vector_add_timeline.nsys-rep" ]; then
        echo "✅ Timeline profile generated: vector_add_timeline.nsys-rep"
        echo "   Size: $(ls -lh vector_add_timeline.nsys-rep | awk '{print $5}')"
    else
        echo "❌ Timeline profile generation failed"
    fi
else
    echo "❌ nsys not found - install CUDA Toolkit"
fi

echo ""
echo "=== STEP 2: Generate Nsight Compute Detailed Analysis ==="
echo "Command: ncu --export vector_add_detailed --force-overwrite ./vector_add"

if command -v ncu &> /dev/null; then
    # Try full analysis first, fall back to basic if it fails
    ncu --export vector_add_detailed --force-overwrite \
        --set full \
        --target-processes all \
        ./vector_add 2>/dev/null || \
    ncu --export vector_add_detailed --force-overwrite \
        --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
        ./vector_add
    
    if [ -f "vector_add_detailed.ncu-rep" ]; then
        echo "✅ Detailed profile generated: vector_add_detailed.ncu-rep"
        echo "   Size: $(ls -lh vector_add_detailed.ncu-rep | awk '{print $5}')"
    else
        echo "❌ Detailed profile generation failed"
    fi
else
    echo "❌ ncu not found - install CUDA Toolkit"
fi

echo ""
echo "=== STEP 3: Generate Text Reports ==="

# Nsight Systems text reports
if [ -f "vector_add_timeline.nsys-rep" ]; then
    echo "Generating Nsight Systems text reports..."
    
    # Overall statistics
    nsys stats vector_add_timeline.nsys-rep > nsys_overall_stats.txt 2>/dev/null || echo "nsys stats failed"
    
    # Kernel summary
    nsys stats --report kernel_sum vector_add_timeline.nsys-rep > nsys_kernel_summary.txt 2>/dev/null || echo "nsys kernel summary failed"
    
    # CUDA API summary
    nsys stats --report cuda_api_sum vector_add_timeline.nsys-rep > nsys_cuda_api.txt 2>/dev/null || echo "nsys cuda api failed"
    
    # CSV export for spreadsheet analysis
    nsys stats --output=csv --force-overwrite=true vector_add_timeline.nsys-rep > vector_add_stats.csv 2>/dev/null || echo "nsys CSV export failed"
    
    echo "✅ Text reports generated"
fi

# Quick performance analysis with built-in tools
echo ""
echo "=== STEP 4: Built-in Performance Analysis ==="
if [ -f "./vector_add_optimized" ]; then
    echo "Running comprehensive performance analysis..."
    echo ""
    ./vector_add_optimized | head -50  # Show first 50 lines
    echo "... (truncated - full output in vector_add_optimized)"
fi

echo ""
echo "=== STEP 5: Summary of Generated Files ==="
echo ""

# List all generated files
files_generated=()

if [ -f "vector_add_timeline.nsys-rep" ]; then
    files_generated+=("vector_add_timeline.nsys-rep")
fi

if [ -f "vector_add_detailed.ncu-rep" ]; then
    files_generated+=("vector_add_detailed.ncu-rep")
fi

if [ -f "vector_add_stats.csv" ]; then
    files_generated+=("vector_add_stats.csv")
fi

if [ -f "nsys_overall_stats.txt" ]; then
    files_generated+=("nsys_overall_stats.txt")
fi

if [ -f "nsys_kernel_summary.txt" ]; then
    files_generated+=("nsys_kernel_summary.txt")
fi

if [ -f "nsys_cuda_api.txt" ]; then
    files_generated+=("nsys_cuda_api.txt")
fi

if [ ${#files_generated[@]} -eq 0 ]; then
    echo "❌ No profile files generated - check CUDA installation"
    exit 1
fi

echo "Generated profile files:"
for file in "${files_generated[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  📁 $file ($size)"
    fi
done

echo ""
echo "=== STEP 6: Client Download Instructions ==="
echo ""

# Get current directory and hostname
current_dir=$(pwd)
hostname=$(hostname)
username=$(whoami)

echo "To download these files to your client laptop:"
echo ""

for file in "${files_generated[@]}"; do
    echo "scp $username@$hostname:$current_dir/$file ."
done

echo ""
echo "Or download all at once:"
echo "scp $username@$hostname:$current_dir/{$(IFS=,; echo "${files_generated[*]}")} ."

echo ""
echo "=== STEP 7: Client Analysis Instructions ==="
echo ""

echo "After downloading, install NVIDIA tools on your client:"
echo ""
echo "Windows/Mac:"
echo "  - Download CUDA Toolkit from https://developer.nvidia.com/cuda-toolkit"
echo "  - Or download standalone Nsight tools"
echo ""
echo "Linux Client:"
echo "  sudo apt install cuda-nsight-systems cuda-nsight-compute"
echo ""

echo "Then open the profiles:"
if [ -f "vector_add_timeline.nsys-rep" ]; then
    echo "  nsys-ui vector_add_timeline.nsys-rep    # Timeline analysis"
fi

if [ -f "vector_add_detailed.ncu-rep" ]; then
    echo "  ncu-ui vector_add_detailed.ncu-rep      # Detailed kernel analysis"
fi

echo ""
echo "=== STEP 8: Text-Based Analysis (Available Now) ==="
echo ""

if [ -f "nsys_kernel_summary.txt" ]; then
    echo "Kernel execution summary:"
    echo "------------------------"
    head -20 nsys_kernel_summary.txt 2>/dev/null || echo "Summary not available"
    echo ""
fi

if [ -f "vector_add_stats.csv" ]; then
    echo "CSV data preview (first 10 lines):"
    echo "-----------------------------------"
    head -10 vector_add_stats.csv 2>/dev/null || echo "CSV not available"
    echo ""
fi

echo "=== ALTERNATIVE: X11 Forwarding ==="
echo ""
echo "If you have X11 forwarding enabled:"
echo "  ssh -X $username@$hostname"
echo "  cd $current_dir"
if [ -f "vector_add_timeline.nsys-rep" ]; then
    echo "  nsys-ui vector_add_timeline.nsys-rep"
fi

echo ""
echo "=== DONE! ==="
echo ""
echo "Profile files ready for download and analysis."
echo "Use 'ls -la *.rep *.txt *.csv' to see all generated files."
echo ""
echo "For help: cat remote_profiling_guide.md"
