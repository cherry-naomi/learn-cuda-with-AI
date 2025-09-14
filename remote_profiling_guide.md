# Remote CUDA Profiling Guide: Server → Client

## 🖥️ Server-Client Profiling Workflow

Since you're running CUDA on a Linux server but want to view results on your client laptop, here are the best approaches:

## 📊 Method 1: Download Profile Files (Recommended)

### Step 1: Generate Profile on Server
```bash
# On your Linux server
cd YOUR_WORKSPACE/cuda_examples

# Generate timeline profile
nsys profile --output=vector_add_timeline --force-overwrite=true ./vector_add

# Generate detailed kernel profile  
ncu --export vector_add_detailed --force-overwrite ./vector_add

# List generated files
ls -la *.nsys-rep *.ncu-rep
```

### Step 2: Download Files to Client
```bash
# From your client laptop, download the profile files
scp user@server:YOUR_WORKSPACE/cuda_examples/vector_add_timeline.nsys-rep .
scp user@server:YOUR_WORKSPACE/cuda_examples/vector_add_detailed.ncu-rep .
```

### Step 3: Install Nsight Tools on Client
**Windows Client:**
- Download CUDA Toolkit from NVIDIA website
- Install includes Nsight Compute and Nsight Systems
- Or download standalone Nsight tools

**Mac Client:**
- Download CUDA Toolkit for macOS (if available)
- Or use Windows VM/dual boot

**Linux Client:**
```bash
# Install CUDA Toolkit on your laptop
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-nsight-systems cuda-nsight-compute
```

### Step 4: Open on Client
```bash
# On your client laptop
nsys-ui vector_add_timeline.nsys-rep    # Timeline view
ncu-ui vector_add_detailed.ncu-rep      # Detailed kernel analysis
```

## 🌐 Method 2: X11 Forwarding (Linux Client Only)

### Setup X11 Forwarding
```bash
# From your Linux client laptop
ssh -X user@server

# Test X11 forwarding works
xclock  # Should show a clock window

# If X11 forwarding works, run GUI tools directly
cd YOUR_WORKSPACE/cuda_examples
nsys profile --output=timeline ./vector_add
nsys-ui timeline.nsys-rep  # Opens on your client!
```

### Troubleshooting X11
```bash
# If X11 doesn't work, try:
ssh -Y user@server  # Trusted X11 forwarding
# Or
ssh -X -C user@server  # With compression

# Install X11 server on client if missing
sudo apt-get install xorg-dev  # Linux
# Or install Xming/VcXsrv on Windows
```

## 🐳 Method 3: Web-Based Viewer (Alternative)

### Convert to Web Format
```bash
# On server, convert to text/CSV format
nsys stats --output=csv vector_add_timeline.nsys-rep > timeline.csv
ncu --csv --log-file kernel_metrics.csv ./vector_add

# Download CSV files and view in Excel/browser
```

## 📈 Method 4: Text-Based Analysis (No GUI)

### Comprehensive Text Reports
```bash
# On server - generate detailed text reports
nsys stats vector_add_timeline.nsys-rep
nsys stats --report cuda_api_sum vector_add_timeline.nsys-rep
nsys stats --report kernel_sum vector_add_timeline.nsys-rep

# Nsight Compute text output
ncu --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed --csv ./vector_add
```

## 🛠️ Complete Workflow Example

Here's a complete script for server-side profiling with client viewing:

```bash
#!/bin/bash
# remote_profile.sh - Run on server

echo "=== Remote CUDA Profiling Workflow ==="

# Build programs
make vector_add vector_add_optimized

# Generate comprehensive profiles
echo "Generating Nsight Systems timeline..."
nsys profile --output=vector_add_timeline --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    ./vector_add

echo "Generating Nsight Compute detailed analysis..."
ncu --export vector_add_detailed --force-overwrite \
    --set full \
    --target-processes all \
    ./vector_add

echo "Generating text reports..."
nsys stats --output=csv --force-overwrite=true vector_add_timeline.nsys-rep > vector_add_stats.csv
nsys stats --report kernel_sum vector_add_timeline.nsys-rep > kernel_summary.txt

echo "Files generated:"
ls -la vector_add_timeline.nsys-rep vector_add_detailed.ncu-rep vector_add_stats.csv kernel_summary.txt

echo ""
echo "To view on your client laptop:"
echo "1. Download files:"
echo "   scp user@$(hostname):$(pwd)/vector_add_timeline.nsys-rep ."
echo "   scp user@$(hostname):$(pwd)/vector_add_detailed.ncu-rep ."
echo ""
echo "2. Install CUDA Toolkit on client (includes Nsight tools)"
echo ""
echo "3. Open profiles:"
echo "   nsys-ui vector_add_timeline.nsys-rep"
echo "   ncu-ui vector_add_detailed.ncu-rep"
```

## 🎯 Recommended Workflow

### For Best Experience:
1. **Profile on server** (has GPU)
2. **Download profile files** to client
3. **Analyze on client** with GUI tools

### Quick Commands:
```bash
# On server
cd YOUR_WORKSPACE/cuda_examples
./remote_profile.sh

# On client (replace with your server details)
scp user@your-server:YOUR_WORKSPACE/cuda_examples/*.nsys-rep .
scp user@your-server:YOUR_WORKSPACE/cuda_examples/*.ncu-rep .

# Open with GUI tools
nsys-ui vector_add_timeline.nsys-rep
ncu-ui vector_add_detailed.ncu-rep
```

## 📱 Alternative: Browser-Based Viewing

### Convert to Web Format
```bash
# Generate HTML reports (some tools support this)
nsys stats --format html vector_add_timeline.nsys-rep > timeline.html

# View in any browser
python3 -m http.server 8080  # Serve locally
# Open http://localhost:8080/timeline.html
```

## 🔧 Tool Installation Locations

### NVIDIA Nsight Tools Download:
- **Official**: https://developer.nvidia.com/nsight-tools
- **Nsight Systems**: Standalone download available
- **Nsight Compute**: Included with CUDA Toolkit
- **Nsight Graphics**: For graphics profiling

### Client Installation:
```bash
# Ubuntu/Debian
sudo apt install cuda-nsight-systems-12-3 cuda-nsight-compute-12-3

# CentOS/RHEL
sudo yum install cuda-nsight-systems-12-3 cuda-nsight-compute-12-3

# Manual download and extract
wget https://developer.nvidia.com/nsight-systems
```

## 🎉 What You'll See in GUI

### Nsight Systems Timeline:
- GPU kernel execution timeline
- Memory transfers (H2D, D2H)
- CPU activity correlation
- API call timings

### Nsight Compute Details:
- Kernel performance metrics
- Memory bandwidth utilization
- Occupancy analysis
- Bottleneck identification
- Optimization suggestions

## 💡 Pro Tips

1. **Profile small examples first** - easier to understand
2. **Use meaningful output names** - `--output=project_kernel_date`
3. **Compress large profiles** - `gzip *.nsys-rep` before download
4. **Save analysis scripts** - automate repetitive profiling
5. **Compare before/after** - profile optimizations

The **download and analyze locally** approach is usually best for server-client setups! 🚀
