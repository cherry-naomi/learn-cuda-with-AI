# CUDA Installation Guide

Since you're getting "nvcc: Command not found", you need to install the CUDA Toolkit first.

## Check if you have an NVIDIA GPU

First, check if you have an NVIDIA GPU:

```bash
lspci | grep -i nvidia
```

If you don't see any NVIDIA GPU, these examples won't work on your system.

## Installation Options

### Option 1: Install CUDA Toolkit from NVIDIA (Recommended)

1. **Check your GPU compatibility and driver version:**
   ```bash
   nvidia-smi
   ```

2. **Download CUDA Toolkit from NVIDIA:**
   - Go to: https://developer.nvidia.com/cuda-downloads
   - Select Linux → x86_64 → Ubuntu → your version
   - Choose "deb (local)" for easier installation

3. **Install (example for Ubuntu 20.04):**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.1-545.23.08-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.1-545.23.08-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

4. **Add CUDA to your PATH:**
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

### Option 2: Install via Package Manager (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-470

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

### Option 3: Using Conda/Mamba

```bash
# Install cudatoolkit
conda install cudatoolkit-dev

# Or with mamba
mamba install cudatoolkit-dev
```

## Verify Installation

After installation, verify that CUDA is working:

```bash
# Check nvcc compiler
nvcc --version

# Check GPU status
nvidia-smi

# Test compilation
cd YOUR_WORKSPACE/cuda_examples
make vector_add
```

## Alternative: Run Without CUDA

If you don't have an NVIDIA GPU or can't install CUDA, I can create a CPU-only version that simulates the threading concepts for educational purposes. Just let me know!

## Troubleshooting

1. **"nvidia-smi" not found**: Install NVIDIA drivers first
2. **"nvcc" not found after installation**: Check PATH and restart terminal
3. **Compilation errors**: Ensure your GPU supports the architecture specified in Makefile (-arch=sm_50)
4. **Runtime errors**: Check that your GPU has sufficient memory

## Update Makefile for Different GPU Architectures

You might need to change the `-arch=` flag in the Makefile based on your GPU:

- GTX 900 series: `-arch=sm_52`
- GTX 10 series: `-arch=sm_61`
- RTX 20 series: `-arch=sm_75`
- RTX 30 series: `-arch=sm_86`
- RTX 40 series: `-arch=sm_89`

Check your GPU's compute capability at: https://developer.nvidia.com/cuda-gpus
