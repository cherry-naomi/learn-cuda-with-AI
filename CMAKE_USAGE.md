# CMake Usage Guide

This project now supports both Make and CMake build systems. CMake provides better IDE integration, cross-platform support, and more advanced build features.

## 🚀 Quick Start

### Basic CMake Build
```bash
# Create build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build all examples
make -j$(nproc)

# Run all examples
make run_all
```

### Using CMake Presets (Recommended)
```bash
# Configure with preset
cmake --preset=default

# Build with preset
cmake --build --preset=default

# Run examples
cd build && make run_all
```

## 📋 Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Standard Release build | General development |
| `debug` | Debug build with device debugging | Debugging CUDA kernels |
| `release` | Optimized Release build | Performance testing |
| `sm80-only` | Ampere architecture only | RTX 30/40 series GPUs |

## 🎯 Build Targets

### All Examples
```bash
make all                    # Build all examples
make run_all               # Build and run all examples
```

### Individual Examples
```bash
# Vector Addition Examples
make vector_add            # Basic vector addition
make run1                  # Run basic example
make run2                  # Run 2D example
make run3                  # Run threading explanation (LEARNING!)
make run4                  # Run hardware mapping
make run5                  # Run shared memory (BLOCKS!)
make run6                  # Run block isolation
make run7                  # Run performance analysis
make run8                  # Run insufficient threads
make run9                  # Run block essence (中文)

# Softmax Examples
make softmax_basic         # Basic softmax
make softmax_optimized     # Optimized softmax
make softmax_unittest      # Unit tests
make run10                 # Run basic softmax
make run11                 # Run optimized softmax
make test                  # Run unit tests (alias for run12)
```

### Performance Analysis
```bash
make profile               # Run comprehensive profiling
```

## 🔧 Advanced Usage

### Custom CUDA Architecture
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
```

### Debug Build
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### Install to System
```bash
cmake --build . --target install
```

## 🆚 CMake vs Make

| Feature | CMake | Make |
|---------|-------|------|
| **IDE Integration** | ✅ Excellent | ❌ Limited |
| **Cross-platform** | ✅ Yes | ❌ Linux only |
| **Dependency Management** | ✅ Automatic | ❌ Manual |
| **Build Types** | ✅ Debug/Release | ❌ Single type |
| **Architecture Detection** | ✅ Automatic | ❌ Manual |
| **Installation** | ✅ `make install` | ❌ Manual copy |
| **Presets** | ✅ Yes | ❌ No |
| **Parallel Builds** | ✅ `-j` | ✅ `-j` |

## 🎓 Learning Path

### For CUDA Beginners
1. **`make run3`** - Threading explanation (ESSENTIAL!)
2. **`make run4`** - Hardware mapping
3. **`make run5`** - Shared memory concepts
4. **`make run6`** - Block isolation
5. **`make run9`** - Block essence (Chinese)
6. **`make run8`** - Thread coverage
7. **`make run1`** - Basic vector addition
8. **`make run2`** - 2D grids

### For Softmax Learning
1. **`make run10`** - Basic softmax
2. **`make run11`** - Optimized softmax
3. **`make test`** - Unit tests

### For Performance Analysis
1. **`make run7`** - Performance analysis
2. **`make profile`** - Comprehensive profiling

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.0
   export PATH=$CUDA_HOME/bin:$PATH
   ```

2. **Architecture not supported**
   ```bash
   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
   ```

3. **Debug build issues**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS_DEBUG="-g -G" ..
   ```

### Clean Build
```bash
rm -rf build/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 📁 Project Structure

```
cuda_examples/
├── CMakeLists.txt          # Main CMake configuration
├── CMakePresets.json       # CMake presets
├── CMAKE_USAGE.md          # This guide
├── Makefile               # Original Make build system
├── src/
│   ├── vector_add/        # Vector addition examples
│   └── softmax/           # Softmax examples
└── build/                 # CMake build directory
```

## 🎉 Benefits of CMake

- **Better IDE Support**: Full IntelliSense, debugging, and navigation
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Dependency Management**: Automatically finds CUDA
- **Build Variants**: Debug, Release, custom configurations
- **Installation**: `make install` for system-wide installation
- **Presets**: Easy switching between configurations
- **Modern**: Industry standard for C++/CUDA projects

Choose CMake for professional development, or stick with Make for simple builds!
