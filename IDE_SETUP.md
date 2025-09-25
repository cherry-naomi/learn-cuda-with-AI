# CUDA Development IDE Setup

This document explains how to set up proper IDE support for CUDA development with "Go to Declaration" and "Go to Definition" functionality.

## Problem
- "Go to Declaration" and "Go to Definition" not working in CUDA files
- Missing IntelliSense for CUDA functions
- No proper language server support

## Solution

### 1. VS Code Configuration (Already Done)
The following files have been created:
- `.vscode/c_cpp_properties.json` - C++ IntelliSense configuration
- `.vscode/settings.json` - VS Code settings for CUDA
- `compile_commands.json` - Compilation database for better IntelliSense

### 2. Install Language Server (Recommended)
Run the installation script:
```bash
bash install_clangd.sh
```

Or manually install:
```bash
sudo apt update
sudo apt install -y clangd clang-tools clang-format
```

### 3. VS Code Extensions
Install these VS Code extensions:
- C/C++ (ms-vscode.cpptools)
- C/C++ Extension Pack (ms-vscode.cpptools-extension-pack)

### 4. Restart and Reload
1. Restart VS Code
2. Press `Ctrl+Shift+P` and run "C/C++: Reload IntelliSense Database"
3. Wait for IntelliSense to finish indexing

## Testing
1. Open `src/softmax/softmax_basic.cu`
2. Right-click on `cudaMalloc` or `__global__` and select "Go to Definition"
3. Try "Go to Declaration" on function names like `softmax_basic`

## Troubleshooting

### If IntelliSense still doesn't work:
1. Check that CUDA is properly installed: `nvcc --version`
2. Verify include paths in `.vscode/c_cpp_properties.json`
3. Reload IntelliSense database: `Ctrl+Shift+P` → "C/C++: Reload IntelliSense Database"
4. Check VS Code output panel for C/C++ errors

### If clangd is not available:
- The VS Code C++ extension will still provide basic IntelliSense
- For full functionality, install clangd using the provided script

## File Associations
CUDA files (`.cu`, `.cuh`) are now associated with C++ language mode for proper syntax highlighting and IntelliSense.

## Compilation Database
The `compile_commands.json` file contains the exact compilation commands used by the Makefile, which helps the language server understand your project structure and dependencies.
