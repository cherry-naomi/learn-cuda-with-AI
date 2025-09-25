#!/bin/bash

# Script to install clangd for CUDA development
# Run with: bash install_clangd.sh

echo "Installing clangd for CUDA development..."

# Update package list
sudo apt update

# Install clangd and related tools
sudo apt install -y clangd clang-tools clang-format

# Verify installation
if command -v clangd &> /dev/null; then
    echo "✅ clangd installed successfully!"
    clangd --version
else
    echo "❌ clangd installation failed"
    exit 1
fi

# Install VS Code C++ extension if not already installed
echo "Installing VS Code C++ extension..."
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cpptools-extension-pack

echo "✅ Setup complete! Restart VS Code for full functionality."
echo ""
echo "Usage:"
echo "1. Open VS Code in this directory"
echo "2. Press Ctrl+Shift+P and run 'C/C++: Reload IntelliSense Database'"
echo "3. Try 'Go to Definition' (F12) on CUDA functions"
