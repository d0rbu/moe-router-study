#!/bin/bash

# Script to install Intel Extension for PyTorch with XPU support
# This script manually installs the XPU version since it's not available through standard package managers

echo "🚀 Installing Intel Extension for PyTorch with XPU support..."

# Remove existing CPU-only version
echo "📦 Removing existing Intel Extension for PyTorch..."
uv run pip uninstall intel-extension-for-pytorch -y

# Install XPU version from Intel's index
echo "⚡ Installing XPU version..."
uv run pip install intel-extension-for-pytorch==2.8.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
    --force-reinstall

# Verify installation
echo "🔍 Verifying installation..."
uv run python -c "
import intel_extension_for_pytorch as ipex
print(f'✅ IPEX version: {ipex.__version__}')

import torch
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f'✅ XPU available: {torch.xpu.is_available()}')
    print(f'✅ XPU device count: {torch.xpu.device_count()}')
else:
    print('⚠️  XPU not available (this is expected if not running on Intel XPU hardware)')
"

echo "🏁 Installation complete!"
echo ""
echo "💡 Note: XPU functionality requires Intel XPU hardware and drivers."
echo "   On systems without Intel XPU hardware, the library will install but XPU features won't be available."
