#!/bin/bash

# Intel XPU Installation Script
# Based on Intel's official installation guide and user testing

set -e

echo "🚀 Intel XPU Installation Script"
echo "================================"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📍 Using Python $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.12" && "$PYTHON_VERSION" != "3.13" ]]; then
    echo "⚠️  Warning: Python $PYTHON_VERSION detected. Intel XPU is tested with Python 3.12 and 3.13"
fi

# Intel XPU package versions (from official docs)
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0" 
TORCHAUDIO_VERSION="2.8.0"
IPEX_VERSION="2.8.10+xpu"
ONECCL_VERSION="2.8.0+xpu"

echo "📦 Package versions:"
echo "   - PyTorch: $TORCH_VERSION"
echo "   - Intel Extension: $IPEX_VERSION"
echo "   - OneCCL: $ONECCL_VERSION"

# Step 1: Install PyTorch XPU version
echo ""
echo "🔥 Step 1: Installing PyTorch XPU version..."
uv pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url https://download.pytorch.org/whl/xpu

if [ $? -ne 0 ]; then
    echo "❌ Failed to install PyTorch XPU version"
    exit 1
fi

echo "✅ PyTorch XPU version installed successfully"

# Step 2: Install Intel Extension for PyTorch
echo ""
echo "⚡ Step 2: Installing Intel Extension for PyTorch..."
uv pip install "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Intel Extension for PyTorch"
    echo "   This might be due to:"
    echo "   - Network connectivity issues"
    echo "   - Authentication issues with Intel's package index"
    echo "   - Platform compatibility issues"
    echo ""
    echo "   You can try installing manually with pip:"
    echo "   pip install intel-extension-for-pytorch==$IPEX_VERSION oneccl_bind_pt==$ONECCL_VERSION --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    exit 1
fi

echo "✅ Intel Extension for PyTorch installed successfully"

# Step 3: Verification
echo ""
echo "🔍 Step 3: Verifying installation..."

# Test basic imports
echo "   Testing imports..."
if uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    echo "   ✅ PyTorch import successful"
else
    echo "   ❌ PyTorch import failed"
    exit 1
fi

if uv run python -c "import intel_extension_for_pytorch as ipex; print(f'Intel Extension version: {ipex.__version__}')" 2>/dev/null; then
    echo "   ✅ Intel Extension import successful"
else
    echo "   ❌ Intel Extension import failed"
    exit 1
fi

# Test XPU availability (if hardware is present)
echo "   Testing XPU device detection..."
XPU_COUNT=$(uv run python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null || echo "0")
if [ "$XPU_COUNT" -gt 0 ]; then
    echo "   ✅ XPU devices detected: $XPU_COUNT"
    uv run python -c "import torch; [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]" 2>/dev/null || true
else
    echo "   ⚠️  No XPU devices detected (this is normal if you don't have Intel XPU hardware)"
fi

echo ""
echo "🎉 Intel XPU installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure you have Intel XPU drivers installed (see Intel's installation guide)"
echo "   2. Test your installation with: uv run python -c \"import torch; import intel_extension_for_pytorch as ipex; print('Ready!')\""
echo "   3. Check the Intel Extension for PyTorch documentation for usage examples"
