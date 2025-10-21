#!/bin/bash

# Script to install Intel Extension for PyTorch with XPU support
# This script manually installs the XPU version since it's not available through standard package managers

set -e  # Exit on any error

echo "🚀 Installing Intel Extension for PyTorch with XPU support..."

# Check if we're in a uv project
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: This script must be run from the project root directory (where pyproject.toml is located)"
    exit 1
fi

# Check Python version compatibility
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Using Python $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "   ⚠️  Python 3.13 detected - this may cause compatibility issues with some packages"
fi

# Remove existing versions (CPU, CUDA, or XPU)
echo "📦 Removing existing PyTorch and Intel Extension installations..."
uv remove torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt 2>/dev/null || echo "   (No existing installation found)"

# Clean up any conflicting system-wide packages that might interfere
echo "🧹 Cleaning up potential conflicting system packages..."
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt numpy psutil packaging -y 2>/dev/null || echo "   (No conflicting system packages found)"

# Also clean up from the uv environment to ensure fresh install
uv remove torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt numpy psutil packaging 2>/dev/null || echo "   (No conflicting uv packages found)"

# Use official Intel documentation versions
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
IPEX_VERSION="2.8.10+xpu"
ONECCL_VERSION="2.8.0+xpu"

# Install PyTorch XPU version first (following Intel's official docs)
echo "🔥 Installing PyTorch XPU version..."
echo "   Following Intel's official installation guide"

# Try multiple installation approaches
INSTALL_SUCCESS=false

# Approach 1: Install PyTorch XPU version first, then Intel Extension
echo "   Trying approach 1: PyTorch XPU + Intel Extension..."
if uv add "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url https://download.pytorch.org/whl/xpu --reinstall 2>/dev/null && \
   uv add "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ 2>/dev/null; then
    INSTALL_SUCCESS=true
    echo "   ✅ Approach 1 succeeded"
fi

# Approach 2: Use pip fallback if uv fails (common with specialized indexes)
if [ "$INSTALL_SUCCESS" = false ]; then
    echo "   Trying approach 2: pip fallback for specialized indexes..."
    if uv run pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
        --index-url https://download.pytorch.org/whl/xpu 2>/dev/null && \
       uv run pip install "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
        --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ 2>/dev/null; then
        INSTALL_SUCCESS=true
        echo "   ✅ Approach 2 succeeded"
    fi
fi

# Approach 3: Try with frozen flag to skip dependency resolution
if [ "$INSTALL_SUCCESS" = false ]; then
    echo "   Trying approach 3: frozen installation..."
    if uv add "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/xpu --frozen 2>/dev/null && \
       uv add "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
        --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --frozen 2>/dev/null; then
        INSTALL_SUCCESS=true
        echo "   ✅ Approach 3 succeeded"
    fi
fi

if [ "$INSTALL_SUCCESS" = false ]; then
    echo "❌ All installation approaches failed!"
    echo "   This may be due to:"
    echo "   - Network connectivity issues"
    echo "   - Authentication issues with Intel's package index"
    echo "   - Dependency resolution conflicts"
    echo ""
    echo "   Manual installation commands (run these individually):"
    echo "   uv run pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/xpu"
    echo "   uv run pip install intel-extension-for-pytorch==$IPEX_VERSION oneccl_bind_pt==$ONECCL_VERSION --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    exit 1
fi

# Verify installation
echo "🔍 Verifying installation..."

# Change to a different directory to avoid numpy source directory issues
cd /tmp

if uv run python -c "
import sys
try:
    import intel_extension_for_pytorch as ipex
    print(f'✅ IPEX version: {ipex.__version__}')
    
    import torch
    if hasattr(torch, 'xpu'):
        if torch.xpu.is_available():
            print(f'✅ XPU available: {torch.xpu.is_available()}')
            print(f'✅ XPU device count: {torch.xpu.device_count()}')
        else:
            print('⚠️  XPU not available (this is expected if not running on Intel XPU hardware)')
    else:
        print('⚠️  torch.xpu module not found (XPU support may not be properly installed)')
    
    # Check if this is the XPU version
    if '+xpu' in ipex.__version__:
        print('✅ XPU version successfully installed')
    else:
        print('❌ Warning: CPU version installed instead of XPU version')
        sys.exit(1)
        
except ImportError as e:
    print(f'❌ Error importing Intel Extension for PyTorch: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error during verification: {e}')
    sys.exit(1)
"; then
    echo "🏁 Installation complete!"
    echo ""
    echo "💡 Note: XPU functionality requires Intel XPU hardware and drivers."
    echo "   On systems without Intel XPU hardware, the library will install but XPU features won't be available."
else
    echo "❌ Installation verification failed!"
    echo "   Please check the error messages above and try running the script again."
    exit 1
fi

# Change back to the original directory
cd - > /dev/null
