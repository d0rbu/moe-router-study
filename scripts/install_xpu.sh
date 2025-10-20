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

# Remove existing CPU-only version
echo "📦 Removing existing Intel Extension for PyTorch and OneCCL..."
uv remove intel-extension-for-pytorch oneccl_bind_pt 2>/dev/null || echo "   (No existing installation found)"

# Clean up any conflicting system-wide packages that might interfere
echo "🧹 Cleaning up potential conflicting system packages..."
pip uninstall intel-extension-for-pytorch oneccl_bind_pt numpy psutil packaging -y 2>/dev/null || echo "   (No conflicting system packages found)"

# Also clean up from the uv environment to ensure fresh install
uv remove intel-extension-for-pytorch oneccl_bind_pt numpy psutil packaging 2>/dev/null || echo "   (No conflicting uv packages found)"

# Ensure PyTorch is installed first
echo "🔥 Installing PyTorch..."
uv add torch --force

# Use official Intel documentation versions
IPEX_VERSION="2.8.10+xpu"
ONECCL_VERSION="2.8.0+xpu"

# Install XPU versions from Intel's index
echo "⚡ Installing Intel Extension for PyTorch $IPEX_VERSION and OneCCL $ONECCL_VERSION..."

# Try multiple installation approaches
INSTALL_SUCCESS=false

# Approach 1: Direct uv add
echo "   Trying approach 1: uv add..."
if uv add "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
    --force 2>/dev/null; then
    INSTALL_SUCCESS=true
    echo "   ✅ Approach 1 succeeded"
fi

# Approach 2: Try with different URL format if first approach failed
if [ "$INSTALL_SUCCESS" = false ]; then
    echo "   Trying approach 2: alternative index URL..."
    if uv add "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
        --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
        --force 2>/dev/null; then
        INSTALL_SUCCESS=true
        echo "   ✅ Approach 2 succeeded"
    fi
fi

# Approach 3: Try with basic uv add if still failed
if [ "$INSTALL_SUCCESS" = false ]; then
    echo "   Trying approach 3: basic uv add..."
    if uv add "intel-extension-for-pytorch==$IPEX_VERSION" "oneccl_bind_pt==$ONECCL_VERSION" \
        --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/; then
        INSTALL_SUCCESS=true
        echo "   ✅ Approach 3 succeeded"
    fi
fi

if [ "$INSTALL_SUCCESS" = false ]; then
    echo "❌ All installation approaches failed!"
    echo "   Please check your internet connection and try again."
    echo "   You may also try installing manually with:"
    echo "   uv add intel-extension-for-pytorch==$IPEX_VERSION oneccl_bind_pt==$ONECCL_VERSION --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
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
