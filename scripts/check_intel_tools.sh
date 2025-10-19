#!/bin/bash
echo "🔍 Checking Intel GPU monitoring tools and environment..."
echo "=================================================="

echo "📋 System Information:"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo ""

echo "🔧 Checking for Intel GPU tools:"
echo "----------------------------------"

echo "Checking for xpu-smi:"
if command -v xpu-smi &> /dev/null; then
    echo "✅ xpu-smi found at: $(which xpu-smi)"
    echo "Running xpu-smi discovery:"
    xpu-smi discovery 2>&1 || echo "❌ xpu-smi discovery failed"
else
    echo "❌ xpu-smi not found in PATH"
fi

echo ""
echo "Checking for intel_gpu_top:"
if command -v intel_gpu_top &> /dev/null; then
    echo "✅ intel_gpu_top found at: $(which intel_gpu_top)"
else
    echo "❌ intel_gpu_top not found in PATH"
fi

echo ""
echo "Checking for Level Zero tools:"
if command -v ze_info &> /dev/null; then
    echo "✅ ze_info found at: $(which ze_info)"
    echo "Running ze_info:"
    ze_info 2>&1 || echo "❌ ze_info failed"
else
    echo "❌ ze_info not found in PATH"
fi

if command -v level_zero_tests &> /dev/null; then
    echo "✅ level_zero_tests found at: $(which level_zero_tests)"
else
    echo "❌ level_zero_tests not found in PATH"
fi

echo ""
echo "🌍 Environment Variables:"
echo "-------------------------"
echo "ZE_ENABLE_PCI_ID_DEVICE_ORDER: ${ZE_ENABLE_PCI_ID_DEVICE_ORDER:-not set}"
echo "SYCL_DEVICE_FILTER: ${SYCL_DEVICE_FILTER:-not set}"
echo "ONEAPI_ROOT: ${ONEAPI_ROOT:-not set}"
echo "INTEL_GPU_DRIVER_PATH: ${INTEL_GPU_DRIVER_PATH:-not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"

echo ""
echo "🖥️ Hardware Detection:"
echo "----------------------"
echo "Checking for Intel GPU devices in /dev/dri/:"
if [ -d "/dev/dri" ]; then
    ls -la /dev/dri/ 2>/dev/null | grep -E "(card|render)" || echo "No GPU devices found in /dev/dri/"
else
    echo "❌ /dev/dri/ directory not found"
fi

echo ""
echo "Checking for Intel GPU in lspci:"
lspci | grep -i "vga\|display\|3d" | grep -i intel || echo "No Intel GPU found in lspci"

echo ""
echo "🔌 Kernel Modules:"
echo "------------------"
echo "Checking loaded Intel GPU modules:"
lsmod | grep -E "(i915|xe|intel)" || echo "No Intel GPU kernel modules found"

echo ""
echo "📦 Package Information:"
echo "-----------------------"
echo "Checking for Intel GPU packages:"

# Check for common Intel GPU packages
packages_to_check=(
    "intel-gpu-tools"
    "intel-level-zero-gpu"
    "intel-opencl-icd"
    "intel-media-va-driver"
    "libdrm-intel1"
    "xpu-smi"
)

for pkg in "${packages_to_check[@]}"; do
    # Check if dpkg is available (Debian/Ubuntu systems)
    if command -v dpkg >/dev/null 2>&1; then
        if dpkg -l | grep -q "$pkg" 2>/dev/null; then
            echo "✅ $pkg is installed"
            continue
        fi
    fi
    
    # Check if rpm is available (RHEL/CentOS/SUSE systems)
    if command -v rpm >/dev/null 2>&1; then
        if rpm -qa | grep -q "$pkg" 2>/dev/null; then
            echo "✅ $pkg is installed (RPM)"
            continue
        fi
    fi
    
    echo "❌ $pkg not found"
done

echo ""
echo "🐍 Python Environment:"
echo "----------------------"
echo "Python version: $(python --version 2>&1)"
echo "Python path: $(which python)"

echo ""
echo "Checking Python packages:"
python -c "
import sys
packages = ['torch', 'intel_extension_for_pytorch']
for pkg in packages:
    try:
        module = __import__(pkg)
        if hasattr(module, '__version__'):
            print(f'✅ {pkg}: {module.__version__}')
        else:
            print(f'✅ {pkg}: installed (no version info)')
    except ImportError:
        print(f'❌ {pkg}: not installed')
"

echo ""
echo "=================================================="
echo "🏁 Intel GPU environment check complete!"
echo ""
echo "💡 If XPU is not working, common solutions:"
echo "1. Install Intel GPU drivers and Level Zero runtime"
echo "2. Set environment variables:"
echo "   export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1"
echo "   export SYCL_DEVICE_FILTER=level_zero:gpu"
echo "3. Install Intel Extension for PyTorch:"
echo "   pip install intel-extension-for-pytorch"
echo "4. Check if Intel GPU kernel modules are loaded"
