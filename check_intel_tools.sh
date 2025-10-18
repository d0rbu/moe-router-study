#!/bin/bash
echo "🔍 Checking Intel GPU monitoring tools..."
echo "=" * 50

echo "Checking for xpu-smi:"
if command -v xpu-smi &> /dev/null; then
    echo "✅ xpu-smi found"
    xpu-smi discovery
else
    echo "❌ xpu-smi not found"
fi

echo -e "\nChecking for intel_gpu_top:"
if command -v intel_gpu_top &> /dev/null; then
    echo "✅ intel_gpu_top found"
else
    echo "❌ intel_gpu_top not found"
fi

echo -e "\nChecking for level-zero devices:"
if command -v ze_info &> /dev/null; then
    echo "✅ ze_info found"
    ze_info
else
    echo "❌ ze_info not found"
fi

echo -e "\nChecking environment variables:"
echo "ZE_ENABLE_PCI_ID_DEVICE_ORDER: ${ZE_ENABLE_PCI_ID_DEVICE_ORDER:-not set}"
echo "SYCL_DEVICE_FILTER: ${SYCL_DEVICE_FILTER:-not set}"

echo -e "\nChecking for Intel GPU devices in /dev:"
ls -la /dev/dri/ 2>/dev/null || echo "No /dev/dri/ found"

echo -e "\nChecking loaded Intel GPU modules:"
lsmod | grep -E "(i915|xe)" || echo "No Intel GPU kernel modules found"

