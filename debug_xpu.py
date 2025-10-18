#!/usr/bin/env python3
"""Debug script to check XPU availability and device detection."""

import sys

import torch as th

print("🔍 PyTorch XPU Debug Information")
print("=" * 50)

# Check PyTorch version
print(f"PyTorch version: {th.__version__}")

# Check if XPU extension is available
try:
    import intel_extension_for_pytorch as ipex

    print(f"✅ Intel Extension for PyTorch version: {ipex.__version__}")
except ImportError as e:
    print(f"❌ Intel Extension for PyTorch not found: {e}")
    sys.exit(1)

# Check XPU availability
print(f"\nXPU available: {th.xpu.is_available()}")

if th.xpu.is_available():
    print(f"XPU device count: {th.xpu.device_count()}")

    for i in range(th.xpu.device_count()):
        print(f"  Device {i}: {th.xpu.get_device_name(i)}")

    # Test basic XPU operations
    try:
        device = th.device("xpu:0")
        x = th.randn(10, 10, device=device)
        y = th.randn(10, 10, device=device)
        z = th.matmul(x, y)
        print("✅ Basic XPU tensor operations work")
        print(f"   Test tensor shape: {z.shape}")
        print(f"   Test tensor device: {z.device}")
    except Exception as e:
        print(f"❌ XPU tensor operations failed: {e}")
else:
    print("❌ XPU not available")

    # Check for common issues
    print("\n🔧 Troubleshooting:")
    print("1. Check if Intel GPU drivers are installed")
    print("2. Check if Intel Extension for PyTorch is properly installed")
    print("3. Check if XPU runtime is available")

# Check CUDA for comparison
print(f"\nFor comparison - CUDA available: {th.cuda.is_available()}")
if th.cuda.is_available():
    print(f"CUDA device count: {th.cuda.device_count()}")

print("\n" + "=" * 50)
