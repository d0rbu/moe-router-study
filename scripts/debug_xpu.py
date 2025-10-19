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
    ipex_available = True
except ImportError as e:
    print(f"❌ Intel Extension for PyTorch not found: {e}")
    ipex_available = False

# Check XPU availability
xpu_available = th.xpu.is_available() if hasattr(th, "xpu") else False
print(f"\nXPU available: {xpu_available}")

if xpu_available:
    try:
        device_count = th.xpu.device_count()
        print(f"XPU device count: {device_count}")

        for i in range(device_count):
            try:
                device_name = th.xpu.get_device_name(i)
                print(f"  Device {i}: {device_name}")
            except Exception as e:
                print(f"  Device {i}: Error getting name - {e}")

        # Test basic XPU operations
        try:
            device = th.device("xpu:0")
            print("\n🧪 Testing basic XPU operations...")
            x = th.randn(10, 10, device=device)
            y = th.randn(10, 10, device=device)
            z = th.matmul(x, y)
            print("✅ Basic XPU tensor operations work")
            print(f"   Test tensor shape: {z.shape}")
            print(f"   Test tensor device: {z.device}")
            print(f"   Test tensor dtype: {z.dtype}")

            # Test memory operations
            memory_allocated = th.xpu.memory_allocated(device)
            memory_reserved = th.xpu.memory_reserved(device)
            print(f"   Memory allocated: {memory_allocated / 1024**2:.2f} MB")
            print(f"   Memory reserved: {memory_reserved / 1024**2:.2f} MB")

        except Exception as e:
            print(f"❌ XPU tensor operations failed: {e}")

    except Exception as e:
        print(f"❌ Error accessing XPU devices: {e}")
else:
    print("❌ XPU not available")

    # Check for common issues
    print("\n🔧 Troubleshooting:")
    if not ipex_available:
        print("1. ❌ Intel Extension for PyTorch is not installed")
        print("   Install with: pip install intel-extension-for-pytorch")
    else:
        print("1. ✅ Intel Extension for PyTorch is installed")

    if not hasattr(th, "xpu"):
        print("2. ❌ PyTorch XPU module not found")
        print("   This suggests XPU support is not compiled into PyTorch")
    else:
        print("2. ✅ PyTorch XPU module found")

    print("3. Check if Intel GPU drivers are installed")
    print("4. Check if XPU runtime is available")
    print("5. Verify environment variables (see check_intel_tools.sh)")

# Check CUDA for comparison
print("\n📊 For comparison:")
print(f"CUDA available: {th.cuda.is_available()}")
if th.cuda.is_available():
    print(f"CUDA device count: {th.cuda.device_count()}")
    for i in range(th.cuda.device_count()):
        print(f"  CUDA Device {i}: {th.cuda.get_device_name(i)}")

# Check MPS (Apple Silicon)
if hasattr(th.backends, "mps") and th.backends.mps.is_available():
    print(f"MPS available: {th.backends.mps.is_available()}")

print("\n" + "=" * 50)
print("🏁 Debug complete!")

# Exit with appropriate code
if xpu_available:
    print("✅ XPU is working correctly!")
    sys.exit(0)
else:
    print("❌ XPU is not available - check troubleshooting steps above")
    sys.exit(1)
