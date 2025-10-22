#!/usr/bin/env python3
"""Debug script to diagnose Intel XPU detection issues."""

import os
import sys
import traceback
from typing import Any


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def safe_import_and_test(module_name: str, test_func=None) -> Any:
    """Safely import a module and optionally test it."""
    try:
        if module_name == "torch":
            import torch

            module = torch
        elif module_name == "intel_extension_for_pytorch":
            import intel_extension_for_pytorch as ipex

            module = ipex
        else:
            module = __import__(module_name)

        print(f"✅ {module_name} imported successfully")
        if hasattr(module, "__version__"):
            print(f"   Version: {module.__version__}")

        if test_func:
            test_func(module)

        return module
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  {module_name} imported but test failed: {e}")
        traceback.print_exc()
        return None


def test_torch(torch_module) -> None:
    """Test torch functionality."""
    print(f"   CUDA available: {torch_module.cuda.is_available()}")
    if torch_module.cuda.is_available():
        print(f"   CUDA device count: {torch_module.cuda.device_count()}")

    # Test XPU
    if hasattr(torch_module, "xpu"):
        print("   XPU module exists: True")
        try:
            print(f"   XPU available: {torch_module.xpu.is_available()}")
            if torch_module.xpu.is_available():
                print(f"   XPU device count: {torch_module.xpu.device_count()}")
        except Exception as e:
            print(f"   XPU test failed: {e}")
    else:
        print("   XPU module exists: False")


def test_ipex(ipex_module) -> None:
    """Test Intel Extension for PyTorch."""
    if hasattr(ipex_module, "__version__"):
        print(f"   IPEX Version: {ipex_module.__version__}")

    # Test XPU functionality
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu:0")
            tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
            print(f"   XPU tensor creation: ✅ {tensor}")
        else:
            print("   XPU tensor creation: ❌ XPU not available")
    except Exception as e:
        print(f"   XPU tensor creation failed: {e}")


def test_device_backend() -> None:
    """Test our device backend abstraction."""
    try:
        from core.device import get_backend

        print("Testing CUDA backend:")
        cuda_backend = get_backend("cuda")
        print(f"   CUDA backend: {cuda_backend}")
        print(f"   CUDA available: {cuda_backend.is_available()}")
        if cuda_backend.is_available():
            print(f"   CUDA device count: {cuda_backend.device_count()}")

        print("\nTesting XPU backend:")
        xpu_backend = get_backend("xpu")
        print(f"   XPU backend: {xpu_backend}")
        print(f"   XPU available: {xpu_backend.is_available()}")
        if xpu_backend.is_available():
            print(f"   XPU device count: {xpu_backend.device_count()}")
        else:
            print("   ❌ XPU backend reports not available!")

    except Exception as e:
        print(f"❌ Device backend test failed: {e}")
        traceback.print_exc()


def check_environment() -> None:
    """Check environment variables and system info."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check relevant environment variables
    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "XPU_VISIBLE_DEVICES",
        "INTEL_XPU_VISIBLE_DEVICES",
        "ONEAPI_ROOT",
        "INTEL_EXTENSION_FOR_PYTORCH_PATH",
        "LD_LIBRARY_PATH",
        "PATH",
    ]

    print("\nEnvironment variables:")
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"   {var}: {value}")


def main() -> None:
    """Main debug function."""
    print("🔍 Intel XPU Detection Debug Script")
    print("This script will help diagnose why XPU is not being detected.")

    print_section("Environment Information")
    check_environment()

    print_section("Module Import Tests")
    torch_module = safe_import_and_test("torch", test_torch)
    ipex_module = safe_import_and_test("intel_extension_for_pytorch", test_ipex)

    print_section("Device Backend Tests")
    test_device_backend()

    print_section("Manual XPU Tests")
    if torch_module:
        try:
            # Try to manually initialize XPU
            print("Attempting manual XPU initialization...")
            if hasattr(torch_module, "xpu"):
                print(f"torch.xpu module: {torch_module.xpu}")
                print(f"torch.xpu.is_available(): {torch_module.xpu.is_available()}")

                if torch_module.xpu.is_available():
                    print(
                        f"torch.xpu.device_count(): {torch_module.xpu.device_count()}"
                    )

                    # Try creating a tensor
                    device = torch_module.device("xpu:0")
                    tensor = torch_module.tensor([1.0, 2.0, 3.0]).to(device)
                    print(f"XPU tensor: {tensor}")
                    print(f"XPU tensor device: {tensor.device}")
                else:
                    print("❌ torch.xpu.is_available() returned False")
            else:
                print("❌ torch.xpu module not found")

        except Exception as e:
            print(f"❌ Manual XPU test failed: {e}")
            traceback.print_exc()

    print_section("Recommendations")
    print("If XPU is not detected, try:")
    print("1. Ensure Intel Extension for PyTorch is properly installed")
    print("2. Check that Intel XPU drivers are installed")
    print("3. Verify environment variables are set correctly")
    print("4. Try importing intel_extension_for_pytorch before torch")
    print("5. Check if you're running on a system with Intel XPU hardware")

    print("\n🏁 Debug script completed!")


if __name__ == "__main__":
    print("Starting debug script...")
    main()
