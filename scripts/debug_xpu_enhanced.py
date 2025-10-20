#!/usr/bin/env python3
"""Enhanced XPU debugging script to diagnose Intel GPU availability issues."""

import os
from pathlib import Path
import subprocess
import sys


def run_command(cmd: str, capture_output: bool = True) -> tuple[int, str, str]:
    """Run a shell command and return (return_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_environment_variables():
    """Check XPU-related environment variables."""
    print("🔧 Environment Variables:")
    print("=" * 50)

    xpu_vars = [
        "ZE_ENABLE_PCI_ID_DEVICE_ORDER",
        "SYCL_DEVICE_FILTER",
        "ONEAPI_ROOT",
        "INTEL_EXTENSION_FOR_PYTORCH_BACKEND",
        "LEVEL_ZERO_DEBUG",
        "ZE_FLAT_DEVICE_HIERARCHY",
    ]

    for var in xpu_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    print()


def check_system_tools():
    """Check for system-level XPU tools."""
    print("🛠️ System Tools:")
    print("=" * 50)

    tools = [
        ("xpu-smi", "Intel GPU System Management Interface"),
        ("level_zero_loader", "Level Zero Loader"),
        ("ze_info", "Level Zero Device Info"),
        ("clinfo", "OpenCL Info"),
        ("lspci", "PCI Device Listing"),
    ]

    for tool, _description in tools:
        ret_code, stdout, stderr = run_command(f"which {tool}")
        if ret_code == 0:
            print(f"  ✅ {tool}: {stdout.strip()}")
            if tool == "xpu-smi":
                # Try to run xpu-smi discovery
                ret_code, stdout, stderr = run_command("xpu-smi discovery")
                if ret_code == 0:
                    print("    📊 xpu-smi discovery output:")
                    for line in stdout.strip().split("\n")[:10]:  # First 10 lines
                        print(f"      {line}")
                else:
                    print(f"    ❌ xpu-smi discovery failed: {stderr.strip()}")
        else:
            print(f"  ❌ {tool}: not found")
    print()


def check_level_zero():
    """Check Level Zero runtime."""
    print("🔌 Level Zero Runtime:")
    print("=" * 50)

    # Check for Level Zero libraries
    lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libze_loader.so",
        "/usr/local/lib/libze_loader.so",
        "/opt/intel/oneapi/level-zero/latest/lib/libze_loader.so",
    ]

    found_lib = False
    for lib_path in lib_paths:
        if Path(lib_path).exists():
            print(f"  ✅ Level Zero library found: {lib_path}")
            found_lib = True
            break

    if not found_lib:
        print("  ❌ Level Zero library not found in common locations")

    # Try to find Level Zero via ldconfig
    ret_code, stdout, stderr = run_command("ldconfig -p | grep ze_loader")
    if ret_code == 0 and stdout.strip():
        print("  ✅ Level Zero found via ldconfig:")
        for line in stdout.strip().split("\n"):
            print(f"    {line}")
    else:
        print("  ❌ Level Zero not found via ldconfig")
    print()


def check_intel_gpu_drivers():
    """Check Intel GPU drivers."""
    print("🎮 Intel GPU Drivers:")
    print("=" * 50)

    # Check for Intel GPU devices
    ret_code, stdout, stderr = run_command("lspci | grep -i intel | grep -i vga")
    if ret_code == 0 and stdout.strip():
        print("  ✅ Intel GPU devices found:")
        for line in stdout.strip().split("\n"):
            print(f"    {line}")
    else:
        print("  ❌ No Intel GPU devices found via lspci")

    # Check for Intel GPU kernel modules
    modules = ["i915", "xe"]
    for module in modules:
        ret_code, stdout, stderr = run_command(f"lsmod | grep {module}")
        if ret_code == 0 and stdout.strip():
            print(f"  ✅ Kernel module {module} loaded")
        else:
            print(f"  ❌ Kernel module {module} not loaded")
    print()


def check_pytorch_installation():
    """Check PyTorch and Intel Extension installation details."""
    print("🐍 PyTorch Installation Details:")
    print("=" * 50)

    try:
        import torch

        print(f"  ✅ PyTorch version: {torch.__version__}")
        print(f"  📁 PyTorch location: {torch.__file__}")

        # Check PyTorch build info
        if hasattr(torch, "version"):
            if hasattr(torch.version, "cuda"):
                print(f"  🔧 PyTorch CUDA version: {torch.version.cuda}")
            if hasattr(torch.version, "hip"):
                print(f"  🔧 PyTorch HIP version: {torch.version.hip}")

        # Check for XPU support in PyTorch
        if hasattr(torch, "xpu"):
            print("  ✅ PyTorch XPU module found")
            try:
                device_count = torch.xpu.device_count()
                print(f"  📊 XPU device count: {device_count}")
                if device_count > 0:
                    for i in range(device_count):
                        try:
                            device_name = torch.xpu.get_device_name(i)
                            print(f"    Device {i}: {device_name}")
                        except Exception as e:
                            print(f"    Device {i}: Error getting name - {e}")
                else:
                    print("  ❌ No XPU devices detected by PyTorch")
            except Exception as e:
                print(f"  ❌ Error checking XPU devices: {e}")
        else:
            print("  ❌ PyTorch XPU module not found")

    except ImportError as e:
        print(f"  ❌ PyTorch not available: {e}")

    try:
        import intel_extension_for_pytorch as ipex

        print(f"  ✅ Intel Extension for PyTorch version: {ipex.__version__}")
        print(f"  📁 IPEX location: {ipex.__file__}")

        # Check IPEX backend
        if hasattr(ipex, "xpu"):
            print("  ✅ IPEX XPU backend available")
            try:
                is_available = ipex.xpu.is_available()
                print(f"  📊 IPEX XPU available: {is_available}")
                if is_available:
                    device_count = ipex.xpu.device_count()
                    print(f"  📊 IPEX XPU device count: {device_count}")
            except Exception as e:
                print(f"  ❌ Error checking IPEX XPU: {e}")
        else:
            print("  ❌ IPEX XPU backend not found")

    except ImportError as e:
        print(f"  ❌ Intel Extension for PyTorch not available: {e}")
    print()


def check_oneapi_installation():
    """Check oneAPI installation."""
    print("🔧 oneAPI Installation:")
    print("=" * 50)

    oneapi_root = os.environ.get("ONEAPI_ROOT")
    if oneapi_root:
        print(f"  ✅ ONEAPI_ROOT: {oneapi_root}")

        # Check for key oneAPI components
        components = [
            "compiler/latest",
            "mkl/latest",
            "tbb/latest",
            "level-zero/latest",
        ]

        for component in components:
            component_path = Path(oneapi_root) / component
            if component_path.exists():
                print(f"  ✅ {component}: found")
            else:
                print(f"  ❌ {component}: not found")
    else:
        print("  ❌ ONEAPI_ROOT not set")

        # Try to find oneAPI in common locations
        common_paths = [
            "/opt/intel/oneapi",
            "/sw/hprc/sw/oneAPI/2024.2",
            "/sw/hprc/sw/oneAPI/latest",
        ]

        for path in common_paths:
            if Path(path).exists():
                print(f"  💡 Found oneAPI at: {path}")
                break
    print()


def main():
    """Main debugging function."""
    print("🔍 Enhanced XPU Debug Information")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()

    check_environment_variables()
    check_system_tools()
    check_level_zero()
    check_intel_gpu_drivers()
    check_oneapi_installation()
    check_pytorch_installation()

    print("🏁 Enhanced Debug Complete!")
    print("=" * 50)

    # Provide specific recommendations
    print("💡 Recommendations:")
    print("-" * 50)

    # Check if we have the CPU-only version
    try:
        import intel_extension_for_pytorch as ipex

        if "+cpu" in ipex.__version__:
            print("1. ⚠️  You have the CPU-only version of Intel Extension for PyTorch")
            print("   Consider installing the XPU version:")
            print("   uv remove intel-extension-for-pytorch")
            print("   uv add intel-extension-for-pytorch[xpu]")
    except ImportError:
        pass

    # Check for missing xpu-smi
    ret_code, _, _ = run_command("which xpu-smi")
    if ret_code != 0:
        print("2. ❌ xpu-smi not found - this is needed for XPU device management")
        print("   This should be available in the Intel GPU driver package")

    # Check environment variables
    if not os.environ.get("ZE_ENABLE_PCI_ID_DEVICE_ORDER"):
        print("3. ⚠️  Consider setting XPU environment variables:")
        print("   export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1")
        print("   export SYCL_DEVICE_FILTER=level_zero:gpu")


if __name__ == "__main__":
    main()
