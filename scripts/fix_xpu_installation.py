#!/usr/bin/env python3
"""Script to investigate and fix Intel Extension for PyTorch XPU installation."""

import subprocess
import sys
import os

def run_command(cmd: str) -> tuple[int, str, str]:
    """Run a shell command and return (return_code, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_current_installation():
    """Check current Intel Extension installation."""
    print("🔍 Current Installation Status:")
    print("=" * 50)
    
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✅ Intel Extension for PyTorch version: {ipex.__version__}")
        print(f"📁 Location: {ipex.__file__}")
        
        # Check if it's CPU-only or includes XPU
        if "+cpu" in ipex.__version__:
            print("⚠️  This appears to be a CPU-only build")
        elif "+xpu" in ipex.__version__:
            print("✅ This appears to be an XPU build")
        else:
            print("❓ Build type unclear from version string")
            
        # Check for XPU backend
        if hasattr(ipex, 'xpu'):
            print("✅ IPEX XPU backend is available")
        else:
            print("❌ IPEX XPU backend is NOT available")
            
    except ImportError:
        print("❌ Intel Extension for PyTorch not installed")
    
    print()

def investigate_xpu_packages():
    """Investigate available XPU packages."""
    print("🔍 Investigating XPU Package Options:")
    print("=" * 50)
    
    # Check PyPI for Intel Extension variants
    print("Searching PyPI for Intel Extension packages...")
    ret_code, stdout, stderr = run_command("pip search intel-extension-for-pytorch 2>/dev/null || echo 'pip search not available'")
    
    # Alternative: use pip index to check available versions
    print("Checking available versions...")
    ret_code, stdout, stderr = run_command("pip index versions intel-extension-for-pytorch")
    if ret_code == 0:
        print("Available versions:")
        print(stdout)
    else:
        print("Could not retrieve version information")
    
    # Check for Intel's official installation instructions
    print("\n💡 Intel's recommended installation methods:")
    print("1. For XPU support, Intel typically provides separate installation instructions")
    print("2. XPU support may require specific PyTorch builds")
    print("3. Check: https://intel.github.io/intel-extension-for-pytorch/")
    print()

def try_alternative_installations():
    """Try alternative installation methods for XPU support."""
    print("🔧 Trying Alternative Installation Methods:")
    print("=" * 50)
    
    # Method 1: Try Intel's conda channel
    print("Method 1: Checking Intel conda channel...")
    ret_code, stdout, stderr = run_command("conda search -c intel intel-extension-for-pytorch")
    if ret_code == 0:
        print("✅ Found Intel Extension in Intel conda channel:")
        print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
    else:
        print("❌ Intel conda channel not accessible or package not found")
    
    # Method 2: Check for XPU-specific wheel
    print("\nMethod 2: Looking for XPU-specific wheels...")
    
    # Intel often provides XPU wheels with specific naming
    xpu_wheel_patterns = [
        "intel-extension-for-pytorch-xpu",
        "intel-extension-for-pytorch[xpu]",
        "torch-xpu"
    ]
    
    for pattern in xpu_wheel_patterns:
        print(f"Checking for: {pattern}")
        ret_code, stdout, stderr = run_command(f"pip search {pattern} 2>/dev/null || echo 'Not found'")
        if "Not found" not in stdout and stdout.strip():
            print(f"  ✅ Found: {stdout.strip()}")
        else:
            print(f"  ❌ Not found: {pattern}")
    
    print()

def check_pytorch_xpu_support():
    """Check if current PyTorch has XPU support."""
    print("🔍 PyTorch XPU Support Check:")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if hasattr(torch, 'xpu'):
            print("✅ PyTorch has XPU module")
            try:
                # Try to check XPU availability
                is_available = torch.xpu.is_available()
                print(f"XPU available: {is_available}")
                
                if is_available:
                    device_count = torch.xpu.device_count()
                    print(f"XPU device count: {device_count}")
                else:
                    print("❌ XPU not available (likely driver/runtime issue)")
            except Exception as e:
                print(f"❌ Error checking XPU: {e}")
        else:
            print("❌ PyTorch does NOT have XPU module")
            print("💡 You may need a PyTorch build with XPU support")
            
    except ImportError:
        print("❌ PyTorch not available")
    
    print()

def provide_recommendations():
    """Provide specific recommendations based on findings."""
    print("💡 Recommendations:")
    print("=" * 50)
    
    # Check current state
    has_ipex = False
    has_xpu_backend = False
    has_pytorch_xpu = False
    
    try:
        import intel_extension_for_pytorch as ipex
        has_ipex = True
        has_xpu_backend = hasattr(ipex, 'xpu')
    except ImportError:
        pass
    
    try:
        import torch
        has_pytorch_xpu = hasattr(torch, 'xpu')
    except ImportError:
        pass
    
    if not has_ipex:
        print("1. ❌ Install Intel Extension for PyTorch first")
        print("   uv add intel-extension-for-pytorch")
    elif has_ipex and not has_xpu_backend:
        print("1. ⚠️  You have Intel Extension but without XPU backend")
        print("   This might be expected - XPU support may be in PyTorch itself")
    
    if not has_pytorch_xpu:
        print("2. ❌ PyTorch doesn't have XPU module")
        print("   You may need a PyTorch build with XPU support")
        print("   Check Intel's documentation for compatible PyTorch versions")
    
    print("\n🔗 Useful Resources:")
    print("- Intel Extension for PyTorch docs: https://intel.github.io/intel-extension-for-pytorch/")
    print("- Intel GPU software installation: https://dgpu-docs.intel.com/")
    print("- PyTorch XPU documentation: https://pytorch.org/docs/stable/notes/xpu.html")
    
    print("\n🚨 System-level Requirements:")
    print("- Intel GPU drivers must be installed")
    print("- Level Zero runtime must be available")
    print("- oneAPI toolkit should be properly configured")
    print("- Environment variables (ZE_ENABLE_PCI_ID_DEVICE_ORDER, SYCL_DEVICE_FILTER)")

def main():
    """Main function."""
    print("🔧 Intel Extension for PyTorch XPU Investigation")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    check_current_installation()
    check_pytorch_xpu_support()
    investigate_xpu_packages()
    try_alternative_installations()
    provide_recommendations()
    
    print("\n🏁 Investigation Complete!")

if __name__ == "__main__":
    main()
