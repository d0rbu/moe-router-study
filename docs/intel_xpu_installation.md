# Intel XPU Installation Guide

## Quick Start

```bash
# Run the installation script
bash scripts/install_intel_xpu.sh
```

## The Problem with `uv add`

When trying to install Intel XPU packages with `uv add`, you might encounter this error:

```bash
uv add intel-extension-for-pytorch==2.8.10+xpu oneccl_bind_pt==2.8.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# ❌ Error:
# No solution found when resolving dependencies:
# Because datasets was not found in the package registry...
# hint: An index URL could not be queried due to a lack of valid authentication credentials (403 Forbidden).
```

## Why This Happens

1. **`uv add` vs `uv pip install`**: 
   - `uv add` tries to add packages to `pyproject.toml` and resolve ALL dependencies
   - `uv pip install` just installs packages in the current environment

2. **Index URL Issues**:
   - When using `--extra-index-url`, uv tries to find ALL dependencies in the Intel index
   - Intel's index only has PyTorch-related packages, not general packages like `datasets`
   - The 403 Forbidden error suggests authentication issues with Intel's index

3. **Dependency Resolution**:
   - `uv add` tries to resolve dependencies across all supported Python versions
   - Intel XPU packages may not be available for all Python/platform combinations

## The Solution

Use `uv pip install` instead of `uv add` for Intel XPU packages:

```bash
# ✅ This works:
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
uv pip install intel-extension-for-pytorch==2.8.10+xpu oneccl_bind_pt==2.8.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## Why This Works

- **Environment-only installation**: `uv pip install` doesn't modify `pyproject.toml`
- **Specialized packages**: Intel XPU packages are hardware-specific and shouldn't be in general project dependencies
- **Simpler resolution**: No cross-platform dependency resolution issues
- **Direct installation**: Follows Intel's official installation guide exactly

## Hardware Requirements

Intel XPU support requires specific hardware:
- Intel® Arc™ A-Series or B-Series Graphics
- Intel® Data Center GPU Max Series
- Proper Intel GPU drivers installed

## Verification

After installation, verify with:

```python
import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch version: {torch.__version__}")
print(f"Intel Extension version: {ipex.__version__}")
print(f"XPU devices: {torch.xpu.device_count()}")
```

## Alternative: Manual Installation

If the script fails, you can install manually:

```bash
# Install PyTorch XPU version
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu

# Install Intel Extension
pip install intel-extension-for-pytorch==2.8.10+xpu oneccl_bind_pt==2.8.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
