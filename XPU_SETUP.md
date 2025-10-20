# Intel XPU Support Setup

This document explains how to set up Intel XPU support for the moe-router-study project.

## Overview

The project now supports Intel XPU devices (Intel Data Center GPU Max series) in addition to CUDA GPUs. This enables running experiments on Intel's discrete GPUs using the Intel Extension for PyTorch.

## Installation

### Automatic Installation

Run the provided installation script:

```bash
./scripts/install_xpu.sh
```

### Manual Installation

If you prefer to install manually:

1. Remove the existing CPU-only version:
   ```bash
   uv run pip uninstall intel-extension-for-pytorch -y
   ```

2. Install the XPU version:
   ```bash
   uv run pip install intel-extension-for-pytorch==2.8.10+xpu \
       --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
       --force-reinstall
   ```

3. Verify the installation:
   ```bash
   uv run python -c "import intel_extension_for_pytorch as ipex; print(f'IPEX version: {ipex.__version__}')"
   ```

## Hardware Requirements

- Intel Data Center GPU Max series (Ponte Vecchio)
- Intel GPU drivers installed
- Level Zero runtime
- oneAPI toolkit (recommended)

## Environment Variables

For optimal performance, set these environment variables:

```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu
```

## Usage

Once installed, you can use Intel XPU devices by specifying `device_type="xpu"` in the relevant functions:

```python
# Example: Running SAE training on XPU
python -m exp.sae \
    --model-name "gpt2" \
    --dataset-name "openwebtext" \
    --device-type "xpu"
```

## Troubleshooting

### Common Issues

1. **XPU not detected**: Ensure Intel GPU drivers and Level Zero runtime are installed
2. **Import errors**: Make sure you have the XPU version installed (version should end with `+xpu`)
3. **Performance issues**: Set the recommended environment variables

### Verification

To check if XPU is working correctly:

```python
import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch version: {torch.__version__}")
print(f"IPEX version: {ipex.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")
```

### Debug Information

If you encounter issues, run the debug script from your HPC environment to gather system information:

```bash
# This will show detailed information about your Intel GPU setup
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'XPU available: {torch.xpu.is_available()}')
    print(f'XPU device count: {torch.xpu.device_count()}')
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import intel_extension_for_pytorch as ipex
    print(f'IPEX version: {ipex.__version__}')
except Exception as e:
    print(f'IPEX error: {e}')
"
```

## Notes

- The Intel Extension for PyTorch XPU version is not available through standard package managers due to authentication requirements on Intel's package index
- The installation script uses pip directly to bypass these limitations
- XPU functionality requires Intel XPU hardware; on systems without Intel XPU hardware, the library will install but XPU features won't be available
