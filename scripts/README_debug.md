# XPU Debug Scripts

This directory contains debug scripts to help diagnose Intel XPU (GPU) availability and configuration issues.

## Files

- **`debug_xpu.py`** - Python script to test PyTorch XPU detection and basic operations
- **`check_intel_tools.sh`** - Shell script to check Intel GPU tools, drivers, and environment
- **`debug_xpu.slurm`** - SLURM batch script to run both debug scripts on a compute node
- **`README_debug.md`** - This documentation file

## Quick Usage

### Option 1: Submit SLURM Job (Recommended)

```bash
# Submit the debug job to SLURM
sbatch scripts/debug_xpu.slurm

# Check job status
squeue -u $USER

# View results when complete
cat debug_xpu_<job_id>.out
cat debug_xpu_<job_id>.err
```

### Option 2: Run Manually (if you have interactive access)

```bash
# Check Intel GPU environment
chmod +x scripts/check_intel_tools.sh
./scripts/check_intel_tools.sh

# Test PyTorch XPU detection
python scripts/debug_xpu.py
```

## Expected Output

### ✅ Working XPU Setup
```
🔍 PyTorch XPU Debug Information
==================================================
PyTorch version: 2.1.0+xpu
✅ Intel Extension for PyTorch version: 2.1.0+xpu
XPU available: True
XPU device count: 4
  Device 0: Intel(R) Data Center GPU Max 1550
  Device 1: Intel(R) Data Center GPU Max 1550
  ...
✅ Basic XPU tensor operations work
✅ XPU is working correctly!
```

### ❌ Non-Working XPU Setup
```
🔍 PyTorch XPU Debug Information
==================================================
PyTorch version: 2.1.0
❌ Intel Extension for PyTorch not found: No module named 'intel_extension_for_pytorch'
XPU available: False
❌ XPU not available
🔧 Troubleshooting:
1. ❌ Intel Extension for PyTorch is not installed
   Install with: pip install intel-extension-for-pytorch
...
```

## Troubleshooting Common Issues

### 1. Intel Extension for PyTorch Missing
```bash
pip install intel-extension-for-pytorch
```

### 2. Missing Intel GPU Drivers
- Install Intel GPU drivers and Level Zero runtime
- Check if `/dev/dri/renderD*` devices exist
- Verify kernel modules: `lsmod | grep -E "(i915|xe)"`

### 3. Environment Variables
```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu
```

### 4. SLURM Configuration
You may need to modify `debug_xpu.slurm` for your cluster:

```bash
# Uncomment and modify these lines in debug_xpu.slurm:
#SBATCH --partition=xpu          # Your XPU partition name
#SBATCH --gres=xpu:1            # Request XPU resources
#SBATCH --constraint=xpu        # XPU node constraint

# Add module loads if needed:
# module load intel/oneapi
# module load pytorch
```

## Intel GPU Monitoring Tools

Once XPU is working, you can use these tools to monitor Intel GPUs:

- **`xpu-smi`** - Intel's equivalent to nvidia-smi
- **`intel_gpu_top`** - Like nvtop/nvitop for Intel GPUs  
- **`ze_info`** - Level Zero device information

```bash
# Check GPU status
xpu-smi discovery
xpu-smi dump -d 0

# Monitor GPU usage
intel_gpu_top

# Device information
ze_info
```

## Integration with Main Scripts

Once XPU is working, you can use it with the main kmeans script:

```bash
# Run kmeans with XPU support
python exp/kmeans.py --device-type xpu [other args...]
```

The debug scripts help ensure your environment is properly configured before running the main workloads.

