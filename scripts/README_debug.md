# XPU Debug Scripts for ACES Cluster

This directory contains debug scripts to help diagnose Intel XPU (GPU) availability and configuration issues on the ACES cluster.

## Files

- **`debug_xpu.py`** - Python script to test PyTorch XPU detection and basic operations
- **`check_intel_tools.sh`** - Shell script to check Intel GPU tools, drivers, and environment
- **`debug_xpu.slurm`** - SLURM batch script configured for ACES cluster
- **`README_debug.md`** - This documentation file

## Quick Usage on ACES

### Submit SLURM Job (Recommended)

```bash
# Submit the debug job to SLURM on ACES
sbatch scripts/debug_xpu.slurm

# Check job status
squeue -u $USER

# View results when complete
cat debug_xpu_<job_id>.out
cat debug_xpu_<job_id>.err
```

### Manual Testing (if you have interactive access)

```bash
# Load ACES modules first
module load intel/oneapi/2024.0
module load python/3.11
module load pytorch/2.1.0

# Check Intel GPU environment
chmod +x scripts/check_intel_tools.sh
./scripts/check_intel_tools.sh

# Test PyTorch XPU detection
python scripts/debug_xpu.py
```

## ACES Cluster Configuration

The SLURM script is pre-configured for ACES with:

```bash
# ACES cluster SLURM configuration:
#SBATCH --partition=gpu          # GPU partition on ACES
#SBATCH --gres=gpu:1            # Request 1 GPU resource
#SBATCH --constraint=intel      # Intel GPU nodes

# ACES module loads:
module load intel/oneapi/2024.0
module load python/3.11
module load pytorch/2.1.0
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

## Troubleshooting Common Issues on ACES

### 1. Intel Extension for PyTorch Missing
```bash
# In your virtual environment or user space
pip install intel-extension-for-pytorch
```

### 2. Missing Intel GPU Drivers
- Check if `/dev/dri/renderD*` devices exist on the compute node
- Verify kernel modules: `lsmod | grep -E "(i915|xe)"`
- Contact ACES support if drivers are missing

### 3. Environment Variables
The script automatically sets:
```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu
```

### 4. Module Loading Issues
If modules fail to load, check available versions:
```bash
module avail intel
module avail python
module avail pytorch
```

## Intel GPU Monitoring Tools on ACES

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

Once XPU is working, you can use it with the main kmeans script on ACES:

```bash
# Run kmeans with XPU support
python exp/kmeans.py --device-type xpu [other args...]
```

The debug scripts help ensure your environment is properly configured before running the main workloads on ACES cluster.

