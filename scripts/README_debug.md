# XPU Debug Scripts for ACES Cluster

This directory contains debug scripts specifically configured for the ACES cluster at Texas A&M to help diagnose Intel XPU (GPU) availability and configuration issues.

## Files

- **`debug_xpu.py`** - Python script to test PyTorch XPU detection and basic operations
- **`check_intel_tools.sh`** - Shell script to check Intel GPU tools, drivers, and environment
- **`debug_xpu.slurm`** - SLURM batch script configured for ACES cluster Intel PVC GPUs
- **`README_debug.md`** - This documentation file

## Quick Usage on ACES

### Submit SLURM Job (Recommended)

```bash
# Submit the debug job to ACES SLURM
sbatch scripts/debug_xpu.slurm

# Check job status
squeue -u $USER

# View results when complete
cat debug_xpu_<job_id>.out
cat debug_xpu_<job_id>.err
```

### Manual Testing (if you have interactive access)

```bash
# Get interactive session on Intel PVC node
srun --time=01:00:00 --partition=pvc --gres=gpu:pvc:1 --pty bash -i

# Load ACES modules
module purge
module load WebProxy
module load intelpython/2024.1.0_814
module load intel/2023.07

# Activate Intel AI environment
source /sw/hprc/sw/Python/virtualenvs/intelpython/2024.1.0_814/intel-ai-python-env/bin/activate
source /sw/hprc/sw/oneAPI/2024.2/setvars.sh

# Run debug scripts
./scripts/check_intel_tools.sh
python scripts/debug_xpu.py
```

## ACES Cluster Configuration

The SLURM script is pre-configured for ACES with:

- **Partition**: `pvc` (Intel PVC GPU partition)
- **GPU Resource**: `gpu:pvc:1` (1 Intel Data Center GPU Max 1100)
- **Modules**: WebProxy, intelpython/2024.1.0_814, intel/2023.07
- **Environment**: ACES shared Intel AI environment with PyTorch XPU support
- **oneAPI**: Automatically sources oneAPI 2024.2 environment variables

## Expected Output

### ✅ Working XPU Setup on ACES
```
🔍 PyTorch XPU Debug Information
==================================================
PyTorch version: 2.1.0.post3+cxx11.abi
✅ Intel Extension for PyTorch version: 2.1.40+xpu
XPU available: True
XPU device count: 1
  Device 0: Intel(R) Data Center GPU Max 1100
🧪 Testing basic XPU operations...
✅ Basic XPU tensor operations work
   Test tensor shape: torch.Size([10, 10])
   Test tensor device: xpu:0
   Memory allocated: 0.40 MB
   Memory reserved: 2.00 MB
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
...
```

## Troubleshooting on ACES

### 1. Module Loading Issues
Make sure you're loading the correct ACES modules:
```bash
module purge
module load WebProxy
module load intelpython/2024.1.0_814
module load intel/2023.07
```

### 2. Environment Activation
Use the ACES shared Intel AI environment:
```bash
source /sw/hprc/sw/Python/virtualenvs/intelpython/2024.1.0_814/intel-ai-python-env/bin/activate
source /sw/hprc/sw/oneAPI/2024.2/setvars.sh
```

### 3. SLURM Resource Request
Ensure you're requesting Intel PVC GPUs correctly:
```bash
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
```

## Intel GPU Monitoring on ACES

ACES provides these tools to monitor Intel PVC GPUs:

### sysmon (ACES-specific)
```bash
# Basic GPU information and processes
sysmon

# Monitor periodically
watch -n 5 sysmon
```

### xpumcli (Intel XPU Manager)
```bash
# Discover GPUs
xpumcli discovery

# Get device statistics
xpumcli dump

# Monitor health
xpumcli health
```

## Integration with Main Scripts

Once XPU is working on ACES, you can use it with the main kmeans script:

```bash
# Run kmeans with XPU support on ACES
python exp/kmeans.py --device-type xpu [other args...]
```

## ACES Cluster Details

- **Total Intel PVC GPUs**: 120 Intel Data Center GPU Max 1100 GPUs
- **GPU Memory**: ~46GB per GPU
- **Partition**: `pvc`
- **Resource Request**: `gpu:pvc:N` (where N is number of GPUs needed)
- **Shared Environment**: Pre-configured with PyTorch XPU support

The debug scripts help ensure your environment is properly configured before running the main workloads on ACES.

