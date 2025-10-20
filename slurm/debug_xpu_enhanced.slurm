#!/bin/bash
#SBATCH --job-name=debug_xpu_enhanced
#SBATCH --output=debug_xpu_enhanced_%j.out
#SBATCH --error=debug_xpu_enhanced_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# ACES cluster specific configuration for Intel PVC GPUs
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1

echo "🚀 Starting Enhanced XPU Debug Job"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# Load ACES cluster modules for Intel PVC GPUs
module purge
module load WebProxy
module load intelpython/2024.1.0_814
module load intel/2023.07

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu

echo "🔧 Environment setup complete"
echo "ZE_ENABLE_PCI_ID_DEVICE_ORDER: $ZE_ENABLE_PCI_ID_DEVICE_ORDER"
echo "SYCL_DEVICE_FILTER: $SYCL_DEVICE_FILTER"
echo ""

# Change to the repository root directory (where the job was submitted from)
cd "$SLURM_SUBMIT_DIR"

echo "📁 Current directory: $(pwd)"
echo ""

# Make the shell script executable
chmod +x scripts/check_intel_tools.sh

echo "🔍 Step 1: Checking Intel GPU tools and environment"
echo "=================================================="
scripts/check_intel_tools.sh

echo ""
echo ""
echo "🐍 Step 2: Running Enhanced Python XPU debug script"
echo "=================================================="

# Activate user virtual environment (.venv is required)
if [ -f ".venv/bin/activate" ]; then
    echo "Activating user virtual environment (.venv)..."
    source .venv/bin/activate
    echo "Python version: $(python --version)"
    echo "Python path: $(which python)"
    
    # Sync dependencies with uv if pyproject.toml exists
    if [ -f "pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
        echo "📦 Syncing dependencies with uv..."
        uv sync --quiet
        echo "✅ Dependencies synced"
    fi
    
    # Check if torch is available in this environment
    if python -c "import torch" 2>/dev/null; then
        echo "✅ PyTorch is available in user environment"
    else
        echo "❌ PyTorch not found in user environment"
        exit 1
    fi
else
    echo "❌ .venv/bin/activate not found - virtual environment is required"
    exit 1
fi

echo "Final Python version: $(python --version)"
echo "Final Python path: $(which python)"

# Set environment variables for oneAPI toolkits
echo "Setting up oneAPI environment..."
source /sw/hprc/sw/oneAPI/2024.2/setvars.sh

# Run the enhanced Python debug script
echo ""
echo "🔍 Running enhanced XPU debug script..."
python scripts/debug_xpu_enhanced.py

echo ""
echo "🔍 Step 3: Attempting to install XPU version of Intel Extension"
echo "=============================================================="

# Check current Intel Extension version
echo "Current Intel Extension version:"
python -c "import intel_extension_for_pytorch as ipex; print(f'Version: {ipex.__version__}')" 2>/dev/null || echo "Not installed"

# Try to install XPU version
echo ""
echo "Attempting to install XPU version..."
echo "Note: This may fail if XPU packages are not available in the current environment"

# First, try to remove current version and install XPU version
if python -c "import intel_extension_for_pytorch" 2>/dev/null; then
    echo "Removing current Intel Extension for PyTorch..."
    uv remove intel-extension-for-pytorch || echo "Failed to remove via uv"
fi

echo "Trying to install XPU version..."
uv add "intel-extension-for-pytorch[xpu]" || echo "❌ Failed to install XPU version via uv"

# Alternative: try uv pip install in the venv
if ! python -c "import intel_extension_for_pytorch as ipex; 'xpu' in ipex.__version__" 2>/dev/null; then
    echo "Trying alternative installation method..."
    uv pip install intel-extension-for-pytorch[xpu] --upgrade || echo "❌ Failed to install XPU version via uv pip"
fi

echo ""
echo "🔍 Step 4: Final XPU availability check"
echo "======================================="

# Check final status
python -c "
try:
    import torch
    import intel_extension_for_pytorch as ipex
    print(f'PyTorch version: {torch.__version__}')
    print(f'Intel Extension version: {ipex.__version__}')
    
    if hasattr(torch, 'xpu'):
        print('✅ PyTorch XPU module available')
        try:
            xpu_available = torch.xpu.is_available()
            print(f'XPU available: {xpu_available}')
            if xpu_available:
                device_count = torch.xpu.device_count()
                print(f'XPU device count: {device_count}')
            else:
                print('❌ XPU not available - check drivers and runtime')
        except Exception as e:
            print(f'❌ Error checking XPU: {e}')
    else:
        print('❌ PyTorch XPU module not found')
        
    if hasattr(ipex, 'xpu'):
        print('✅ IPEX XPU backend available')
        try:
            ipex_xpu_available = ipex.xpu.is_available()
            print(f'IPEX XPU available: {ipex_xpu_available}')
        except Exception as e:
            print(f'❌ Error checking IPEX XPU: {e}')
    else:
        print('❌ IPEX XPU backend not found')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo ""
echo "🏁 Enhanced XPU Debug Job Complete"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID completed at $(date)"

# Print some final system info
echo ""
echo "📊 Final System Info:"
echo "CPU info: $(lscpu | grep 'Model name' | head -1)"
echo "Memory info: $(free -h | grep 'Mem:')"
echo "Disk usage: $(df -h . | tail -1)"

exit 0
