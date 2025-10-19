#!/bin/bash
#SBATCH --job-name=debug_xpu
#SBATCH --output=debug_xpu_%j.out
#SBATCH --error=debug_xpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# ACES cluster specific configuration for Intel PVC GPUs
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1

echo "🚀 Starting XPU Debug Job"
echo "========================="
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

# Change to the repository root directory
cd "$(dirname "$0")/.."

echo "📁 Current directory: $(pwd)"
echo ""

# Make the shell script executable
chmod +x scripts/check_intel_tools.sh

echo "🔍 Step 1: Checking Intel GPU tools and environment"
echo "=================================================="
scripts/check_intel_tools.sh

echo ""
echo ""
echo "🐍 Step 2: Running Python XPU debug script"
echo "==========================================="

# Activate ACES shared Intel AI environment or user virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating user virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating user virtual environment..."
    source venv/bin/activate
else
    echo "Activating ACES shared Intel AI environment..."
    source /sw/hprc/sw/Python/virtualenvs/intelpython/2024.1.0_814/intel-ai-python-env/bin/activate
fi

# Set environment variables for oneAPI toolkits
echo "Setting up oneAPI environment..."
source /sw/hprc/sw/oneAPI/2024.2/setvars.sh

# Run the Python debug script
python scripts/debug_xpu.py

echo ""
echo "🏁 XPU Debug Job Complete"
echo "========================"
echo "Job ID: $SLURM_JOB_ID completed at $(date)"

# Print some final system info
echo ""
echo "📊 Final System Info:"
echo "CPU info: $(lscpu | grep 'Model name' | head -1)"
echo "Memory info: $(free -h | grep 'Mem:')"
echo "Disk usage: $(df -h . | tail -1)"

exit 0
