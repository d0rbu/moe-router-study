#!/bin/bash
#SBATCH --job-name=debug_xpu
#SBATCH --output=debug_xpu_%j.out
#SBATCH --error=debug_xpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Uncomment and modify these lines based on your cluster's XPU partition/constraints
# #SBATCH --partition=xpu
# #SBATCH --gres=xpu:1
# #SBATCH --constraint=xpu

echo "🚀 Starting XPU Debug Job"
echo "========================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# Load any necessary modules (uncomment and modify as needed for your cluster)
# module load intel/oneapi
# module load pytorch
# module load intel-extension-for-pytorch

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu

echo "🔧 Environment setup complete"
echo "ZE_ENABLE_PCI_ID_DEVICE_ORDER: $ZE_ENABLE_PCI_ID_DEVICE_ORDER"
echo "SYCL_DEVICE_FILTER: $SYCL_DEVICE_FILTER"
echo ""

# Change to the script directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"
echo ""

# Make the shell script executable
chmod +x check_intel_tools.sh

echo "🔍 Step 1: Checking Intel GPU tools and environment"
echo "=================================================="
./check_intel_tools.sh

echo ""
echo ""
echo "🐍 Step 2: Running Python XPU debug script"
echo "==========================================="

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
else
    echo "No virtual environment found, using system Python"
fi

# Run the Python debug script
python debug_xpu.py

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

