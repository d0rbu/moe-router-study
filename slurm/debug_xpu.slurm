#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=debug_xpu            # Set the job name to "debug_xpu"
#SBATCH --time=00:30:00                 # Set the wall clock limit to 30 minutes
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=4               # Number of CPUs per task
#SBATCH --mem=16G                       # Request 16GB per node (minimal for debug)
#SBATCH --output=debug_xpu.%j           # Send stdout/err to "debug_xpu.[jobID]"
#SBATCH --error=debug_xpu.%j.err        # Send stderr to separate file
#SBATCH --gres=gpu:pvc:1                # Request 1 Intel PVC XPU per node
#SBATCH --partition=pvc                 # Request the XPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456               # Set billing account to 123456
##SBATCH --mail-type=ALL                # Send email on all job events
##SBATCH --mail-user=email_address      # Send all emails to email_address

# Enable detailed logging
set -x

echo "🔍 Starting Intel XPU Debug Script"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo ""

module purge
module load WebProxy

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu

# Change to the project directory
cd $SCRATCH/moe-router-study

echo "📁 Current directory: $(pwd)"
echo ""

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

echo ""
echo "🚀 Running Intel XPU Debug Script..."
echo "=================================================="

# Run the debug script
python debug_xpu.py

echo ""
echo "=================================================="
echo "🏁 Debug script completed!"
echo "Check the output above for diagnostic information."
echo ""
echo "If XPU is not detected, the debug output will help identify:"
echo "  - Missing packages or dependencies"
echo "  - Environment configuration issues"
echo "  - Hardware availability problems"
echo "  - Backend detection logic issues"
