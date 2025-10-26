#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=kmeans_xpu           # Set the job name to "kmeans_xpu"
#SBATCH --time=24:00:00                 # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=192G                      # Request 192GB per node (adjusted for lower VRAM)
#SBATCH --output=kmeans_xpu.%j          # Send stdout/err to "kmeans_xpu.[jobID]"
#SBATCH --error=kmeans_xpu.%j.err       # Send stderr to separate file
#SBATCH --gres=gpu:pvc:8                # Request 8 Intel PVC XPUs per node
#SBATCH --partition=pvc                 # Request the XPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456               # Set billing account to 123456
##SBATCH --mail-type=ALL                # Send email on all job events
##SBATCH --mail-user=email_address      # Send all emails to email_address

# Enable detailed logging
set -x

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

module purge
module load WebProxy

mv uv.intel.lock uv.lock
mv pyproject.intel.toml pyproject.toml

# Set up environment variables that might be needed for Intel XPU
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_DEVICE_FILTER=level_zero:gpu

# Change to the project directory
cd $SCRATCH/moe-router-study

echo "📁 Current directory: $(pwd)"
echo ""

# Activate user virtual environment (.venv is required)
if [ -f ".venv/bin/activate" ]; then
    echo "📦 Syncing dependencies with uv..."
    uv sync --quiet
    echo "✅ Dependencies synced"
    echo "Activating user virtual environment (.venv)..."
    source .venv/bin/activate
    echo "Python version: $(python --version)"
    echo "Python path: $(which python)"
    
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

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/kmeans.py main --log-level DEBUG --device-type xpu --expansion-factor 1,2,4,8,16,32,64 --tokens-per-file 100000 --reshuffled-tokens-per-file 100000 --minibatch-size 4000 --centroid-minibatch-size 16384 --batch-size 40000 --validate-every 8000
