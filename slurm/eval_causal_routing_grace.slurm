#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=eval_causal_routing     # Set the job name to "eval_causal_routing"
#SBATCH --time=24:00:00                    # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                         # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --cpus-per-task=32                 # Number of CPUs per task
#SBATCH --mem=256G                         # Request 256GB per node
#SBATCH --output=eval_causal_routing.%j    # Send stdout/err to "eval_causal_routing.[jobID]"
#SBATCH --error=eval_causal_routing.%j.err # Send stderr to separate file
#SBATCH --gres=gpu:a100:1                  # Request 1 A100 GPU per node
#SBATCH --partition=gpu                    # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                  # Set billing account to 123456
##SBATCH --mail-type=ALL                   # Send email on all job events
##SBATCH --mail-user=email_address         # Send all emails to email_address

# Enable detailed logging
set -x

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Change to the project directory
cd $SCRATCH/moe-router-study

module purge
module load WebProxy

if [ -z "$1" ]; then
    echo "Error: experiment_dir argument is required"
    echo "Usage: $0 <experiment_dir> [num_centroids] [influence]"
    echo "Example: $0 kmeans_2024-01-01_00-00-00 64 0.8"
    exit 1
fi

# Set default values
EXPERIMENT_DIR="$1"
NUM_CENTROIDS="${2:-64}"
INFLUENCE="${3:-0.8}"

# Run the causal routing evaluation
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run python -m exp.eval_causal_routing \
    --experiment-dir "$EXPERIMENT_DIR" \
    --num-centroids "$NUM_CENTROIDS" \
    --influence "$INFLUENCE" \
    --model-name "olmoe-i" \
    --num-causal-samples 10 \
    --max-gen-length 256 \
    --gen-temperature 1.0 \
    --seed-dataset "lmsys" \
    --seed-sample-length 256 \
    --max-seed-samples 1000 \
    --ctxlen 256 \
    --n-tokens 10000000 \
    --batchsize 8 \
    --n-latents 1000 \
    --example-ctx-len 32 \
    --min-examples 200 \
    --num-non-activating 50 \
    --num-examples 50 \
    --n-quantiles 10 \
    --explainer-model "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4" \
    --explainer-model-max-len 5120 \
    --log-level INFO
