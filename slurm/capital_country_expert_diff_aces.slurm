#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=capital_country_expert_diff          # Set the job name
#SBATCH --time=12:00:00                     # Set the wall clock limit to 12 hours
#SBATCH --ntasks=1                          # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=128G                          # Request 128GB per node
#SBATCH --output=capital_country_expert_diff.%j         # Send stdout/err to "capital_country_expert_diff.[jobID]"
#SBATCH --error=capital_country_expert_diff.%j.err      # Send stderr to separate file
#SBATCH --gres=gpu:h100:1                   # Request 1 H100 GPU per node
#SBATCH --partition=gpu                     # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                   # Set billing account to 123456
##SBATCH --mail-type=ALL                    # Send email on all job events
##SBATCH --mail-user=email_address          # Send all emails to email_address

# Enable detailed logging
set -x

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

module load GCCcore/13.3.0 Python/3.12.3 WebProxy

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the capital country expert diff experiment
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run python -m exp.capital_country_expert_diff capital-country-expert-diff --model-name olmoe-i --target-country "South Korea" --router-path-batch-size 5000 --postprocessor masks --log-level INFO


