#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=capital_country_expert_importance          # Set the job name
#SBATCH --time=02:00:00                     # Set the wall clock limit to 2 hours (much faster than diff)
#SBATCH --ntasks=1                          # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=128G                          # Request 128GB per node
#SBATCH --output=capital_country_expert_importance.%j         # Send stdout/err to "capital_country_expert_importance.[jobID]"
#SBATCH --error=capital_country_expert_importance.%j.err      # Send stderr to separate file
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

# Run the capital country expert importance computation
# For identity postprocessor results, adjust the input-dir accordingly
# Adjust input-dir and output-file as needed for your specific country
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run python -m exp.capital_country_expert_importance --input-dir "out/capital_country_expert_diff/south_korea" --output-file "out/capital_country_expert_importance/south_korea.pt" --log-level INFO


