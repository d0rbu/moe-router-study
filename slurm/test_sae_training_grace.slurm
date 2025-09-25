#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test_train_saes      # Set the job name to "test_train_saes"
#SBATCH --time=02:00:00                 # Set the wall clock limit to 2 hours
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=64G                       # Request 64GB per node
#SBATCH --output=test_train_saes.%j     # Send stdout/err to "test_train_saes.[jobID]"
#SBATCH --error=test_train_saes.%j.err  # Send stderr to separate file
#SBATCH --gres=gpu:a100:1               # Request 1 a100 per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

# Enable detailed logging
set -x

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Change to the project directory
cd ~/moe-router-study

# Run the distributed job
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/sae.py main --log-level TRACE --num-workers 16
