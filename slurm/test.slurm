#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test                 # Set the job name to "test"
#SBATCH --time=00:10:00                 # Set the wall clock limit to 1 minute
#SBATCH --ntasks=2                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=2               # Number of CPUs per task
#SBATCH --mem=8G                        # Request 8GB per node
#SBATCH --output=test_basic.%j          # Send stdout/err to "get_activations.[jobID]"
#SBATCH --error=test_basic.%j.err       # Send stderr to separate file
#SBATCH --gres=gpu:t4:1                 # Request 1 "t4" GPUs per node
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
srun uv run scripts/test_slurm.py
