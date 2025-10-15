#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=kmeans               # Set the job name to "kmeans"
#SBATCH --time=24:00:00                 # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=256G                      # Request 256GB per node
#SBATCH --output=kmeans.%j              # Send stdout/err to "kmeans.[jobID]"
#SBATCH --error=kmeans.%j.err           # Send stderr to separate file
#SBATCH --gres=gpu:a100:2               # Request 2 A100 GPUs per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456               # Set billing account to 123456
##SBATCH --mail-type=ALL                # Send email on all job events
##SBATCH --mail-user=email_address      # Send all emails to email_address

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

# Run the distributed job
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/kmeans.py main --log-level DEBUG --expansion-factor 1,2,4,8,16,32,64 --tokens-per-file 100000 --reshuffled-tokens-per-file 100000 --minibatch-size 10000 --accumulation-size 5
