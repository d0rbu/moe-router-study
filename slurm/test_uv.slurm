#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test                 # Set the job name to "test"
#SBATCH --time=00:05:00                 # Set the wall clock limit to 5 minutes
#SBATCH --ntasks=2                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=4               # Number of CPUs per task
#SBATCH --mem=16G                       # Request 16GB per node
#SBATCH --output=test_uv.%j.%t.out      # %j jobid, %t task id
#SBATCH --error=test_uv.%j.%t.err       # %j jobid, %t task id
#SBATCH --gres=gpu:t4:1                 # Request 1 "t4" GPUs per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export WORLD_SIZE=$SLURM_NTASKS

# Change to the project directory
cd ~/moe-router-study
source .venv/bin/activate

# Run the distributed job
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE python scripts/test_slurm.py
