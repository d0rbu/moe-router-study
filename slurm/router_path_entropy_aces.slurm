#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=router_path_entropy      # Set the job name to "router_path_entropy"
#SBATCH --time=06:00:00                     # Set the wall clock limit to 6 hours
#SBATCH --ntasks=1                          # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=64G                           # Request 64GB per node
#SBATCH --output=router_path_entropy.%j     # Send stdout/err to "router_path_entropy.[jobID]"
#SBATCH --error=router_path_entropy.%j.err  # Send stderr to separate file
#SBATCH --gres=gpu:h100:1                   # Request 1 H100 GPU per node
#SBATCH --partition=gpu                     # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                   # Set billing account to 123456
##SBATCH --mail-type=ALL                    # Send email on all job events
##SBATCH --mail-user=email_address          # Send all emails to email_address

# Enable detailed logging
set -x

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

module load GCCcore/13.3.0 Python/3.12.3

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the router path entropy analysis
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run viz/router_path_entropy.py router-path-entropy --model-name "olmoe-i" --dataset-name "lmsys" --context-length 2048 --tokens-per-file 100000 --batch-size 100000
