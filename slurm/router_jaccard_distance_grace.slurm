#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=router_jaccard           # Set the job name to "router_jaccard"
#SBATCH --time=06:00:00                     # Set the wall clock limit to 6 hours
#SBATCH --ntasks=1                          # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=64G                           # Request 64GB per node
#SBATCH --output=router_jaccard.%j          # Send stdout/err to "router_jaccard.[jobID]"
#SBATCH --error=router_jaccard.%j.err       # Send stderr to separate file
#SBATCH --gres=gpu:a100:1                   # Request 1 A100 GPU per node
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

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the router Jaccard distance analysis
# Using max-samples for faster testing - remove or increase for full analysis
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run viz/router_jaccard_distance.py router-jaccard-distance --model-name olmoe-i --dataset-name lmsys --context-length 2048 --tokens-per-file 100000 --batch-size 10000 --max-samples 10000000
