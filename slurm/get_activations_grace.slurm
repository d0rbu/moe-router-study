#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=get_activations      # Set the job name to "get_activations"
#SBATCH --time=24:00:00                  # Set the wall clock limit to 6 hours
#SBATCH --ntasks=2                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=64G                       # Request 64GB per node
#SBATCH --output=get_activations.%j     # Send stdout/err to "get_activations.[jobID]"
#SBATCH --error=get_activations.%j.err  # Send stderr to separate file
#SBATCH --gres=gpu:a100:2               # Request 2 "a100" GPUs per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

# Enable detailed logging
set -x

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Print SLURM environment information for debugging
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM Number of Nodes: $SLURM_NNODES"
echo "SLURM Number of Tasks: $SLURM_NTASKS"
echo "SLURM Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "SLURM Local ID: $SLURM_LOCALID"
echo "SLURM Procedure ID: $SLURM_PROCID"
echo "SLURM Node ID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd ~/moe-router-study

# Run the distributed job
srun --mpi=pmi2 uv run exp/get_activations.py --tokens-per-file 5000
