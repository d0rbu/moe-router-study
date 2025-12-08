#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J eval_all_paths         # Set the job name to "eval_all_paths"
#SBATCH -N 1                       # Number of nodes to request
#SBATCH -n 1                       # Total number of tasks across all nodes
#SBATCH -t 24:00:00                # Set the wall clock limit to 24 hours
#SBATCH -tasks-per-node 1          # Tasks per node
#SBATCH -o eval_all_paths-%j      # Send stdout/err to "eval_all_paths.[jobID]"
#SBATCH -e eval_all_paths-%j.err  # Send stderr to separate file
#SBATCH -p h100                    # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

# Enable detailed logging
set -x

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
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
cd $SCRATCH/moe-router-study

if [ -z "$1" ]; then
    echo "Error: experiment_dir argument is required"
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 kmeans_2024-01-01_00-00-00"
    exit 1
fi

# Run the evaluation pipeline
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run python -m exp.eval_all_paths eval-all-paths --experiment-dir "$1" --saebench-batchsize 64 --intruder-batchsize 16 --intruder-n-tokens 10000000 --intruder-explainer-model Qwen/Qwen3-32B-AWQ --log-level DEBUG

