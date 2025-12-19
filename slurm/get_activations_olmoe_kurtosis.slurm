#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J get_activations_olmoe   # Set the job name to "get_activations_olmoe"
#SBATCH -N 1                       # Number of nodes to request
#SBATCH -n 1                       # Total number of tasks across all nodes
#SBATCH -t 06:00:00                # Set the wall clock limit to 6 hours
#SBATCH -o get_activations_olmoe-%j      # Send stdout/err to "get_activations_olmoe.[jobID]"
#SBATCH -e get_activations_olmoe-%j.err  # Send stderr to separate file
#SBATCH -p h100                    # Request the GPU partition/queue

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

if [ -z "$SLURM_NTASKS_PER_NODE" ]; then
  if [ -z "$SLURM_NNODES" ] || [ -z "$SLURM_NTASKS" ]; then
    echo "Error: SLURM_NTASKS_PER_NODE is undefined and SLURM_NNODES/SLURM_NTASKS are not set"
    exit 1
  fi
  if [ $((SLURM_NTASKS % SLURM_NNODES)) -ne 0 ]; then
    echo "Error: SLURM_NTASKS ($SLURM_NTASKS) is not divisible by SLURM_NNODES ($SLURM_NNODES)"
    exit 1
  fi
  export SLURM_NTASKS_PER_NODE=$((SLURM_NTASKS / SLURM_NNODES))
fi

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

# Run the distributed job
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/get_activations.py olmoe-i --minibatch-size 10 --gpus-per-worker 1 --tokens-per-file 50000 --layers-to-store 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --activations-to-store router_logits,mlp_output,layer_output --num-tokens 11000000

