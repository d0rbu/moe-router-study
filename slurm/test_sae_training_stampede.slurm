#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J test_train_saes         # Set the job name to "test_train_saes"
#SBATCH -N 1                       # Number of nodes to request
#SBATCH -ntasks-per-node 1         # Tasks per node
#SBATCH -t 04:00:00                # Set the wall clock limit to 4 hours
#SBATCH -o test_train_saes-%j      # Send stdout/err to "test_train_saes.[jobID]"
#SBATCH -e test_train_saes-%j.err  # Send stderr to separate file
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
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/sae.py main --log-level TRACE --num-workers 16 --trainers-per-gpu 20 --expansion-factor 16,32,64 --k 160,320,640 --layer 6,7,8,9 --submodule-name mlp_output,layer_output --architecture batchtopk,matryoshka --tokens-per-file 100000 --reshuffled-tokens-per-file 100000

