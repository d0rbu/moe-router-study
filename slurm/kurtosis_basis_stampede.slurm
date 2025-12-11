#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J kurtosis_basis         # Set the job name to "kurtosis_basis"
#SBATCH -N 1                       # Number of nodes to request
#SBATCH -n 1                       # Total number of tasks across all nodes
#SBATCH -t 24:00:00                # Set the wall clock limit to 24 hours
#SBATCH -o kurtosis_basis-%j       # Send stdout/err to "kurtosis_basis.[jobID]"
#SBATCH -e kurtosis_basis-%j.err   # Send stderr to separate file
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

# Run the kurtosis basis analysis
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/kurtosis_basis.py kurtosis-basis --context-length 2048 --tokens-per-file 100000 --reshuffled-tokens-per-file 100000 --batch-size 4096 --max-samples 10000000 --device cuda

