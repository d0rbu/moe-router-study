#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH -J shuffle_activations         # Set the job name to "shuffle_activations"
#SBATCH -N 1                       # Number of nodes to request
#SBATCH -n 1                       # Total number of tasks across all nodes
#SBATCH -t 12:00:00                # Set the wall clock limit to 12 hours
#SBATCH -o shuffle_activations-%j      # Send stdout/err to "shuffle_activations.[jobID]"
#SBATCH -e shuffle_activations-%j.err  # Send stderr to separate file
#SBATCH -p skx                     # Request the CPU partition/queue (no GPU needed)

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

# Enable detailed logging
set -x

# Print SLURM environment information for debugging
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM Number of Nodes: $SLURM_NNODES"
echo "SLURM Number of Tasks: $SLURM_NTASKS"
echo "SLURM Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "SLURM Local ID: $SLURM_LOCALID"
echo "SLURM Procedure ID: $SLURM_PROCID"
echo "SLURM Node ID: $SLURM_NODEID"

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

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the activation shuffling
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run scripts/shuffle_activations.py shuffle-activations --model-name q3-i --num-workers 4 --tokens-per-file 50000 --reshuffled-tokens-per-file 50000

