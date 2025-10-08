#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=shuffle_activations     # Set the job name to "shuffle_activations"
#SBATCH --time=12:00:00                    # Set the wall clock limit to 12 hours
#SBATCH --ntasks=1                         # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --cpus-per-task=32                 # Number of CPUs per task (increased for faster processing)
#SBATCH --mem=256G                         # Request 256GB per node (increased for large datasets)
#SBATCH --output=shuffle_activations.%j    # Send stdout/err to "shuffle_activations.[jobID]"
#SBATCH --error=shuffle_activations.%j.err # Send stderr to separate file

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                  # Set billing account to 123456
##SBATCH --mail-type=ALL                   # Send email on all job events
##SBATCH --mail-user=email_address         # Send all emails to email_address

set -x

module load GCCcore/13.3.0 Python/3.12.3

cd $SCRATCH/moe-router-study

uv run scripts/shuffle_activations.py \
    "mixtral-8x7b" \
    "openwebtext" \
    100000 \
    2048 \
    --reshuffled-tokens-per-file 100000 \
    --seed 0 \
    --shuffle-batch-size 50 \
    --num-workers 32
