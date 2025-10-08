#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=shuffle_activations        # Set the job name to "shuffle_activations"
#SBATCH --time=12:00:00                       # Set the wall clock limit to 12 hours
#SBATCH --ntasks=1                            # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH --cpus-per-task=16                    # Number of CPUs per task
#SBATCH --mem=128G                            # Request 128GB per node
#SBATCH --output=shuffle_activations.%j       # Send stdout/err to "shuffle_activations.[jobID]"
#SBATCH --error=shuffle_activations.%j.err    # Send stderr to separate file

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                     # Set billing account to 123456
##SBATCH --mail-type=ALL                      # Send email on all job events
##SBATCH --mail-user=email_address            # Send all emails to email_address

# Enable detailed logging
set -x

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the activation shuffling
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run scripts/shuffle_activations.py shuffle-activations --tokens-per-file 100000 --reshuffled-tokens-per-file 100000
