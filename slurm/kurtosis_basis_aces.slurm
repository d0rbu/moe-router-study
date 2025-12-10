#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=kurtosis_basis           # Set the job name to "kurtosis_basis"
#SBATCH --time=24:00:00                     # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                          # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=256G                          # Request 256GB per node
#SBATCH --output=kurtosis_basis.%j          # Send stdout/err to "kurtosis_basis.[jobID]"
#SBATCH --error=kurtosis_basis.%j.err       # Send stderr to separate file
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

# Run the kurtosis basis analysis
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/kurtosis_basis.py kurtosis-basis --model-name olmoe --dataset-name lmsys --context-length 2048 --tokens-per-file 100000 --reshuffled-tokens-per-file 100000 --batch-size 4096 --max-samples 10000000 --device cuda

