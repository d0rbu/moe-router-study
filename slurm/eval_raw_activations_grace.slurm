#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=eval_raw_activations    # Set the job name to "eval_raw_activations"
#SBATCH --time=24:00:00                    # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                         # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --cpus-per-task=32                 # Number of CPUs per task
#SBATCH --mem=256G                         # Request 256GB per node
#SBATCH --output=eval_raw_activations.%j   # Send stdout/err to "eval_raw_activations.[jobID]"
#SBATCH --error=eval_raw_activations.%j.err # Send stderr to separate file
#SBATCH --gres=gpu:a100:1                  # Request 1 A100 GPU per node
#SBATCH --partition=gpu                    # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                  # Set billing account to 123456
##SBATCH --mail-type=ALL                   # Send email on all job events
##SBATCH --mail-user=email_address         # Send all emails to email_address

# Enable detailed logging
set -x

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the evaluation pipeline
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run python -m exp.eval_raw_activations eval-raw-activations --model-name olmoe-i --activation-key LAYER_OUTPUT --saebench-batchsize 64 --intruder-n-tokens 10000000 --log-level DEBUG
