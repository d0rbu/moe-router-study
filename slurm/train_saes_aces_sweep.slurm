#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train_saes_sweep     # Set the job name to "train_saes_sweep"
#SBATCH --time=24:00:00                 # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=128G                      # Request 128GB per node
#SBATCH --output=train_saes_sweep.%j    # Send stdout/err to "train_saes_sweep.[jobID]"
#SBATCH --error=train_saes_sweep.%j.err # Send stderr to separate file
#SBATCH --gres=gpu:h100:2               # Request 1 H100 GPU per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

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

module load GCCcore/13.3.0 Python/3.12.3 GCC/12.3.0 MPICH/4.2.1 imkl/2024.2.0 iimpi/2024a

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the distributed job
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/sae.py main --trainers-per-gpu 16 --expansion-factor 16,32,64 --k 160,320,640 --layer 6,7,8,9 --submodule-name mlp_output,layer_output --architecture batchtopk,matryoshka --tokens-per-file 100000 --reshuffled-tokens-per-file 100000 --steps 32768 --warmup-steps 128
