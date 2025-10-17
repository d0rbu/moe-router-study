#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=kmeans_xpu           # Set the job name to "kmeans_xpu"
#SBATCH --time=24:00:00                 # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # Number of CPUs per task
#SBATCH --mem=192G                      # Request 192GB per node (adjusted for lower VRAM)
#SBATCH --output=kmeans_xpu.%j          # Send stdout/err to "kmeans_xpu.[jobID]"
#SBATCH --error=kmeans_xpu.%j.err       # Send stderr to separate file
#SBATCH --gres=gpu:pvc:4                # Request 4 Intel PVC XPUs per node
#SBATCH --partition=xpu                 # Request the XPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456               # Set billing account to 123456
##SBATCH --mail-type=ALL                # Send email on all job events
##SBATCH --mail-user=email_address      # Send all emails to email_address

# Enable detailed logging
set -x

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

module purge
module load WebProxy
module load AI-Tools-GPU/20240816
module load intel/2024a
module load GCCcore/13.3.0 Python/3.12.3

# Change to the project directory
cd $SCRATCH/moe-router-study

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run exp/kmeans.py main --device-type xpu --expansion-factor 1,2,4,8,16,32,64 --tokens-per-file 100000 --reshuffled-tokens-per-file 100000 --minibatch-size 10000 --centroid-minibatch-size 16384 --batch-size 40000 --validate-every 4000

