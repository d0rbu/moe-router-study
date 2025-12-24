#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=capital_country_expert_subset_analysis          # Set the job name
#SBATCH --time=12:00:00                     # Set the wall clock limit to 12 hours (longer for many subsets)
#SBATCH --ntasks=1                          # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --mem=128G                          # Request 128GB per node
#SBATCH --output=capital_country_expert_subset_analysis.%j         # Send stdout/err to "capital_country_expert_subset_analysis.[jobID]"
#SBATCH --error=capital_country_expert_subset_analysis.%j.err      # Send stderr to separate file
#SBATCH --gres=gpu:h100:1                   # Request 1 H100 GPU per node
#SBATCH --partition=gpu                     # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                   # Set billing account to 123456
##SBATCH --mail-type=ALL                    # Send email on all job events
##SBATCH --mail-user=email_address          # Send all emails to email_address

# Enable detailed logging
set -x

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

module load GCCcore/13.3.0 Python/3.12.3 WebProxy

# Change to the project directory
cd $SCRATCH/moe-router-study

# Run the capital country expert subset analysis
# For identity postprocessor results, adjust the importance-file accordingly
# Adjust parameters as needed for your specific country and analysis
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE uv run python -m exp.capital_country_expert_subset_analysis capital-country-expert-subset-analysis \
    --model-name "olmoe-i" \
    --model-dtype "bf16" \
    --importance-file "out/capital_country_expert_importance/south_korea.pt" \
    --target-country "South Korea" \
    --importance-threshold 0.1 \
    --alpha-min 0.0 \
    --alpha-max 2.0 \
    --alpha-steps 9 \
    --intervention-batch-size 8 \
    --output-dir "out/capital_country_expert_subset_analysis" \
    --log-level INFO


