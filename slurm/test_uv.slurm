#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test                 # Set the job name to "test"
#SBATCH --time=00:05:00                 # Set the wall clock limit to 5 minutes
#SBATCH --ntasks=2                      # Total number of tasks (processes) across all nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=4               # Number of CPUs per task
#SBATCH --mem=16G                       # Request 16GB per node
#SBATCH --output=test_uv.%j             # Send stdout/err to "get_activations.[jobID]"
#SBATCH --error=test_uv.%j.err          # Send stderr to separate file
#SBATCH --gres=gpu:t4:1                 # Request 1 "t4" GPUs per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

# Enable detailed logging
set -x

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export WORLD_SIZE=$SLURM_NTASKS

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd ~/moe-router-study
source .venv/bin/activate
# inside job script, before srun
echo "node hostname: $(hostname -s)"
echo "node fqdn:    $(hostname -f 2>/dev/null || true)"
echo "ip addr (short):"
ip -4 addr show scope global | sed -n '1,200p'

# quick TCP connect test from each node to master port (requires nc / netcat)
echo "Testing TCP connectivity to $MASTER_ADDR:$MASTER_PORT ..."
if command -v nc >/dev/null 2>&1; then
  nc -vz -w 3 "$MASTER_ADDR" "$MASTER_PORT" || echo "nc connection failed"
elif command -v timeout >/dev/null 2>&1 && command -v bash >/dev/null 2>&1; then
  (echo > /dev/tcp/"$MASTER_ADDR"/"$MASTER_PORT") >/dev/null 2>&1 || echo "tcp redirect to master failed"
else
  echo "nc not installed; can't test TCP connection easily"
fi

# Print which python/venv is active (on each task)
echo "python: $(which python) ; python -V"

# Run the distributed job
srun python scripts/test_slurm.py
