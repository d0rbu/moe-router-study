#!/bin/bash

# Local script for evaluating a k-means path experiment
# This runs with minimal resources for testing/development
# 
# Usage: ./slurm/eval_all_paths.sh <experiment_dir>
# Example: ./slurm/eval_all_paths.sh kmeans_2024-01-01_00-00-00

if [ -z "$1" ]; then
    echo "Error: experiment_dir argument is required"
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 kmeans_2024-01-01_00-00-00"
    exit 1
fi

EXPERIMENT_DIR="$1"

uv run python -m exp.eval_all_paths \
    --experiment-dir "$EXPERIMENT_DIR" \
    --saebench-batchsize 64 \
    --intruder-n-tokens 1000000 \
    --log-level DEBUG

