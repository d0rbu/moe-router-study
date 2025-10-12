#!/bin/bash

# Local script for evaluating a k-means path experiment
# This runs with minimal resources for testing/development

if [ -z "$1" ]; then
    echo "Error: experiment_dir argument is required"
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 kmeans_2024-01-01_00-00-00"
    exit 1
fi

uv run python -m exp.eval_all_paths --experiment-dir "$1" --saebench-batchsize 16 --intruder-batchsize 4 --intruder-n-tokens 1000000 --log-level DEBUG
