#!/bin/bash

# Local script for evaluating raw router space
# This runs with minimal resources for testing/development

if [ -z "$1" ]; then
    echo "Error: model_name argument is required"
    echo "Usage: $0 <model_name>"
    echo "Example: $0 olmoe-i"
    exit 1
fi

uv run python -m exp.eval_router_space eval-router-space --model-name "$1" --saebench-batchsize 16 --intruder-batchsize 1 --intruder-n-tokens 1000000 --log-level TRACE --skip-autointerp
