#!/bin/bash

# Local script for running router path entropy analysis
# Usage: ./router_path_entropy.sh [model_name] [dataset_name] [context_length] [tokens_per_file] [batch_size]

# Set default values
MODEL_NAME=${1:-"olmoe-i"}
DATASET_NAME=${2:-"lmsys"}
CONTEXT_LENGTH=${3:-2048}
TOKENS_PER_FILE=${4:-2000}
BATCH_SIZE=${5:-10000}

echo "Running router path entropy analysis locally..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Context length: $CONTEXT_LENGTH"
echo "Tokens per file: $TOKENS_PER_FILE"
echo "Batch size: $BATCH_SIZE"

# Run the router path entropy analysis
uv run viz/router_path_entropy.py router-path-entropy --model-name "$MODEL_NAME" --dataset-name "$DATASET_NAME" --context-length "$CONTEXT_LENGTH" --tokens-per-file "$TOKENS_PER_FILE" --batch-size "$BATCH_SIZE"

echo "Router path entropy analysis completed!"
echo "Check the fig/ directory for generated plots:"
echo "  - router_path_frequency.png"
echo "  - router_path_coverage.png"
echo "  - router_path_histogram.png"
