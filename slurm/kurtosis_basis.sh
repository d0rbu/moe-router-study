#!/bin/bash

# Local script for running kurtosis basis analysis
# Usage: ./kurtosis_basis.sh [model_name] [dataset_name] [context_length] [tokens_per_file] [batch_size] [max_samples]

# Set default values
MODEL_NAME=${1:-"olmoe-i"}
DATASET_NAME=${2:-"lmsys"}
CONTEXT_LENGTH=${3:-2048}
TOKENS_PER_FILE=${4:-10000}
BATCH_SIZE=${5:-4096}
MAX_SAMPLES=${6:-100000}

echo "Running kurtosis basis analysis locally..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Context length: $CONTEXT_LENGTH"
echo "Tokens per file: $TOKENS_PER_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Max samples: $MAX_SAMPLES"

# Run the kurtosis basis analysis
uv run exp/kurtosis_basis.py kurtosis-basis --model-name $MODEL_NAME --dataset-name $DATASET_NAME --context-length $CONTEXT_LENGTH --tokens-per-file $TOKENS_PER_FILE --reshuffled-tokens-per-file $TOKENS_PER_FILE --batch-size $BATCH_SIZE --max-samples $MAX_SAMPLES --device cuda

echo "Kurtosis basis analysis completed!"
echo "Check the output/kurtosis_basis/ directory for results and visualizations"

