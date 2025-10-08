#!/bin/bash

# Local script for running activation shuffling
# Usage: ./shuffle_activations.sh [model_name] [dataset_name] [tokens_per_file] [context_length] [reshuffled_tokens_per_file] [seed]

# Set default values
MODEL_NAME=${1:-"olmoe-i"}
DATASET_NAME=${2:-"lmsys"}
TOKENS_PER_FILE=${3:-5000}
CONTEXT_LENGTH=${4:-2048}
RESHUFFLED_TOKENS_PER_FILE=${5:-100000}
SEED=${6:-0}

echo "Running activation shuffling locally..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Tokens per file: $TOKENS_PER_FILE"
echo "Context length: $CONTEXT_LENGTH"
echo "Reshuffled tokens per file: $RESHUFFLED_TOKENS_PER_FILE"
echo "Seed: $SEED"

# Run the activation shuffling
uv run scripts/shuffle_activations.py shuffle-activations --model-name $MODEL_NAME --dataset-name $DATASET_NAME --tokens-per-file $TOKENS_PER_FILE --context-length $CONTEXT_LENGTH --reshuffled-tokens-per-file $RESHUFFLED_TOKENS_PER_FILE --seed $SEED
