#!/bin/bash

# Helper script to run activation shuffling with common parameters
# Usage: ./shuffle_activations.sh <model_name> <dataset_name> <tokens_per_file> <context_length> [additional_args...]

if [ $# -lt 4 ]; then
    echo "Usage: $0 <model_name> <dataset_name> <tokens_per_file> <context_length> [additional_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 mixtral-8x7b openwebtext 100000 2048"
    echo "  $0 mixtral-8x7b openwebtext 100000 2048 --seed 42 --debug"
    echo "  $0 mixtral-8x7b openwebtext 100000 2048 --reshuffled-tokens-per-file 50000"
    echo ""
    echo "Required parameters:"
    echo "  model_name         - Name of the model (e.g., 'mixtral-8x7b')"
    echo "  dataset_name       - Name of the dataset (e.g., 'openwebtext')"
    echo "  tokens_per_file    - Number of tokens per original activation file"
    echo "  context_length     - Context length used for activation generation"
    echo ""
    echo "Optional parameters (passed as additional_args):"
    echo "  --reshuffled-tokens-per-file N  - Tokens per reshuffled file (default: 100000)"
    echo "  --seed N                        - Random seed (default: 0)"
    echo "  --shuffle-batch-size N          - Batch size for shuffling (default: 100)"
    echo "  --num-workers N                 - Number of worker threads (default: 8)"
    echo "  --debug                         - Run in debug mode"
    exit 1
fi

MODEL_NAME="$1"
DATASET_NAME="$2"
TOKENS_PER_FILE="$3"
CONTEXT_LENGTH="$4"
shift 4  # Remove the first 4 arguments, leaving any additional ones

echo "Running activation shuffling with:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Tokens per file: $TOKENS_PER_FILE"
echo "  Context length: $CONTEXT_LENGTH"
echo "  Additional args: $@"
echo ""

uv run scripts/shuffle_activations.py \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --tokens-per-file "$TOKENS_PER_FILE" \
    --context-length "$CONTEXT_LENGTH" \
    "$@"
