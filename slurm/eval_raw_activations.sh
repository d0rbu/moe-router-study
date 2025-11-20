#!/bin/bash

# Local script for evaluating raw model activations
# This runs with minimal resources for testing/development

if [ -z "$1" ]; then
    echo "Error: model_name argument is required"
    echo "Usage: $0 <model_name> [activation_key] [layer]"
    echo "Example: $0 olmoe-i layer_output 0"
    exit 1
fi

MODEL_NAME="$1"
ACTIVATION_KEY="${2:-layer_output}"
LAYER="${3:-0}"

uv run python -m exp.eval_raw_activations eval-raw-activations --model-name "$MODEL_NAME" --activation-key "$ACTIVATION_KEY" --layer "$LAYER" --saebench-batchsize 16 --log-level TRACE --skip-autointerp
