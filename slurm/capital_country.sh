#!/bin/bash

# Local script for running capital country experiment
# Usage: ./capital_country.sh [model_name] [alpha_min] [alpha_max] [alpha_steps] [batch_size]

# Set default values
MODEL_NAME=${1:-"olmoe-i"}
ALPHA_MIN=${2:-0.0}
ALPHA_MAX=${3:-5.0}
ALPHA_STEPS=${4:-11}
BATCH_SIZE=${5:-8}
LOG_LEVEL=${6:-"INFO"}

echo "Running capital country experiment locally..."
echo "Model: $MODEL_NAME"
echo "Alpha range: $ALPHA_MIN to $ALPHA_MAX ($ALPHA_STEPS steps)"
echo "Batch size: $BATCH_SIZE"
echo "Log level: $LOG_LEVEL"

# Run the capital country experiment
uv run python -m exp.capital_country \
    --model-name $MODEL_NAME \
    --alpha-min $ALPHA_MIN \
    --alpha-max $ALPHA_MAX \
    --alpha-steps $ALPHA_STEPS \
    --batch-size $BATCH_SIZE \
    --log-level $LOG_LEVEL

echo "Capital country experiment completed!"
echo "Check the out/capital_country/ directory for results"
echo "Check the fig/capital_country/ directory for generated plots"

