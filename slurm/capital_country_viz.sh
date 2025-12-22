#!/bin/bash

# Local script for running capital country visualization experiment
# Usage: ./capital_country_viz.sh [model_name] [router_path_batch_size] [postprocessor] [log_level]

# Set default values
MODEL_NAME=${1:-"olmoe-i"}
ROUTER_PATH_BATCH_SIZE=${2:-500}
POSTPROCESSOR=${3:-"masks"}
LOG_LEVEL=${4:-"INFO"}

echo "Running capital country visualization experiment locally..."
echo "Model: $MODEL_NAME"
echo "Router path batch size: $ROUTER_PATH_BATCH_SIZE"
echo "Postprocessor: $POSTPROCESSOR"
echo "Log level: $LOG_LEVEL"

# Run the capital country visualization experiment
uv run python -m exp.capital_country_viz capital-country-viz \
    --model-name $MODEL_NAME \
    --router-path-batch-size $ROUTER_PATH_BATCH_SIZE \
    --postprocessor $POSTPROCESSOR \
    --log-level $LOG_LEVEL

echo "Capital country visualization experiment completed!"
echo "Check the out/capital_country_viz/ directory for results"
echo "Check the fig/capital_country_viz/ directory for generated plots"


