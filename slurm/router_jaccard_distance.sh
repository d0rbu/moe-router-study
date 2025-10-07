#!/bin/bash

# Local script for running router Jaccard distance analysis
# Usage: ./run_router_jaccard_distance_local.sh [experiment_name] [batch_size]

# Set default values
EXPERIMENT_NAME=${1:-"olmoe-i_lmsys_context_length=2048_tokens_per_file=2000"}
BATCH_SIZE=${2:-10000}

echo "Running router Jaccard distance analysis locally..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Batch size: $BATCH_SIZE"

# Run the router Jaccard distance analysis
uv run viz/router_jaccard_distance.py router-jaccard-distance --experiment-name "$EXPERIMENT_NAME" --batch-size "$BATCH_SIZE"

echo "Router Jaccard distance analysis completed!"
echo "Check the fig/ directory for generated plots:"
echo "  - router_jaccard_matrix_absolute.png"
echo "  - router_jaccard_matrix_independent.png"
echo "  - router_jaccard_matrix_relative.png"
echo "  - router_jaccard_bar_sorted.png"
echo "  - router_jaccard_bar_cross_layer.png"

