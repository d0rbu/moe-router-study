#!/bin/bash

# Local script for running capital country expert interpretability experiment
# Usage: ./capital_country_expert_interp.sh [target_country] [model_name] [max_experts_to_ablate] [top_k_similar] [log_level]

# Set default values
TARGET_COUNTRY=${1:-"France"}
MODEL_NAME=${2:-"olmoe-i"}
MAX_EXPERTS_TO_ABLATE=${3:-16}
TOP_K_SIMILAR=${4:-10}
LOG_LEVEL=${5:-"INFO"}

# Optional parameters with defaults
DATASET_NAME=${6:-"lmsys"}
MAX_SAMPLES_TO_SEARCH=${7:-100000}
PROCESSING_BATCH_SIZE=${8:-32}
ACTIVATION_BATCH_SIZE=${9:-4096}
SIMILARITY_METHOD=${10:-"jaccard"}

echo "Running capital country expert interpretability experiment locally..."
echo "Target country: $TARGET_COUNTRY"
echo "Model: $MODEL_NAME"
echo "Max experts to ablate: $MAX_EXPERTS_TO_ABLATE"
echo "Top K similar: $TOP_K_SIMILAR"
echo "Dataset: $DATASET_NAME"
echo "Max samples to search: $MAX_SAMPLES_TO_SEARCH"
echo "Processing batch size: $PROCESSING_BATCH_SIZE"
echo "Activation batch size: $ACTIVATION_BATCH_SIZE"
echo "Similarity method: $SIMILARITY_METHOD"
echo "Log level: $LOG_LEVEL"

# Run the capital country expert interpretability experiment
uv run python -m exp.capital_country_expert_interp \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --target-country "$TARGET_COUNTRY" \
    --max-experts-to-ablate "$MAX_EXPERTS_TO_ABLATE" \
    --tokens-per-file 10_000 \
    --top-k-similar "$TOP_K_SIMILAR" \
    --max-samples-to-search "$MAX_SAMPLES_TO_SEARCH" \
    --processing-batch-size "$PROCESSING_BATCH_SIZE" \
    --activation-batch-size "$ACTIVATION_BATCH_SIZE" \
    --similarity-method "$SIMILARITY_METHOD" \
    --log-level "$LOG_LEVEL"

echo "Capital country expert interpretability experiment completed!"
echo "Check the out/capital_country_expert_interp/ directory for results"

