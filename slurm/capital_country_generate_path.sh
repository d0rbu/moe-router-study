#!/bin/bash

# Local script for generating capital country intervention paths
# Usage: ./capital_country_generate_path.sh [target_country] [model_name] [batch_size] [postprocessor]

# Set default values
TARGET_COUNTRY=${1:-"South Korea"}
MODEL_NAME=${2:-"olmoe-i"}
BATCH_SIZE=${3:-128}
POSTPROCESSOR=${4:-"masks"}
EXPERIMENT_TYPE=${5:-"pre_answer"}
LOG_LEVEL=${6:-"INFO"}

# Generate output file path from country name
COUNTRY_SLUG=$(echo "$TARGET_COUNTRY" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
OUTPUT_FILE="out/intervention_paths/${COUNTRY_SLUG}.pt"

echo "Generating capital country intervention path locally..."
echo "Target country: $TARGET_COUNTRY"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Postprocessor: $POSTPROCESSOR"
echo "Experiment type: $EXPERIMENT_TYPE"
echo "Output file: $OUTPUT_FILE"
echo "Log level: $LOG_LEVEL"

# Run the capital country generate path script
uv run python -m exp.capital_country_generate_path \
    --model-name "$MODEL_NAME" \
    --target-country "$TARGET_COUNTRY" \
    --experiment-type "$EXPERIMENT_TYPE" \
    --postprocessor "$POSTPROCESSOR" \
    --batch-size "$BATCH_SIZE" \
    --output-file "$OUTPUT_FILE" \
    --log-level "$LOG_LEVEL"

echo "Intervention path generation completed!"
echo "Output saved to: $OUTPUT_FILE"
echo ""
echo "To use this path in chat, run:"
echo "  uv run python -m exp.capital_country_chat \\"
echo "      --model-name $MODEL_NAME \\"
echo "      --intervention-path $OUTPUT_FILE \\"
echo "      --alpha 1.0"


