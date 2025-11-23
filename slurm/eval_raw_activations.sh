#!/bin/bash

# Local script for evaluating raw model activations
# This runs with minimal resources for testing/development

uv run python -m exp.eval_raw_activations eval-raw-activations \
    --model-name olmoe-i \
    --activation-key layer-output \
    --saebench-batchsize 16 \
    --batchsize 1 \
    --lower-sparse-probing-vram-usage \
    --skip-autointerp \
    --log-level TRACE
