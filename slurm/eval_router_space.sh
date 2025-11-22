#!/bin/bash

# Local script for evaluating raw router space
# This runs with minimal resources for testing/development

uv run python -m exp.eval_router_space eval-router-space \
    --model-name olmoe-i \
    --saebench-batchsize 16 \
    --intruder-batchsize 1 \
    --intruder-n-tokens 1000000 \
    --postprocessor identity \
    --skip-autointerp \
    --log-level TRACE