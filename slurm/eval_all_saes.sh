#!/bin/bash

# Local script for evaluating all SAEs
# This runs with minimal resources for testing/development

uv run python -m exp.eval_all_saes --model-name olmoe-i --saebench-batchsize 64 --intruder-n-tokens 1000000 --log-level DEBUG
