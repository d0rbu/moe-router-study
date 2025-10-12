#!/bin/bash

# Local script for evaluating all SAEs
# This runs with minimal resources for testing/development

uv run python -m exp.eval_all_saes --saebench-batchsize 16 --intruder-batchsize 4 --intruder-n-tokens 1000000 --log-level DEBUG