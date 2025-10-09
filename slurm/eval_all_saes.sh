#!/bin/bash

# Local script for evaluating all SAEs
# This runs with minimal resources for testing/development

uv run python -m exp.eval_all_saes --saebench-batchsize 8 --intruder-n-tokens 100000
