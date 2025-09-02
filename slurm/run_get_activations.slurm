#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=get_activations      # Set the job name to "get_activations"
#SBATCH --time=24:00:00                 # Set the wall clock limit to 24 hours
#SBATCH --ntasks=1                      # Request 1 task
#SBATCH --mem=192G                      # Request 192GB per node
#SBATCH --output=get_activations.%j     # Send stdout/err to "get_activations.[jobID]"
#SBATCH --gres=gpu:a100:2               # Request 2 "a100" GPUs per node
#SBATCH --partition=gpu                 # Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                # Set billing account to 123456
##SBATCH --mail-type=ALL                 # Send email on all job events
##SBATCH --mail-user=email_address       # Send all emails to email_address

#First Executable Line
cd moe-router-study
uv run exp/get_activations.py olmoe-i lmsys --tokens-per-file 5000
