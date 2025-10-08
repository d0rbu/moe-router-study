# Activation Shuffling Scripts

This directory contains scripts for shuffling activation files without requiring GPUs or distributed training.

## Overview

The activation shuffling process takes original activation files and reshuffles them into a new directory structure with a different number of tokens per file. This is useful for preparing data for training or analysis with different batch sizes.

## Files

- `shuffle_activations.py` - Main shuffling script
- `../slurm/shuffle_activations.sh` - Helper script for easy execution
- `../slurm/shuffle_activations_aces.slurm` - SLURM job script for ACES cluster
- `../slurm/shuffle_activations_grace.slurm` - SLURM job script for Grace cluster

## Usage

### Direct Script Usage

```bash
uv run scripts/shuffle_activations.py \
    --model-name "mixtral-8x7b" \
    --dataset-name "openwebtext" \
    --tokens-per-file 100000 \
    --context-length 2048 \
    --reshuffled-tokens-per-file 100000 \
    --seed 0
```

### Using the Helper Script

```bash
# Basic usage
./slurm/shuffle_activations.sh mixtral-8x7b openwebtext 100000 2048

# With additional options
./slurm/shuffle_activations.sh mixtral-8x7b openwebtext 100000 2048 --seed 42 --debug
```

### SLURM Job Submission

For ACES cluster:
```bash
# Edit the script to set your parameters
vim slurm/shuffle_activations_aces.slurm
# Submit the job
sbatch slurm/shuffle_activations_aces.slurm
```

For Grace cluster:
```bash
# Edit the script to set your parameters
vim slurm/shuffle_activations_grace.slurm
# Submit the job
sbatch slurm/shuffle_activations_grace.slurm
```

## Parameters

### Required Parameters

- `--model-name`: Name of the model (e.g., 'mixtral-8x7b')
- `--dataset-name`: Name of the dataset (e.g., 'openwebtext')
- `--tokens-per-file`: Number of tokens per original activation file
- `--context-length`: Context length used for activation generation

### Optional Parameters

- `--reshuffled-tokens-per-file`: Number of tokens per reshuffled file (default: 100000)
- `--seed`: Random seed for shuffling (default: 0)
- `--shuffle-batch-size`: Batch size for shuffling operations (default: 100)
- `--num-workers`: Number of worker threads (default: 8)
- `--debug`: Run in debug mode (process fewer files)
- `--output-dir`: Output directory (default: 'out')

## Directory Structure

The script creates a subdirectory structure like this:

```
out/
└── {experiment_name}/
    └── activations/
        └── reshuffled-seed={seed}-tokens_per_file={reshuffled_tokens_per_file}/
            ├── 0.pt
            ├── 1.pt
            ├── 2.pt
            └── ...
```

Where `{experiment_name}` is generated from the model name, dataset name, tokens per file, and context length.

## Resumability

The script supports resumability through `.pt-temp` files:

1. When processing, the script first creates `.pt-temp` files
2. If the script is interrupted and restarted, it will check for existing `.pt-temp` files
3. If all expected `.pt-temp` files exist, it skips recomputation and proceeds to the final renaming step
4. If some `.pt-temp` files are missing, it will only recompute the missing ones

This allows you to safely restart interrupted jobs without losing progress.

## Resource Requirements

The script is designed to run on CPU-only nodes:

- **CPU**: 16 cores recommended
- **Memory**: 128GB recommended (may need more for large datasets)
- **Time**: 12 hours should be sufficient for most datasets
- **Storage**: Ensure sufficient disk space for both input and output files

## Examples

### Basic Shuffling
```bash
./slurm/shuffle_activations.sh mixtral-8x7b openwebtext 100000 2048
```

### Custom Reshuffled File Size
```bash
./slurm/shuffle_activations.sh mixtral-8x7b openwebtext 100000 2048 --reshuffled-tokens-per-file 50000
```

### Debug Mode (Process Fewer Files)
```bash
./slurm/shuffle_activations.sh mixtral-8x7b openwebtext 100000 2048 --debug
```

### Custom Seed for Different Shuffling
```bash
./slurm/shuffle_activations.sh mixtral-8x7b openwebtext 100000 2048 --seed 42
```

## Troubleshooting

### Common Issues

1. **"Input activation directory does not exist"**
   - Check that you have the correct model name, dataset name, tokens per file, and context length
   - Verify that the activation files were generated successfully

2. **"No activation files found to reshuffle"**
   - Check that the input directory contains `.pt` files
   - Verify file permissions

3. **Out of memory errors**
   - Reduce `--shuffle-batch-size` to process fewer files at once
   - Increase memory allocation in SLURM script
   - Reduce `--num-workers`

4. **Slow performance**
   - Increase `--num-workers` if you have more CPU cores available
   - Increase `--shuffle-batch-size` if you have more memory available

### Monitoring Progress

The script provides detailed logging:
- Progress bars for major operations
- File counts and token counts
- Information about skipped files (when resuming)

Check the SLURM output files (`shuffle_activations.{jobid}` and `shuffle_activations.{jobid}.err`) for detailed logs.
