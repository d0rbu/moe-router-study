#!/usr/bin/env python3
"""
CPU-only activation shuffling script.

This script performs activation shuffling on CPU-only nodes without requiring GPUs
or distributed training. It uses the existing Activations.reshuffle method with
cpu_only=True to handle single-process execution.
"""

import os

import arguably
from loguru import logger

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.activations import Activations
from exp.training import get_experiment_name


@arguably.command()
async def shuffle_activations(
    *,
    model_name: str = "olmoe_i",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 5000,
    context_length: int = 2048,
    reshuffled_tokens_per_file: int = 100_000,
    shuffle_batch_size: int = 10,
    seed: int = 0,
    debug: bool = False,
    num_workers: int = 8,
) -> None:
    """Shuffle activations for CPU-only execution.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        tokens_per_file: Number of tokens per file in original activations
        context_length: Context length used for activations
        reshuffled_tokens_per_file: Number of tokens per file in reshuffled output
        shuffle_batch_size: Batch size for shuffling operations
        seed: Random seed for reproducible shuffling
        debug: Enable debug mode (processes fewer files)
        num_workers: Number of worker threads for file I/O
    """
    # Get experiment name and paths
    experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        context_length=context_length,
    )

    activation_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)
    output_dir = os.path.join(
        OUTPUT_DIR,
        experiment_name,
        ACTIVATION_DIRNAME,
        f"reshuffled-seed={seed}-tokens_per_file={reshuffled_tokens_per_file}",
    )

    logger.info(f"Input directory: {activation_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Perform the reshuffling using CPU-only mode
    reshuffled_filepaths = await Activations.reshuffle(
        activation_dir=activation_dir,
        output_dir=output_dir,
        tokens_per_file_in_reshuffled=reshuffled_tokens_per_file,
        shuffle_batch_size=shuffle_batch_size,
        seed=seed,
        debug=debug,
        num_workers=num_workers,
        cpu_only=True,
    )

    logger.success(
        f"Successfully shuffled activations! Created {len(reshuffled_filepaths)} files."
    )


if __name__ == "__main__":
    arguably.run()
