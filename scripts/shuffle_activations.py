#!/usr/bin/env python3
"""
Standalone activation shuffling script.

This script shuffles activation files without requiring GPUs or distributed training.
It creates a reshuffled subdirectory and skips recomputation if .pt-temp files already exist.
"""

import os

import arguably
from loguru import logger

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.activations import Activations
from exp.training import get_experiment_name


async def shuffle_activations_standalone(
    model_name: str,
    dataset_name: str,
    tokens_per_file: int,
    context_length: int,
    reshuffled_tokens_per_file: int = 100_000,
    seed: int = 0,
    shuffle_batch_size: int = 100,
    debug: bool = False,
    num_workers: int = 8,
    output_dir: str = OUTPUT_DIR,
) -> list[str]:
    """
    Shuffle activation files using the existing Activations class methods.

    This function adapts the existing Activations.load_files method to work
    without distributed training by creating a minimal single-process environment.
    """
    # Generate experiment name
    experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        context_length=context_length,
    )

    logger.info(f"Experiment name: {experiment_name}")

    # Set up directories
    activation_dir = os.path.join(output_dir, experiment_name, ACTIVATION_DIRNAME)

    logger.info(f"Input directory: {activation_dir}")

    # Check if input directory exists
    if not os.path.exists(activation_dir):
        logger.error(f"Input activation directory does not exist: {activation_dir}")
        return []

    # Create a mock distributed environment for single-process execution
    from unittest.mock import patch

    import torch.distributed as dist

    # Mock distributed functions to work in single-process mode
    def mock_get_rank():
        return 0

    def mock_get_world_size():
        return 1

    def mock_barrier():
        pass

    def mock_all_reduce(tensor, op=None):
        pass

    def mock_broadcast_object_list(object_list, src=0):
        pass

    # Apply patches
    with (
        patch.object(dist, "get_rank", mock_get_rank),
        patch.object(dist, "get_world_size", mock_get_world_size),
        patch.object(dist, "barrier", mock_barrier),
        patch.object(dist, "all_reduce", mock_all_reduce),
        patch.object(dist, "broadcast_object_list", mock_broadcast_object_list),
    ):
        # Use the existing Activations.load_files method
        reshuffled_filepaths = await Activations.load_files(
            activation_dir=activation_dir,
            seed=seed,
            tokens_per_file_in_reshuffled=reshuffled_tokens_per_file,
            shuffle_batch_size=shuffle_batch_size,
            debug=debug,
            num_workers=num_workers,
        )

    return reshuffled_filepaths


@arguably.command
async def main(
    model_name: str,
    dataset_name: str,
    tokens_per_file: int,
    context_length: int,
    *,
    reshuffled_tokens_per_file: int = 100_000,
    seed: int = 0,
    shuffle_batch_size: int = 100,
    debug: bool = False,
    num_workers: int = 8,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Shuffle activation files for MoE router study.

    Args:
        model_name: Name of the model (e.g., 'mixtral-8x7b')
        dataset_name: Name of the dataset (e.g., 'openwebtext')
        tokens_per_file: Number of tokens per original activation file
        context_length: Context length used for activation generation
        reshuffled_tokens_per_file: Number of tokens per reshuffled file
        seed: Random seed for shuffling
        shuffle_batch_size: Batch size for shuffling operations
        debug: Run in debug mode (process fewer files)
        num_workers: Number of worker threads
        output_dir: Output directory
    """
    try:
        reshuffled_filepaths = await shuffle_activations_standalone(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            context_length=context_length,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            seed=seed,
            shuffle_batch_size=shuffle_batch_size,
            debug=debug,
            num_workers=num_workers,
            output_dir=output_dir,
        )

        if reshuffled_filepaths:
            logger.info(
                f"Successfully created {len(reshuffled_filepaths)} reshuffled files"
            )

            # Show the output directory
            experiment_name = get_experiment_name(
                model_name=model_name,
                dataset_name=dataset_name,
                tokens_per_file=tokens_per_file,
                context_length=context_length,
            )
            activation_dir = os.path.join(
                output_dir, experiment_name, ACTIVATION_DIRNAME
            )
            shuffle_dirname = (
                f"reshuffled-seed={seed}-tokens_per_file={reshuffled_tokens_per_file}"
            )
            final_output_dir = os.path.join(activation_dir, shuffle_dirname)

            logger.info(f"Files saved to: {final_output_dir}")
        else:
            logger.error("No files were created")

    except Exception as e:
        logger.error(f"Error during reshuffling: {e}")
        raise


if __name__ == "__main__":
    arguably.run()
