#!/usr/bin/env python3
"""
Standalone activation shuffling script.

This script shuffles activation files without requiring GPUs or distributed training.
It creates a reshuffled subdirectory and skips recomputation if .pt-temp files already exist.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

import arguably
from loguru import logger
import torch as th
from tqdm import tqdm

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.activations import Activations
from exp.get_activations import ActivationKeys
from exp.training import get_experiment_name


async def _reshuffle_single_process(
    input_files: list[str],
    output_dir: str,
    tokens_per_file_in_reshuffled: int,
    seed: int,
    shuffle_batch_size: int,
    num_workers: int,
) -> list[str]:
    """
    Simplified single-process reshuffling logic adapted from Activations.reshuffle.
    """
    th.manual_seed(seed)

    new_activation_filepaths = []
    current_batch = defaultdict(list)
    current_batch_idx = 0
    num_batch_tokens = 0

    # Load files in batches to avoid memory issues
    for batch_start in tqdm(
        range(0, len(input_files), shuffle_batch_size), desc="Processing file batches"
    ):
        batch_end = min(batch_start + shuffle_batch_size, len(input_files))
        batch_files = input_files[batch_start:batch_end]

        # Load files in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            load_unsafe = partial(th.load, weights_only=False)
            file_data = list(executor.map(load_unsafe, batch_files))

        # Get batch sizes for shuffling
        batch_sizes = [data[ActivationKeys.MLP_OUTPUT].shape[0] for data in file_data]
        total_size = sum(batch_sizes)

        # Create shuffled indices for this batch
        batch_shuffled_indices = th.randperm(total_size)

        # Process each shuffled index
        for batch_idx in tqdm(
            batch_shuffled_indices,
            desc=f"Shuffling batch {batch_start // shuffle_batch_size}",
            leave=False,
        ):
            # Find which file and local index this corresponds to
            file_idx = 0
            local_idx = batch_idx.item()
            cumsum = 0
            for i, size in enumerate(batch_sizes):
                if local_idx < cumsum + size:
                    file_idx = i
                    local_idx = local_idx - cumsum
                    break
                cumsum += size

            data = file_data[file_idx]
            data_to_copy = data.copy()
            del data_to_copy["tokens"]  # Don't copy tokens since we're mixing them up

            for raw_key, value in data_to_copy.items():
                key = str(raw_key)

                if (
                    isinstance(value, th.Tensor | list)
                    and len(value) == batch_sizes[file_idx]
                ):
                    current_batch[key].append(value[local_idx])
                    continue

                if key in current_batch:
                    if current_batch[key] != value:
                        logger.warning(
                            f"Inconsistent value for {key}, keeping first occurrence"
                        )
                else:
                    current_batch[key] = value

            num_batch_tokens += 1

            # Save batch when we reach the target size
            if num_batch_tokens >= tokens_per_file_in_reshuffled:
                output_filepath = os.path.join(
                    output_dir, f"{current_batch_idx}.pt-temp"
                )
                final_filepath = Activations._collate_and_save_batch(
                    current_batch, output_filepath
                )
                new_activation_filepaths.append(final_filepath)

                current_batch = defaultdict(list)
                num_batch_tokens = 0
                current_batch_idx += 1

    # Handle remaining tokens
    if num_batch_tokens > 0:
        output_filepath = os.path.join(output_dir, f"{current_batch_idx}.pt-temp")
        final_filepath = Activations._collate_and_save_batch(
            current_batch, output_filepath
        )
        new_activation_filepaths.append(final_filepath)

    # Rename temp files to final names with shuffled order
    if new_activation_filepaths:
        reshuffled_indices = th.randperm(
            len(new_activation_filepaths), generator=th.Generator().manual_seed(seed)
        )
        final_filepaths = []

        for new_idx, temp_filepath in tqdm(
            zip(reshuffled_indices, new_activation_filepaths, strict=True),
            desc="Finalizing output files",
            total=len(new_activation_filepaths),
        ):
            final_filepath = os.path.join(output_dir, f"{new_idx}.pt")
            os.rename(temp_filepath, final_filepath)
            final_filepaths.append(final_filepath)

        # Sort by filename for consistent ordering
        final_filepaths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
        return final_filepaths

    return []


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
        raise FileNotFoundError(
            f"Input activation directory does not exist: {activation_dir}"
        )

    # Create output directory for reshuffled files
    shuffle_dirname = (
        f"reshuffled-seed={seed}-tokens_per_file={reshuffled_tokens_per_file}"
    )
    output_dir = os.path.join(activation_dir, shuffle_dirname)
    os.makedirs(output_dir, exist_ok=True)

    # Check if reshuffled files already exist
    existing_files = Activations.get_activation_filepaths(output_dir, debug=debug)
    if existing_files:
        logger.info(
            f"Found {len(existing_files)} existing reshuffled files, skipping reshuffling"
        )
        return existing_files

    # Get input activation files
    input_files = Activations.get_activation_filepaths(activation_dir, debug=debug)
    if not input_files:
        raise FileNotFoundError(f"No activation files found in {activation_dir}")

    logger.info(
        f"Reshuffling {len(input_files)} activation files from {activation_dir} to {output_dir}"
    )

    # Perform the reshuffling using simplified single-process logic
    reshuffled_filepaths = await _reshuffle_single_process(
        input_files=input_files,
        output_dir=output_dir,
        tokens_per_file_in_reshuffled=reshuffled_tokens_per_file,
        seed=seed,
        shuffle_batch_size=shuffle_batch_size,
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

    if not reshuffled_filepaths:
        logger.error("No files were created")
        raise RuntimeError("Reshuffling failed - no output files were created")

    logger.success(f"Successfully created {len(reshuffled_filepaths)} reshuffled files")


if __name__ == "__main__":
    arguably.run()
