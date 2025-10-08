#!/usr/bin/env python3
"""
Standalone activation shuffling script.

This script shuffles activation files without requiring GPUs or distributed training.
It creates a reshuffled subdirectory and skips recomputation if .pt-temp files already exist.
"""

import argparse
import asyncio
from collections import defaultdict
from itertools import batched
import os

from loguru import logger
import torch as th
from tqdm import tqdm

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.get_activations import ActivationKeys
from exp.training import get_experiment_name


def get_activation_filepaths(activation_dir: str, debug: bool = False) -> list[str]:
    """Get list of activation filepaths from directory."""
    if not os.path.exists(activation_dir):
        logger.error(f"Activation directory does not exist: {activation_dir}")
        return []

    all_activation_filenames = {
        filename for filename in os.listdir(activation_dir) if filename.endswith(".pt")
    }

    logger.info(f"Found {len(all_activation_filenames)} activation files")

    if not all_activation_filenames:
        return []

    activation_indices = {
        int(filename.split(".")[0]) for filename in all_activation_filenames
    }

    logger.info(f"Found {len(activation_indices)} activation indices")

    if len(activation_indices) == 0:
        return []

    max_contiguous_activation_index = max(activation_indices)
    sorted_indices = sorted(activation_indices)

    # Find the maximum contiguous range
    for i in range(len(sorted_indices) - 1):
        if sorted_indices[i + 1] - sorted_indices[i] > 1:
            max_contiguous_activation_index = sorted_indices[i]
            break

    logger.info(f"Max contiguous activation index: {max_contiguous_activation_index}")

    contiguous_activation_filepaths = [
        os.path.join(activation_dir, f"{i}.pt")
        for i in range(max_contiguous_activation_index + 1)
    ]

    if debug:
        # Limit to first 32 files for debugging
        contiguous_activation_filepaths = contiguous_activation_filepaths[:32]
        logger.info(
            f"Debug mode: limited to {len(contiguous_activation_filepaths)} files"
        )

    return contiguous_activation_filepaths


async def load_files_async(filepaths: list[str]) -> list[dict]:
    """Load multiple files asynchronously."""
    results = await asyncio.gather(
        *[
            asyncio.to_thread(th.load, filepath, weights_only=False)
            for filepath in filepaths
        ]
    )
    return list(results)


def collate_and_save_batch(batch: dict, output_filepath: str) -> str:
    """Collate batch data and save to file."""
    for key, value in batch.items():
        if isinstance(value, list):
            if len(value) == 0:
                del batch[key]
                continue

            if not isinstance(value[0], th.Tensor):
                continue

            batch[key] = th.stack(value, dim=0)

    th.save(batch, output_filepath)
    return output_filepath


async def reshuffle_activations(
    activation_dir: str,
    output_dir: str,
    tokens_per_file_in_reshuffled: int = 100_000,
    seed: int = 0,
    shuffle_batch_size: int = 100,
    debug: bool = False,
    num_workers: int = 8,  # noqa: ARG001
) -> list[str]:
    """
    Reshuffle activation files into new directory structure.

    Args:
        activation_dir: Directory containing original activation files
        output_dir: Directory to save reshuffled files
        tokens_per_file_in_reshuffled: Number of tokens per reshuffled file
        seed: Random seed for shuffling
        shuffle_batch_size: Batch size for shuffling operations
        debug: Whether to run in debug mode (fewer files)
        num_workers: Number of worker threads

    Returns:
        List of reshuffled activation filepaths
    """
    logger.info(f"Starting reshuffle from {activation_dir} to {output_dir}")

    # Get original activation files
    original_activation_filepaths = get_activation_filepaths(
        activation_dir, debug=debug
    )

    if not original_activation_filepaths:
        logger.error("No activation files found to reshuffle")
        return []

    logger.info(f"Found {len(original_activation_filepaths)} files to reshuffle")

    # Check for existing temp files and skip if they exist
    existing_temp_files = [
        os.path.join(output_dir, filename)
        for filename in os.listdir(output_dir)
        if filename.endswith(".pt-temp")
    ]

    if existing_temp_files:
        logger.info(
            f"Found {len(existing_temp_files)} existing .pt-temp files, checking if complete..."
        )

        # Check if we have enough temp files to complete the reshuffling
        expected_files = []
        current_batch_idx = 0

        # Process files to determine expected output count
        for batch_filepaths in tqdm(
            batched(original_activation_filepaths, shuffle_batch_size),
            desc="Calculating expected files",
            total=(len(original_activation_filepaths) + shuffle_batch_size - 1)
            // shuffle_batch_size,
        ):
            batch_data = await load_files_async(batch_filepaths)

            # Calculate tokens in this batch
            total_tokens = 0
            for file_data in batch_data:
                if ActivationKeys.MLP_OUTPUT in file_data:
                    total_tokens += file_data[ActivationKeys.MLP_OUTPUT].shape[0]

            # Calculate number of output files for this batch
            num_output_files = total_tokens // tokens_per_file_in_reshuffled

            expected_files.extend(
                [f"{current_batch_idx + i}.pt-temp" for i in range(num_output_files)]
            )

            current_batch_idx += num_output_files

        # Check if all expected temp files exist
        existing_temp_names = {os.path.basename(f) for f in existing_temp_files}
        expected_temp_names = set(expected_files)

        if expected_temp_names.issubset(existing_temp_names):
            logger.info(
                "All expected .pt-temp files exist, skipping reshuffling and proceeding to rename"
            )

            # Just rename existing temp files to final names
            new_activation_filepaths = [
                temp_file
                for temp_file in existing_temp_files
                if os.path.basename(temp_file) in expected_temp_names
            ]

            # Sort by the numeric index in filename
            new_activation_filepaths.sort(
                key=lambda x: int(os.path.basename(x).split(".")[0])
            )

            # Rename temp files to final files
            reshuffled_indices = th.randperm(
                len(new_activation_filepaths),
                generator=th.Generator().manual_seed(seed),
            )

            final_filepaths = []
            for new_idx, filepath in tqdm(
                zip(reshuffled_indices, new_activation_filepaths, strict=True),
                desc="Renaming temp files",
                total=len(new_activation_filepaths),
            ):
                final_path = os.path.join(output_dir, f"{new_idx}.pt")
                if not os.path.exists(final_path):
                    os.rename(filepath, final_path)
                final_filepaths.append(final_path)

            # Return sorted final filepaths
            final_filepaths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            return final_filepaths
        else:
            logger.info(
                "Some expected .pt-temp files are missing, will recompute missing ones"
            )

    # Proceed with normal reshuffling
    new_activation_filepaths = []
    current_batch_idx = 0

    for batch_filepaths in tqdm(
        batched(original_activation_filepaths, shuffle_batch_size),
        desc="Processing activation batches",
        total=(len(original_activation_filepaths) + shuffle_batch_size - 1)
        // shuffle_batch_size,
    ):
        logger.debug(f"Processing batch with {len(batch_filepaths)} files")

        # Load batch data
        batch_data = await load_files_async(batch_filepaths)

        # Combine all data from this batch
        combined_batch = defaultdict(list)
        for file_data in batch_data:
            for key, value in file_data.items():
                if isinstance(value, th.Tensor):
                    # Split tensor into individual samples
                    for i in range(value.shape[0]):
                        combined_batch[key].append(value[i])
                else:
                    # For non-tensor data, just store the value
                    combined_batch[key] = value

        # Convert to regular dict
        combined_batch = dict(combined_batch)

        if not combined_batch or ActivationKeys.MLP_OUTPUT not in combined_batch:
            logger.warning("Batch has no MLP output data, skipping")
            continue

        total_tokens = len(combined_batch[ActivationKeys.MLP_OUTPUT])
        logger.debug(f"Batch has {total_tokens} total tokens")

        # Split into files of the desired size
        num_output_files = total_tokens // tokens_per_file_in_reshuffled
        tokens_skipped = total_tokens % tokens_per_file_in_reshuffled

        if tokens_skipped > 0:
            logger.info(f"Skipping {tokens_skipped} tokens for even batching")

        for file_idx in tqdm(
            range(num_output_files),
            desc="Creating output files",
            leave=False,
        ):
            start_idx = file_idx * tokens_per_file_in_reshuffled
            end_idx = start_idx + tokens_per_file_in_reshuffled

            # Create batch for this file
            file_batch = {}
            for key, value in combined_batch.items():
                if (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], th.Tensor)
                ):
                    file_batch[key] = value[start_idx:end_idx]
                else:
                    file_batch[key] = value

            # Save to temp file
            output_filepath = os.path.join(
                output_dir,
                f"{current_batch_idx + file_idx}.pt-temp",
            )

            # Skip if temp file already exists
            if os.path.exists(output_filepath):
                logger.debug(f"Temp file {output_filepath} already exists, skipping")
                new_activation_filepaths.append(output_filepath)
                continue

            new_activation_filepaths.append(
                collate_and_save_batch(file_batch, output_filepath)
            )

        current_batch_idx += num_output_files

    if not new_activation_filepaths:
        logger.error("No activation files were created")
        return []

    logger.info(f"Created {len(new_activation_filepaths)} reshuffled files")

    # Shuffle the file indices and rename
    reshuffled_indices = th.randperm(
        len(new_activation_filepaths),
        generator=th.Generator().manual_seed(seed),
    )

    final_filepaths = []
    for new_idx, filepath in tqdm(
        zip(reshuffled_indices, new_activation_filepaths, strict=True),
        desc="Renaming files",
        total=len(new_activation_filepaths),
    ):
        final_path = os.path.join(output_dir, f"{new_idx}.pt")
        if not os.path.exists(final_path):
            os.rename(filepath, final_path)
        final_filepaths.append(final_path)

    # Return sorted final filepaths
    final_filepaths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    return final_filepaths


async def main():
    parser = argparse.ArgumentParser(
        description="Shuffle activation files for MoE router study"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (e.g., 'mixtral-8x7b')",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'openwebtext')",
    )
    parser.add_argument(
        "--tokens-per-file",
        type=int,
        required=True,
        help="Number of tokens per original activation file",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        help="Context length used for activation generation",
    )
    parser.add_argument(
        "--reshuffled-tokens-per-file",
        type=int,
        default=100_000,
        help="Number of tokens per reshuffled file (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0)",
    )
    parser.add_argument(
        "--shuffle-batch-size",
        type=int,
        default=100,
        help="Batch size for shuffling operations (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (process fewer files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Generate experiment name
    experiment_name = get_experiment_name(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        tokens_per_file=args.tokens_per_file,
        context_length=args.context_length,
    )

    logger.info(f"Experiment name: {experiment_name}")

    # Set up directories
    activation_dir = os.path.join(args.output_dir, experiment_name, ACTIVATION_DIRNAME)
    shuffle_dirname = (
        f"reshuffled-seed={args.seed}-tokens_per_file={args.reshuffled_tokens_per_file}"
    )
    output_dir = os.path.join(activation_dir, shuffle_dirname)

    logger.info(f"Input directory: {activation_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if input directory exists
    if not os.path.exists(activation_dir):
        logger.error(f"Input activation directory does not exist: {activation_dir}")
        return 1

    # Perform reshuffling
    try:
        reshuffled_filepaths = await reshuffle_activations(
            activation_dir=activation_dir,
            output_dir=output_dir,
            tokens_per_file_in_reshuffled=args.reshuffled_tokens_per_file,
            seed=args.seed,
            shuffle_batch_size=args.shuffle_batch_size,
            debug=args.debug,
            num_workers=args.num_workers,
        )

        if reshuffled_filepaths:
            logger.info(
                f"Successfully created {len(reshuffled_filepaths)} reshuffled files"
            )
            logger.info(f"Files saved to: {output_dir}")
        else:
            logger.error("No files were created")
            return 1

    except Exception as e:
        logger.error(f"Error during reshuffling: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
