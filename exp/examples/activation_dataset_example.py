"""Example script demonstrating how to use the activation dataset."""

import os
import sys

import torch as th
from tqdm import tqdm

# Add parent directory to path to allow importing from exp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from exp.activation_dataset import ActivationDataset, create_activation_dataloader


def example_1_basic_usage():
    """Basic usage of ActivationDataset."""
    print("\n=== Example 1: Basic Usage ===")

    # Create dataset
    try:
        dataset = ActivationDataset(device="cpu")
        print(f"Dataset contains {len(dataset)} files")

        # Get top_k
        top_k = dataset.get_top_k()
        print(f"Top-k used during collection: {top_k}")

        # Load first item
        item = dataset[0]
        print(f"Keys in item: {list(item.keys())}")
        print(f"Router logits shape: {item['router_logits'].shape}")
        print(f"Number of token sequences: {len(item['tokens'])}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run exp.get_router_activations first")


def example_2_dataloader():
    """Using DataLoader with ActivationDataset."""
    print("\n=== Example 2: Using DataLoader ===")

    try:
        # Create dataloader
        dataloader, top_k = create_activation_dataloader(
            batch_size=2,
            device="cpu",
            activation_keys=["router_logits"],
            shuffle=True,
            num_workers=0,
        )

        print(f"DataLoader contains {len(dataloader)} batches")
        print(f"Top-k used during collection: {top_k}")

        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"Keys in batch: {list(batch.keys())}")
            print(f"Router logits shape: {batch['router_logits'].shape}")
            print(f"Number of token sequences: {len(batch['tokens'])}")

            # Only process first batch for example
            if batch_idx == 0:
                break

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run exp.get_router_activations first")


def example_3_processing_large_dataset():
    """Processing a large dataset without loading everything into memory."""
    print("\n=== Example 3: Processing Large Dataset ===")

    try:
        # Create dataset
        dataset = ActivationDataset(
            device="cpu",
            activation_keys=["router_logits"],
        )

        print(f"Processing {len(dataset)} files")

        # Initialize counters
        total_tokens = 0
        total_activations = 0

        # Process each file individually
        for idx in tqdm(range(len(dataset)), desc="Processing files"):
            item = dataset[idx]

            # Count tokens
            total_tokens += sum(len(tokens) for tokens in item["tokens"])

            # Count activations (assuming top-k = 2)
            router_logits = item["router_logits"]
            top_k = item["topk"]

            # Get top-k expert indices
            topk_indices = th.topk(router_logits, k=top_k, dim=2).indices

            # Count total activations
            total_activations += topk_indices.numel()

        print(f"Total tokens processed: {total_tokens}")
        print(f"Total expert activations: {total_activations}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run exp.get_router_activations first")


def example_4_custom_transform():
    """Using a custom transform with ActivationDataset."""
    print("\n=== Example 4: Custom Transform ===")

    try:
        # Define a custom transform
        def extract_top_experts(item):
            """Extract top-k expert indices from router logits."""
            router_logits = item["router_logits"]
            top_k = item["topk"]

            # Get top-k expert indices
            topk_indices = th.topk(router_logits, k=top_k, dim=2).indices

            return {
                "expert_indices": topk_indices,
                "tokens": item["tokens"],
            }

        # Create dataset with transform
        dataset = ActivationDataset(
            device="cpu",
            activation_keys=["router_logits"],
            transform=extract_top_experts,
        )

        # Load first item
        item = dataset[0]
        print(f"Keys in transformed item: {list(item.keys())}")
        print(f"Expert indices shape: {item['expert_indices'].shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run exp.get_router_activations first")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_dataloader()
    example_3_processing_large_dataset()
    example_4_custom_transform()
