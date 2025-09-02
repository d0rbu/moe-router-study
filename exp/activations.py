"""Module for loading activation data from disk."""

from collections.abc import Sequence

from loguru import logger
import torch as th
from torch.utils.data import DataLoader

from exp.activation_dataset import (
    ActivationDataset,
    NoDataFilesError,
    create_activation_dataloader,
    get_expert_indices_from_logits,
)

# Define a module-level ROUTER_LOGITS_DIR so tests can patch exp.activations.ROUTER_LOGITS_DIR
ROUTER_LOGITS_DIR = "router_logits"


def load_activations_indices_tokens_and_topk(
    device: str = "cpu",  # default to CPU to avoid requiring CUDA in tests/CI
) -> tuple[th.Tensor, th.Tensor, list[list[str]], int]:
    """Load boolean activation mask, top-k indices, tokens, and top_k.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - activated_expert_indices: (B, L, topk) long indices of selected experts
      - tokens: list[list[str]] tokenized sequences aligned to batch concatenation
      - top_k: int top-k used during collection
    """
    # Create dataset
    dataset = ActivationDataset(
        device=device,
        activation_keys=["router_logits"],
        preload_metadata=True,
    )

    # Get top_k
    top_k = dataset.get_top_k()

    # Initialize storage for results
    activated_experts_list = []
    activated_expert_indices_list = []
    tokens = []

    # Process each file individually to avoid loading everything at once
    for idx in range(len(dataset)):
        item = dataset[idx]
        router_logits = item["router_logits"]

        # Get expert activations and indices
        expert_activations, expert_indices = get_expert_indices_from_logits(
            router_logits, top_k
        )

        activated_experts_list.append(expert_activations)
        activated_expert_indices_list.append(expert_indices)
        tokens.extend(item["tokens"])

    # Concatenate results if we have any
    if not activated_experts_list:
        raise NoDataFilesError(
            "No data files found; ensure exp.get_router_activations has been run"
        )

    # (B, L, E)
    activated_experts = th.cat(activated_experts_list, dim=0)
    # (B, L, topk)
    activated_expert_indices = th.cat(activated_expert_indices_list, dim=0)

    return activated_experts, activated_expert_indices, tokens, top_k


def load_activations_and_indices_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, th.Tensor, int]:
    """Load boolean activation mask, top-k indices, and top_k.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - activated_expert_indices: (B, L, topk) long indices of selected experts
      - top_k: int top-k used during collection
    """
    activated_experts, activated_expert_indices, _tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(device: str = "cuda") -> tuple[th.Tensor, int]:
    """Load boolean activation mask and top_k.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - top_k: int top-k used during collection
    """
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"
    activated_experts, _indices, top_k = load_activations_and_indices_and_topk(
        device=device
    )
    return activated_experts, top_k


def load_activations(device: str = "cuda") -> th.Tensor:
    """Load boolean activation mask.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
    """
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


def load_activations_tokens_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, list[list[str]], int]:
    """Load boolean activation mask, tokens, and top_k.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - tokens: list[list[str]] tokenized sequences aligned to batch concatenation
      - top_k: int top-k used during collection
    """
    activated_experts, _indices, tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, tokens, top_k


def create_activation_loader(
    batch_size: int = 4,
    device: str = "cpu",
    activation_keys: Sequence[str] = ("router_logits",),
    shuffle: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, int]:
    """Create a DataLoader for activation data.

    This is the recommended way to process large activation datasets
    that don't fit in memory.

    Args:
        batch_size: Number of files to load per batch.
        device: Device to load tensors to.
        activation_keys: Keys of activations to load from files.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of worker processes for loading data.

    Returns:
        Tuple of (DataLoader, top_k).
    """
    return create_activation_dataloader(
        batch_size=batch_size,
        device=device,
        activation_keys=activation_keys,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    load_activations_and_topk()
