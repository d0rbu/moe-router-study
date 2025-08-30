"""Utilities for loading activations from router logits."""

import os
from typing import Optional

import torch as th
from loguru import logger
from tqdm import tqdm

from exp import get_experiment_dir, get_router_logits_dir, ROUTER_LOGITS_DIRNAME

# Keep this for backward compatibility with tests
ROUTER_LOGITS_DIR = ROUTER_LOGITS_DIRNAME


def load_activations_indices_tokens_and_topk(
    experiment_name: str,
    device: str = "cpu",  # default to CPU to avoid requiring CUDA in tests/CI
) -> tuple[th.Tensor, th.Tensor, list[list[str]], int]:
    """Load boolean activation mask, top-k indices, tokens, and top_k.

    Args:
        experiment_name: Name of the experiment to load data from.
        device: Device to load tensors to.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - activated_expert_indices: (B, L, topk) long indices of selected experts
      - tokens: list[list[str]] tokenized sequences aligned to batch concatenation
      - top_k: int top-k used during collection
    """
    activated_expert_indices_collection: list[th.Tensor] = []
    activated_experts_collection: list[th.Tensor] = []
    tokens: list[list[str]] = []
    top_k: int | None = None  # handle case of no files

    experiment_dir = get_experiment_dir(name=experiment_name)
    dir_path = get_router_logits_dir(experiment_dir)

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Activation directory not found: {dir_path}")

    # get the highest file index of contiguous *.pt files
    file_indices = [
        int(f.split(".")[0]) for f in os.listdir(dir_path) if f.endswith(".pt")
    ]

    # Check if there are any files
    if not file_indices:
        raise ValueError("No data files found in directory")

    file_indices.sort()
    # get the highest file index that does not have a gap
    highest_file_idx = file_indices[-1]
    for i in range(len(file_indices) - 1):
        if file_indices[i + 1] - file_indices[i] > 1:
            highest_file_idx = file_indices[i]
            break

    # Iterate through files and load data
    for file_idx in tqdm(
        range(highest_file_idx + 1),
        desc="Loading activations",
        total=highest_file_idx + 1,
    ):
        file_path = os.path.join(dir_path, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = th.load(file_path, map_location=device)
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load router logits file: {file_path}") from e

        # Check for required keys
        if "topk" not in data:
            raise KeyError(f"Missing 'topk' key in file {file_path}")
        if "router_logits" not in data:
            raise KeyError(f"Missing 'router_logits' key in file {file_path}")

        # Get topk value
        file_topk = data["topk"]

        # Validate topk
        if file_topk <= 0:
            raise ValueError(f"Invalid topk value: {file_topk} must be > 0")

        # Check router_logits shape
        router_logits = data["router_logits"]
        if len(router_logits.shape) != 3:
            raise ValueError(
                f"Expected 3D tensor for router_logits, got shape {router_logits.shape}"
            )

        # Check if topk is larger than number of experts
        num_experts = router_logits.shape[2]
        if file_topk > num_experts:
            raise ValueError(
                f"topk cannot be larger than number of experts: {file_topk} > {num_experts}"
            )

        # Check for consistent topk values across files
        if top_k is None:
            top_k = file_topk
        elif top_k != file_topk:
            raise ValueError(f"Inconsistent topk values: {top_k} vs {file_topk}")

        # Process the data
        router_logits = router_logits.to(device)

        # Get tokens if available
        if "tokens" in data:
            tokens.extend(data["tokens"])

        # Get top-k indices and create boolean mask
        batch_size, num_layers, num_experts = router_logits.shape
        topk_indices = th.topk(router_logits, k=file_topk, dim=2).indices

        # (B, L, topk) -> (B, L, E)
        activated_experts = th.zeros_like(router_logits, device=device).bool()
        activated_experts.scatter_(2, topk_indices, True)

        activated_expert_indices_collection.append(topk_indices)
        activated_experts_collection.append(activated_experts)

    # Concatenate all tensors
    if not activated_experts_collection:
        raise ValueError("No data loaded from files")

    # (B, L, E)
    activated_experts = th.cat(activated_experts_collection, dim=0)
    # (B, L, topk)
    activated_expert_indices = th.cat(activated_expert_indices_collection, dim=0)

    assert top_k is not None, "top_k should be set by now"
    return activated_experts, activated_expert_indices, tokens, top_k


def load_activations_and_indices_and_topk(
    experiment_name: str, device: str = "cpu"
) -> tuple[th.Tensor, th.Tensor, int]:
    """Load boolean activation mask, top-k indices, and top_k.

    Args:
        experiment_name: Name of the experiment to load data from.
        device: Device to load tensors to.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - activated_expert_indices: (B, L, topk) long indices of selected experts
      - top_k: int top-k used during collection
    """
    activated_experts, activated_expert_indices, _, top_k = (
        load_activations_indices_tokens_and_topk(
            experiment_name=experiment_name, device=device
        )
    )
    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(
    experiment_name: str, device: str = "cpu"
) -> tuple[th.Tensor, int]:
    """Load boolean activation mask and top_k.

    Args:
        experiment_name: Name of the experiment to load data from.
        device: Device to load tensors to.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - top_k: int top-k used during collection
    """
    activated_experts, _, top_k = load_activations_and_indices_and_topk(
        experiment_name=experiment_name, device=device
    )
    return activated_experts, top_k


def load_activations(experiment_name: str, device: str = "cpu") -> th.Tensor:
    """Load boolean activation mask.

    Args:
        experiment_name: Name of the experiment to load data from.
        device: Device to load tensors to.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
    """
    activated_experts, _ = load_activations_and_topk(
        experiment_name=experiment_name, device=device
    )
    return activated_experts
