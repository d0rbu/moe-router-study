"""Utilities for loading router activations."""

import os
from typing import cast

import torch
from tqdm import tqdm

from exp import get_experiment_dir, get_router_logits_dir


def load_activations_indices_tokens_and_topk(
    experiment_name: str | None = None, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]] | None, int]:
    """
    Load router activations, indices, tokens, and topk from saved files.

    Args:
        experiment_name: Name of the experiment to load activations from.
        device: Device to load tensors to.

    Returns:
        Tuple of (activated_experts, activated_expert_indices, tokens, top_k).
        - activated_experts: Boolean tensor of shape (B, L, E) where B is batch size,
          L is number of layers, and E is number of experts.
        - activated_expert_indices: Integer tensor of shape (B, L, top_k) containing
          the indices of the top-k experts for each token.
        - tokens: List of tokenized sequences, or None if not available.
        - top_k: Number of experts activated per token.

    Raises:
        FileNotFoundError: If the activation directory does not exist.
        ValueError: If no data files are found in the directory.
        KeyError: If required keys are missing from the loaded data.
        ValueError: If router_logits has invalid shape or topk is invalid.
        TypeError: If tokens are not in the expected format (list of lists of strings).
    """
    experiment_dir = get_experiment_dir(name=experiment_name)
    dir_path = get_router_logits_dir(experiment_dir)

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Activation directory not found: {dir_path}")

    # Get all .pt files in the directory
    file_paths = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".pt") and f[:-3].isdigit()
    ]

    if not file_paths:
        raise ValueError("No data files found in directory")

    # Sort files by index
    file_indices = [int(os.path.basename(f)[:-3]) for f in file_paths]
    file_paths = [
        file_paths[i]
        for i in sorted(range(len(file_indices)), key=lambda k: file_indices[k])
    ]
    file_indices.sort()

    # Find the highest contiguous index
    max_contiguous_idx = 0
    for i, idx in enumerate(file_indices):
        if idx != i:
            break
        max_contiguous_idx = i

    # Only process files up to the highest contiguous index
    file_paths = file_paths[: max_contiguous_idx + 1]

    all_activated_experts = []
    all_activated_expert_indices = []
    all_tokens = []
    top_k = None

    for file_path in tqdm(file_paths, desc="Loading router logits"):
        try:
            data = torch.load(file_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load router logits file: {file_path}") from e

        if "topk" not in data:
            raise KeyError(f"Missing 'topk' key in file: {file_path}")

        file_topk = data["topk"]
        if file_topk <= 0:
            raise ValueError(f"Invalid topk value: {file_topk}")

        if "router_logits" not in data:
            raise KeyError(f"Missing 'router_logits' key in file: {file_path}")

        router_logits = data["router_logits"]
        if len(router_logits.shape) != 3:
            raise ValueError(
                f"Expected router_logits to be 3D (batch_size, num_layers, num_experts), "
                f"got shape: {router_logits.shape}"
            )

        num_experts = router_logits.shape[2]
        if file_topk > num_experts:
            raise ValueError(
                f"topk ({file_topk}) cannot be greater than number of experts ({num_experts})"
            )

        if top_k is None:
            top_k = file_topk
        elif top_k != file_topk:
            raise ValueError(f"Inconsistent topk values: {top_k} vs {file_topk}")

        # Get the indices of the top-k experts for each token
        _, indices = torch.topk(router_logits, k=file_topk, dim=2)
        batch_size, num_layers, _ = router_logits.shape

        # Create a boolean tensor indicating which experts are activated
        activated = torch.zeros(
            batch_size, num_layers, num_experts, dtype=torch.bool, device=device
        )
        for b in range(batch_size):
            for layer in range(num_layers):
                activated[b, layer, indices[b, layer]] = True

        all_activated_experts.append(activated)
        all_activated_expert_indices.append(indices)

        # Load tokens if available
        if "tokens" in data:
            file_tokens = data["tokens"]
            # Ensure tokens are in the correct format (list of lists of strings)
            if not isinstance(file_tokens, list):
                raise TypeError(f"Tokens in {file_path} must be a list")

            if not all(isinstance(seq, list) for seq in file_tokens):
                raise TypeError(
                    f"Tokens in {file_path} must be a list of lists of strings"
                )

            all_tokens.extend(file_tokens)

    # Concatenate all tensors along the batch dimension
    activated_experts = torch.cat(all_activated_experts, dim=0)
    activated_expert_indices = torch.cat(all_activated_expert_indices, dim=0)
    tokens = all_tokens if all_tokens else None

    # Ensure top_k is not None
    assert top_k is not None, "top_k should not be None at this point"

    return activated_experts, activated_expert_indices, tokens, cast("int", top_k)


def load_activations_and_indices_and_topk(
    experiment_name: str | None = None, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load router activations, indices, and topk from saved files.

    Args:
        experiment_name: Name of the experiment to load activations from.
        device: Device to load tensors to.

    Returns:
        Tuple of (activated_experts, activated_expert_indices, top_k).
    """
    activated_experts, activated_expert_indices, _, top_k = (
        load_activations_indices_tokens_and_topk(
            experiment_name=experiment_name, device=device
        )
    )
    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(
    experiment_name: str | None = None, device: str = "cpu"
) -> tuple[torch.Tensor, int]:
    """
    Load router activations and topk from saved files.

    Args:
        experiment_name: Name of the experiment to load activations from.
        device: Device to load tensors to.

    Returns:
        Tuple of (activated_experts, top_k).
    """
    activated_experts, _, _, top_k = load_activations_indices_tokens_and_topk(
        experiment_name=experiment_name, device=device
    )
    return activated_experts, top_k


def load_activations(
    experiment_name: str | None = None, device: str = "cpu"
) -> torch.Tensor:
    """
    Load router activations from saved files.

    Args:
        experiment_name: Name of the experiment to load activations from.
        device: Device to load tensors to.

    Returns:
        Boolean tensor of shape (B, L, E) where B is batch size,
        L is number of layers, and E is number of experts.
    """
    activated_experts, _ = load_activations_and_topk(
        experiment_name=experiment_name, device=device
    )
    return activated_experts
