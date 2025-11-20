"""
Mixture of Experts (MoE) utility functions.

This module contains helper functions for working with MoE models,
particularly for router logits and path conversions.
"""

import torch as th
import torch.nn.functional as F


def convert_router_logits_to_paths(router_logits: th.Tensor, top_k: int) -> th.Tensor:
    """
    Convert router logits to binary activation paths.

    Args:
        router_logits: Tensor of shape (B, L, E) or (*, L, E) containing router logits
        top_k: Number of top experts to select

    Returns:
        Tensor of shape (B, L, E) or (*, L, E) containing binary activations (0 or 1)
    """
    # Get top-k expert indices
    paths_sparse = th.topk(router_logits, k=top_k, dim=-1).indices
    router_paths = th.zeros_like(router_logits)
    router_paths.scatter_(-1, paths_sparse, 1)

    # Return unflattened format (B, L, E) or (*, L, E)
    return router_paths


def router_logits_to_masks(router_logits: th.Tensor, top_k: int) -> th.Tensor:
    """
    Convert router logits to sparse binary masks indicating which experts were chosen.
    This is the default hardcoded operation that was used throughout the codebase.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        top_k: Number of top experts to select

    Returns:
        Tensor of shape (*, L, E) containing binary masks (0 or 1)
    """
    return convert_router_logits_to_paths(router_logits, top_k)


def router_logits_identity(
    router_logits: th.Tensor, top_k: int | None = None
) -> th.Tensor:
    """
    Identity function for router logits (no-op postprocessor).
    Returns raw logits unchanged.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        top_k: Unused, kept for API compatibility

    Returns:
        Tensor of shape (*, L, E) containing raw router logits
    """
    del top_k  # Silence unused argument warning
    return router_logits


def router_logits_softmax(
    router_logits: th.Tensor, top_k: int | None = None
) -> th.Tensor:
    """
    Apply softmax to router logits across experts dimension.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        top_k: Unused, kept for API compatibility

    Returns:
        Tensor of shape (*, L, E) containing softmaxed probabilities
    """
    del top_k  # Silence unused argument warning
    return F.softmax(router_logits, dim=-1)


def router_logits_top_k_softmax_unnormalized(
    router_logits: th.Tensor, top_k: int
) -> th.Tensor:
    """
    Apply softmax to router logits, then keep only top-k values without renormalization.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        top_k: Number of top experts to select

    Returns:
        Tensor of shape (*, L, E) with top-k softmaxed probabilities, others zeroed
    """
    # Apply softmax
    probs = F.softmax(router_logits, dim=-1)

    # Get top-k indices
    topk_values, topk_indices = th.topk(probs, k=top_k, dim=-1)

    # Create output with zeros everywhere except top-k positions
    result = th.zeros_like(probs)
    result.scatter_(-1, topk_indices, topk_values)

    return result


def router_logits_top_k_softmax(router_logits: th.Tensor, top_k: int) -> th.Tensor:
    """
    Apply softmax to router logits, filter to top-k, and renormalize.
    This produces the final expert weights as used in actual MoE routing.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        top_k: Number of top experts to select

    Returns:
        Tensor of shape (*, L, E) with renormalized top-k softmax probabilities
    """
    # Apply softmax
    probs = F.softmax(router_logits, dim=-1)

    # Get top-k indices
    topk_values, topk_indices = th.topk(probs, k=top_k, dim=-1)

    # Create output with zeros everywhere except top-k positions
    result = th.zeros_like(probs)
    result.scatter_(-1, topk_indices, topk_values)

    # Renormalize to sum to 1
    result = result / result.sum(dim=-1, keepdim=True)

    return result
