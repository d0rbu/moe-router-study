"""
Mixture of Experts (MoE) utility functions.

This module contains helper functions for working with MoE models,
particularly for router logits and path conversions.
"""

from collections.abc import Callable
from enum import StrEnum

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


def router_logits_identity(
    router_logits: th.Tensor, _top_k: int | None = None
) -> th.Tensor:
    """
    Identity function for router logits (no-op postprocessor).
    Returns raw logits unchanged.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        _top_k: Unused, kept for API compatibility

    Returns:
        Tensor of shape (*, L, E) containing raw router logits
    """
    return router_logits


def router_logits_softmax(
    router_logits: th.Tensor, _top_k: int | None = None
) -> th.Tensor:
    """
    Apply softmax to router logits across experts dimension.

    Args:
        router_logits: Tensor of shape (*, L, E) containing router logits
        _top_k: Unused, kept for API compatibility

    Returns:
        Tensor of shape (*, L, E) containing softmaxed probabilities
    """
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
    # Apply softmax first
    probs = router_logits_softmax(router_logits)
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
    # Get top-k indices
    topk_indices = th.topk(router_logits, k=top_k, dim=-1).indices
    
    # Create tensor with -inf everywhere, then scatter top-k logits
    masked_logits = th.full_like(router_logits, float('-inf'))
    masked_logits.scatter_(-1, topk_indices, router_logits.gather(-1, topk_indices))
    
    # Apply softmax (automatically normalizes due to -inf masking)
    return F.softmax(masked_logits, dim=-1)


class RouterLogitsPostprocessor(StrEnum):
    """Enum for different router logits postprocessing options."""

    MASKS = "masks"
    IDENTITY = "identity"
    SOFTMAX = "softmax"
    TOP_K_SOFTMAX_UNNORMALIZED = "top_k_softmax_unnormalized"
    TOP_K_SOFTMAX = "top_k_softmax"


# Mapping from enum to postprocessor functions
_POSTPROCESSOR_MAPPING = {
    RouterLogitsPostprocessor.MASKS: convert_router_logits_to_paths,
    RouterLogitsPostprocessor.IDENTITY: router_logits_identity,
    RouterLogitsPostprocessor.SOFTMAX: router_logits_softmax,
    RouterLogitsPostprocessor.TOP_K_SOFTMAX_UNNORMALIZED: router_logits_top_k_softmax_unnormalized,
    RouterLogitsPostprocessor.TOP_K_SOFTMAX: router_logits_top_k_softmax,
}


def get_postprocessor(
    postprocessor: RouterLogitsPostprocessor,
) -> Callable[[th.Tensor, int], th.Tensor]:
    """
    Get the postprocessor function for a given enum value.

    Args:
        postprocessor: The postprocessor enum value

    Returns:
        The corresponding postprocessor function
    """
    return _POSTPROCESSOR_MAPPING[postprocessor]
