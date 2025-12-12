"""
Mixture of Experts (MoE) utility functions.

This module contains helper functions for working with MoE models,
particularly for router logits and path conversions.
"""

from collections.abc import Callable
from enum import StrEnum

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentroidMetric(StrEnum):
    """Enum for different centroid/path activation metrics."""

    DOT_PRODUCT = "dot_product"
    COSINE_SIMILARITY = "cosine_similarity"
    L1_DISTANCE = "l1_distance"
    L2_DISTANCE = "l2_distance"
    P_DISTANCE = "p_distance"


def centroid_dot_product(
    activations: th.Tensor, centroids: th.Tensor, _p: float
) -> th.Tensor:
    """Simple matrix multiplication: (..., D) @ (D, C) -> (..., C)"""
    return activations @ centroids.T


def centroid_cosine_similarity(
    activations: th.Tensor, centroids: th.Tensor, p: float
) -> th.Tensor:
    """Normalize both activations and centroids, then dot product."""
    activations_norm = F.normalize(activations, p=2, dim=-1)
    centroids_norm = F.normalize(centroids, p=2, dim=-1)
    return centroid_dot_product(activations_norm, centroids_norm, p)


def centroid_p_distance(
    activations: th.Tensor, centroids: th.Tensor, p: float
) -> th.Tensor:
    """General p-norm distance using cdist."""
    original_shape = activations.shape[:-1]
    activations_flat = activations.view(-1, activations.shape[-1])
    distances = th.cdist(activations_flat.unsqueeze(0), centroids.unsqueeze(0), p=p)
    return distances.squeeze(0).view(*original_shape, -1)


def centroid_l1_distance(
    activations: th.Tensor, centroids: th.Tensor, _p: float
) -> th.Tensor:
    """L1 (Manhattan) distance."""
    return centroid_p_distance(activations, centroids, p=1.0)


def centroid_l2_distance(
    activations: th.Tensor, centroids: th.Tensor, _p: float
) -> th.Tensor:
    """L2 (Euclidean) distance."""
    return centroid_p_distance(activations, centroids, p=2.0)


# Mapping from enum to metric functions
CENTROID_METRICS: dict[
    CentroidMetric, Callable[[th.Tensor, th.Tensor, float], th.Tensor]
] = {
    CentroidMetric.DOT_PRODUCT: centroid_dot_product,
    CentroidMetric.COSINE_SIMILARITY: centroid_cosine_similarity,
    CentroidMetric.L1_DISTANCE: centroid_l1_distance,
    CentroidMetric.L2_DISTANCE: centroid_l2_distance,
    CentroidMetric.P_DISTANCE: centroid_p_distance,
}


class CentroidProjection(nn.Module):
    """
    A module that projects activations to centroid space using a specified metric.

    This can be used as a drop-in replacement for nn.Linear when you want to
    compute centroid activations/distances instead of a simple linear projection.
    """

    def __init__(
        self,
        centroids: th.Tensor,
        metric: CentroidMetric = CentroidMetric.DOT_PRODUCT,
        p: float = 2.0,
    ):
        """
        Initialize the CentroidProjection module.

        Args:
            centroids: Tensor of shape (C, D) containing centroid vectors
            metric: The metric to use for computing activations
            p: The p-norm to use for P_DISTANCE metric (default: 2.0)
        """
        super().__init__()
        self.register_buffer("centroids", centroids)
        self.p = p  # type: ignore
        self.metric_fn = CENTROID_METRICS[metric]  # type: ignore

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Compute centroid activations for input tensor.

        Args:
            x: Input tensor of shape (..., D)

        Returns:
            Tensor of shape (..., C) containing centroid activations/distances
        """
        self.centroids = self.centroids.to(x.device)
        return self.metric_fn(x, self.centroids, self.p)


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
    masked_logits = th.full_like(router_logits, float("-inf"))
    masked_logits.scatter_(-1, topk_indices, router_logits.gather(-1, topk_indices))

    # Apply softmax (automatically normalizes due to -inf masking)
    return router_logits_softmax(masked_logits)


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
