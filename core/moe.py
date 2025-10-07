"""
Mixture of Experts (MoE) utility functions.

This module contains helper functions for working with MoE models,
particularly for router logits and path conversions.
"""

import torch as th


def convert_router_logits_to_paths(router_logits: th.Tensor, top_k: int) -> th.Tensor:
    """
    Convert router logits to binary activation paths.

    Args:
        router_logits: Tensor of shape (B, L, E) containing router logits
        top_k: Number of top experts to select

    Returns:
        Tensor of shape (B, L, E) containing binary activations (0 or 1)
    """
    # Get top-k expert indices
    paths_sparse = th.topk(router_logits, k=top_k, dim=-1).indices
    router_paths = th.zeros_like(router_logits)
    router_paths.scatter_(-1, paths_sparse, 1)

    # Return unflattened format (B, L, E)
    return router_paths
