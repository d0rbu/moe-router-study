"""MoE utility functions for router analysis."""

import torch as th


def convert_router_logits_to_paths(router_logits: th.Tensor, top_k: int) -> th.Tensor:
    """Convert router logits to binary activation paths.

    Args:
        router_logits: Router logits tensor of shape (B, L, E) where:
            - B is batch size
            - L is sequence length
            - E is number of experts
        top_k: Number of top experts to activate per token

    Returns:
        Binary tensor of shape (B, L, E) with 1s at top-k expert positions
    """
    # Get top-k expert indices for each token
    paths_sparse = th.topk(router_logits, k=top_k, dim=-1).indices

    # Create binary activation tensor
    router_paths = th.zeros_like(router_logits)
    router_paths.scatter_(-1, paths_sparse, 1)

    return router_paths
