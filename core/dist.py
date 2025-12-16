"""Distributed training utilities and helpers."""

import torch.distributed as dist


def get_world_size() -> int:
    """Get the world size, returning 1 if distributed is not initialized."""
    return dist.get_world_size() if dist.is_initialized() else 1  # type: ignore[possibly-unbound-attribute]


def get_rank() -> int:
    """Get the current rank, returning 0 if distributed is not initialized."""
    return dist.get_rank() if dist.is_initialized() else 0  # type: ignore[possibly-unbound-attribute]
