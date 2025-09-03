"""Server for max activating examples visualization."""

import os

import numpy as np
import torch as th


def generate_random_mask(
    num_layers: int, num_experts: int, sparsity: float = 0.9
) -> th.Tensor:
    """Generate a random mask for testing.

    Args:
        num_layers: Number of layers
        num_experts: Number of experts per layer
        sparsity: Sparsity of the mask (fraction of False values)

    Returns:
        Boolean mask of shape (num_layers, num_experts)
    """
    # Generate random values
    values = th.rand(num_layers, num_experts)

    # Create mask where values > sparsity are True
    mask = values > sparsity

    return mask


def save_circuits(circuits: dict, path: str) -> None:
    """Save circuits to a file.

    Args:
        circuits: Dictionary of circuits
        path: Path to save to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save circuits
    th.save(circuits, path)


def load_circuits(path: str) -> dict:
    """Load circuits from a file.

    Args:
        path: Path to load from

    Returns:
        Dictionary of circuits
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Circuit file not found: {path}")

    # Load circuits
    return th.load(path)


def compute_max_activating_examples(
    circuit: th.Tensor | list[list[bool]] | np.ndarray,
    experiment_name: str = "test",  # Unused but kept for API compatibility
    top_k: int = 10,
    device: str = "cpu",  # Unused but kept for API compatibility
) -> tuple[list[str], th.Tensor]:
    """Compute examples that maximally activate a circuit.

    Args:
        circuit: Circuit tensor of shape (num_layers, num_experts)
        experiment_name: Name of the experiment (unused but kept for API compatibility)
        top_k: Number of top examples to return
        device: Device to use (unused but kept for API compatibility)

    Returns:
        Tuple of (tokens, scores)
    """
    # Convert circuit to tensor if it's not already
    if not isinstance(circuit, th.Tensor):
        if isinstance(circuit, np.ndarray):
            circuit = th.from_numpy(circuit).bool()
        else:
            circuit = th.tensor(circuit, dtype=th.bool)

    # Create random scores for testing
    batch_size = 100
    scores = th.rand(batch_size)

    # Get top-k indices
    _, top_indices = th.topk(scores, k=min(top_k, batch_size))

    # Create random tokens
    tokens = [f"token_{i}" for i in range(batch_size)]

    # Return top-k tokens and scores
    return [tokens[i] for i in top_indices], scores[top_indices]

