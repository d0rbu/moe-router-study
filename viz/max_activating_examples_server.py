import arguably
import streamlit as st

from exp.activations import (
    load_activations_indices_tokens_and_topk,
)
from viz.circuit_max_activating_examples import (
    _load_circuits_tensor,
)


def max_activating_examples_server(
    circuits_path: str = "",
    top_n: int = 64,
    *_args,
    device: str = "cuda",
    minibatch_size: int | None = None,
) -> None:
    """Run the max-activating tokens visualization from the command line.

    Args:
        circuits_path: Path to a .pt file containing a dict with key "circuits" or a raw tensor.
        top_n: Number of sequences to display.
        device: Torch device for computation (e.g., "cuda" or "cpu").
        minibatch_size: Size of the minibatch for the computation.
    """
    # Load all data once at the top level
    token_topk_mask, _activated_expert_indices, tokens, top_k = load_activations_indices_tokens_and_topk(device=device)
    circuits = _load_circuits_tensor(circuits_path, device=device, token_topk_mask=token_topk_mask)


if __name__ == "__main__":
    max_activating_examples_server()
