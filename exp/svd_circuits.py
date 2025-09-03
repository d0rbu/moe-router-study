"""Module for SVD-based circuit discovery."""

import os

import arguably
import matplotlib.pyplot as plt
import torch as th

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_topk
from viz import FIGURE_DIR

# Output file path
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "svd_circuits.pt")
FIGURE_PATH = os.path.join(FIGURE_DIR, "svd_circuits.png")


@arguably.command()
def svd_circuits(
    batch_size: int = 0,
    num_circuits: int = 10,
    device: str = "cpu",
) -> None:
    """Discover circuits using SVD.

    Args:
        batch_size: Number of samples to use (0 for all)
        num_circuits: Number of circuits to extract
        device: Device to use for computation
    """
    # Load activations
    activated_experts, top_k = load_activations_and_topk(device=device)

    # Reshape to (batch_size, layers * experts)
    batch_size_actual = activated_experts.shape[0]
    activated_experts.shape[1]
    activated_experts.shape[2]

    if batch_size > 0 and batch_size < batch_size_actual:
        activated_experts = activated_experts[:batch_size]
        batch_size_actual = batch_size

    activated_experts_flat = activated_experts.view(batch_size_actual, -1).float()

    # Perform SVD
    u, s, vh = th.linalg.svd(activated_experts_flat, full_matrices=False)

    # Extract top circuits
    num_circuits = min(num_circuits, vh.shape[0])
    circuits = vh[:num_circuits]

    # Ensure directories exist
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Plot singular values
    plt.figure(figsize=(10, 6))
    plt.plot(s.cpu().numpy())
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # Save circuits
    th.save({"circuits": circuits, "top_k": top_k}, OUTPUT_FILE)

    # Print summary
    print(f"Extracted {num_circuits} circuits from {batch_size_actual} samples")
    print(f"Circuits shape: {circuits.shape}")
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Singular values plot saved to {FIGURE_PATH}")


if __name__ == "__main__":
    arguably.run()
