"""SVD-based circuit discovery."""

import os

import arguably
import matplotlib.pyplot as plt
import torch as th

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_topk
from viz import FIGURE_DIR


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

    # Convert to float for SVD
    activated_experts = activated_experts.float()

    # Get dimensions
    batch_size_total, num_layers, num_experts = activated_experts.shape

    # Use specified batch_size or all samples
    if batch_size > 0 and batch_size < batch_size_total:
        activated_experts = activated_experts[:batch_size]

    # Reshape to 2D for SVD
    activated_experts_flat = activated_experts.reshape(activated_experts.shape[0], -1)

    # Perform SVD
    u, s, vh = th.linalg.svd(activated_experts_flat, full_matrices=False)

    # Extract top circuits
    circuits = vh[:num_circuits]

    # Plot singular values
    plt.figure(figsize=(10, 6))
    plt.plot(s.cpu().numpy())
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.grid(True)

    # Create figure directory if it doesn't exist
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Save figure
    plt.savefig(os.path.join(FIGURE_DIR, "svd_singular_values.png"))
    plt.close()

    # Save circuits
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    th.save(
        {
            "circuits": circuits,
            "singular_values": s,
            "top_k": top_k,
        },
        os.path.join(output_dir, "svd_circuits.pt"),
    )

    print(
        f"Saved {num_circuits} circuits to {os.path.join(output_dir, 'svd_circuits.pt')}"
    )
