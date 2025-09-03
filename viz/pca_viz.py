"""PCA visualization of router activations."""

import os

import arguably
import matplotlib.pyplot as plt
import torch as th
from torch_pca import PCA

from viz import FIGURE_DIR

FIGURE_PATH = os.path.join(FIGURE_DIR, "pca_circuits.png")


@arguably.command()
def pca_figure(device: str = "cpu") -> str:  # device is unused but kept for API compatibility
    """Generate a PCA visualization of router activations.

    Args:
        device: Device to use for computation (unused but kept for API compatibility)

    Returns:
        Path to the generated figure
    """
    # This is a stub implementation since we removed the loader functions
    # In a real implementation, we would load activations and perform PCA

    # Create a dummy tensor for testing
    batch_size = 100
    num_layers = 2
    num_experts = 8

    # Create random activations
    activated_experts = th.rand(batch_size, num_layers, num_experts)

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()

    # PCA to visualize the expert activations
    pca = PCA(n_components=2, svd_solver="full")
    activated_experts_pca = pca.fit_transform(activated_experts)

    # Ensure figure directory exists before saving
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # scatter plot the expert activations
    plt.figure(figsize=(10, 8))
    plt.scatter(activated_experts_pca[:, 0], activated_experts_pca[:, 1])
    plt.title("PCA of Router Activations")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    return FIGURE_PATH


if __name__ == "__main__":
    arguably.run()

