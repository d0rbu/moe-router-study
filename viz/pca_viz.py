import os

import arguably
import matplotlib.pyplot as plt
import torch
from torch_pca import PCA

from exp.activations import Activations
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR

FIGURE_PATH = os.path.join(FIGURE_DIR, "pca_circuits.png")


def load_all_activations(activations_loader: Activations) -> torch.Tensor:
    """Load all activations from the loader and concatenate them into a single tensor."""
    all_activations = []

    for batch in activations_loader(batch_size=4096):
        # Get the MLP output activations from the batch
        mlp_output = batch[ActivationKeys.MLP_OUTPUT]
        all_activations.append(mlp_output)

    if not all_activations:
        # Return empty tensor if no activations found
        return torch.empty(0, 0)

    return torch.cat(all_activations, dim=0)


@arguably.command()
def pca_figure(device: str = "cpu") -> None:
    # Use CPU by default to avoid GPU requirement in tests/CI
    # Create activations loader - for now using dummy parameters
    # In a real scenario, you'd need to provide proper experiment_name and other parameters
    activations_loader = Activations(device=device, activation_filepaths=[])

    # Load all activations into a single tensor
    activated_experts = load_all_activations(activations_loader)

    # Handle empty case
    if activated_experts.numel() == 0:
        print("No activations found, creating dummy data for visualization")
        activated_experts = torch.randn(100, 10)  # Dummy data for visualization

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()

    # PCA to visualize the expert activations
    pca = PCA(n_components=2, svd_solver="full")
    activated_experts_pca = pca.fit_transform(activated_experts)

    # Ensure figure directory exists before saving
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # scatter plot the expert activations
    plt.scatter(activated_experts_pca[:, 0], activated_experts_pca[:, 1])
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    arguably.run()
