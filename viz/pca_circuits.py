import os

import arguably
import matplotlib.pyplot as plt
from torch_pca import PCA

from exp.activations import load_activations, load_activations_and_topk
from viz import FIGURE_DIR

FIGURE_PATH = os.path.join(FIGURE_DIR, "pca_circuits.png")


@arguably.command()
def pca_figure(device: str = "cpu") -> None:
    # Use CPU by default to avoid GPU requirement in tests/CI
    activated_experts, _ = load_activations_and_topk(device=device)

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
