import os

import arguably
import matplotlib.pyplot as plt
from torch_pca import PCA

# Import module so attribute can be monkeypatched by tests reliably
import exp.activations as activations
from viz import FIGURE_DIR

FIGURE_PATH = os.path.join(FIGURE_DIR, "pca_circuits.png")


@arguably.command()
def pca_figure(device: str = "cpu") -> None:
    # Use CPU by default to avoid GPU requirement in tests/CI
    # Call module attribute (tests patch this symbol to simulate failure)
    activated_experts = activations.load_activations(device=device)

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()

    # PCA to visualize the expert activations
    pca = PCA(n_components=2, svd_solver="full")
    activated_experts_pca = pca.fit_transform(activated_experts)

    # Ensure figure directory exists before saving
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)

    # scatter plot the expert activations
    plt.scatter(activated_experts_pca[:, 0], activated_experts_pca[:, 1])
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    arguably.run()
