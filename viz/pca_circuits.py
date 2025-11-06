import os

import arguably
import matplotlib.pyplot as plt
from torch_pca import PCA

from exp import activations as act
from viz import FIGURE_DIR

FIGURE_PATH = os.path.join(FIGURE_DIR, "pca_circuits.png")


@arguably.command()
def pca_figure(device: str = "cpu") -> None:
    # Load activations with explicit device parameter
    activated_experts, _top_k = act.load_activations_and_topk(device=device)

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()
    print(activated_experts.shape)

    # PCA to visualize the expert activations
    pca = PCA(n_components=2, svd_solver="full")
    activated_experts_pca = pca.fit_transform(activated_experts)
    print(activated_experts_pca.shape)

    # ensure output directory exists
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)

    # scatter plot the expert activations
    plt.scatter(activated_experts_pca[:, 0], activated_experts_pca[:, 1])
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    arguably.run()
