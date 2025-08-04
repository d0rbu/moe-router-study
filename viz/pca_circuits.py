import os

import arguably
import matplotlib.pyplot as plt
from torch_pca import PCA

from viz import FIGURE_DIR
from viz.activations import load_activations

FIGURE_PATH = os.path.join(FIGURE_DIR, "pca_circuits.png")


@arguably.command()
def pca_figure() -> None:
    activated_experts, _, top_k = load_activations()

    # (B, L, E) -> (B, L * E)
    activated_experts = (
        activated_experts.view(activated_experts.shape[0], -1).float().cuda()
    )
    print(activated_experts.shape)

    # PCA to visualize the expert activations
    pca = PCA(n_components=2, svd_solver="full")
    activated_experts_pca = pca.fit_transform(activated_experts).cpu()
    print(activated_experts_pca.shape)

    # scatter plot the expert activations
    plt.scatter(activated_experts_pca[:, 0], activated_experts_pca[:, 1])
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    arguably.run()
