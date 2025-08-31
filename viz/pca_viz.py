import os

import arguably
import matplotlib.pyplot as plt
from torch_pca import PCA

# Import module so attribute can be monkeypatched by tests reliably
import exp.activations as activations
from viz import get_figure_dir


@arguably.command()
def pca_figure(device: str = "cpu", experiment_name: str | None = None) -> None:
    # Use CPU by default to avoid GPU requirement in tests/CI
    # Call module attribute (tests patch this symbol to simulate failure)
    activated_experts = activations.load_activations(
        device=device, experiment_name=experiment_name
    )

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()

    # PCA to visualize the expert activations
    pca = PCA(n_components=2, svd_solver="full")
    activated_experts_pca = pca.fit_transform(activated_experts)

    # Get figure directory for this experiment
    figure_dir = get_figure_dir(experiment_name)
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, "pca_circuits.png")

    # scatter plot the expert activations
    plt.scatter(activated_experts_pca[:, 0], activated_experts_pca[:, 1])
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    arguably.run()
