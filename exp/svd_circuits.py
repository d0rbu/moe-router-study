import os
from typing import Optional

import arguably
import matplotlib.pyplot as plt
import torch as th

from exp import get_experiment_dir
from exp.activations import load_activations_and_topk
from viz import get_figure_dir


@arguably.command()
def svd_circuits(
    batch_size: int = 0, 
    num_circuits: int = 64, 
    device: str = "cpu",
    experiment_name: Optional[str] = None
) -> None:
    activated_experts, top_k = load_activations_and_topk(
        experiment_name=experiment_name, device=device
    )

    activated_experts = (
        activated_experts[:batch_size] if batch_size > 0 else activated_experts
    )

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()
    print(activated_experts.shape)

    # Get experiment directory and figure directory
    experiment_dir = get_experiment_dir(name=experiment_name)
    figure_dir = get_figure_dir(experiment_name)
    os.makedirs(figure_dir, exist_ok=True)

    # SVD to get circuits
    u, s, vh = th.linalg.svd(activated_experts)

    # plot singular values
    singular_values_path = os.path.join(figure_dir, "singular_values.png")
    plt.plot(s)
    plt.savefig(singular_values_path, dpi=300, bbox_inches="tight")
    plt.close()

    circuits = vh[:num_circuits, :]
    print(circuits.shape)

    # save circuits
    out = {"circuits": circuits, "top_k": top_k}
    out_path = os.path.join(experiment_dir, "svd_circuits.pt")
    os.makedirs(experiment_dir, exist_ok=True)
    th.save(out, out_path)


if __name__ == "__main__":
    arguably.run()

