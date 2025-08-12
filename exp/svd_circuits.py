import os

import arguably
import matplotlib.pyplot as plt
import torch as th

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_topk
from viz import FIGURE_DIR


@arguably.command()
def svd_circuits(
    batch_size: int = 0, num_circuits: int = 64, device: str = "cpu"
) -> None:
    activated_experts, top_k = load_activations_and_topk(device=device)

    activated_experts = (
        activated_experts[:batch_size] if batch_size > 0 else activated_experts
    )

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()
    print(activated_experts.shape)

    # SVD to get circuits
    u, s, vh = th.linalg.svd(activated_experts)

    # plot singular values
    singular_values_path = os.path.join(FIGURE_DIR, "singular_values.png")
    plt.plot(s)
    plt.savefig(singular_values_path, dpi=300, bbox_inches="tight")
    plt.close()

    circuits = vh[:num_circuits, :]
    print(circuits.shape)

    # save circuits
    out = {"circuits": circuits, "top_k": top_k}
    out_path = os.path.join(OUTPUT_DIR, "svd_circuits.pt")
    th.save(out, out_path)


if __name__ == "__main__":
    arguably.run()
