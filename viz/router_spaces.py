import os
from typing import cast

import arguably
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import torch as th
from torch import Tensor
from tqdm import tqdm

from core.model import get_model_config
from viz import FIGURE_DIR

ROUTER_VIZ_DIR = os.path.join(FIGURE_DIR, "router_spaces")


@arguably.command()
def router_spaces(
    model_name: str = "olmoe",
    checkpoint_idx: int | None = None,
    device: str = "cpu",
    topk: int = 8,
) -> None:
    """Visualize router spaces directly from model weights.

    This function loads the model weights directly using StandardizedTransformer
    instead of relying on pre-extracted weight files.
    """
    os.makedirs(ROUTER_VIZ_DIR, exist_ok=True)

    model_config = get_model_config(model_name)

    if checkpoint_idx is None:
        revision = None
    else:
        revision = str(model_config.checkpoints[checkpoint_idx])

    model = StandardizedTransformer(
        model_config.hf_name,
        device_map=device,
        revision=revision,
    )

    # Extract weights directly from the model
    router_layers = model.layers_with_routers
    num_router_layers = len(router_layers)

    # Extract router weights
    router_weights: dict[int, th.Tensor] = {}

    with th.no_grad():
        for layer_idx in router_layers:
            router_w = cast("Tensor", model.routers[layer_idx].weight)
            router_weights[layer_idx] = router_w.detach().cpu()

    # first we get the spectra of the router weights
    for layer_idx, router_weight in router_weights.items():
        router_spectrum = th.linalg.svdvals(router_weight)
        plt.plot(router_spectrum)
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"router_spectrum_{layer_idx}.png"), dpi=300
        )
        plt.close()

    sorted_expert_routers = [
        router_weights[layer_idx] for layer_idx in sorted(router_layers)
    ]
    # (L, E, D) -> (L * E, D)
    sorted_expert_vectors = th.cat(sorted_expert_routers, dim=0)

    # next we concatenate the router weights and look at the spectrum
    _u, s, vh = th.linalg.svd(sorted_expert_vectors)
    full_expert_spectrum = s
    plt.plot(full_expert_spectrum)
    plt.savefig(os.path.join(ROUTER_VIZ_DIR, "full_expert_spectrum.png"), dpi=300)
    plt.close()

    # next we look at the cosine similarity between the expert vectors
    sorted_expert_vectors_normalized = sorted_expert_vectors / th.linalg.norm(
        sorted_expert_vectors, dim=-1, keepdim=True
    )
    expert_cosine_similarities = (
        sorted_expert_vectors_normalized @ sorted_expert_vectors_normalized.T
    )
    # plot as a heatmap
    plt.imshow(expert_cosine_similarities, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar()
    plt.savefig(os.path.join(ROUTER_VIZ_DIR, "expert_cosine_similarities.png"), dpi=300)
    plt.close()

    # next we want to find circuits by taking the top right singular vectors
    # and getting the cosine similarity with all expert vectors
    # Use only available singular vectors to avoid shape errors
    num_circuits = min(100, vh.shape[1])
    circuit_vectors = vh[:, :num_circuits]
    # (L * E, D) @ (D, C) -> (L * E, C) -> (L, E, C)
    circuit_logits = (sorted_expert_vectors @ circuit_vectors).view(
        num_router_layers, -1, num_circuits
    )
    circuit_probs = th.nn.functional.softmax(circuit_logits, dim=1)
    circuit_topk = th.topk(circuit_probs, k=topk, dim=1)
    circuit_topk_mask = th.zeros_like(circuit_probs)
    circuit_topk_mask.scatter_(1, circuit_topk.indices, 1)

    for circuit_idx in tqdm(
        range(num_circuits), desc="Plotting circuits", leave=False, total=num_circuits
    ):
        plt.imshow(circuit_topk_mask[:, :, circuit_idx])
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"circuit_topk_mask_{circuit_idx}.png"),
            dpi=300,
        )
        plt.close()

        plt.imshow(circuit_probs[:, :, circuit_idx])
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"circuit_probs_{circuit_idx}.png"), dpi=300
        )
        plt.close()

        plt.imshow(circuit_logits[:, :, circuit_idx])
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"circuit_logits_{circuit_idx}.png"), dpi=300
        )
        plt.close()


if __name__ == "__main__":
    arguably.run()
