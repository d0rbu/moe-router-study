import os

import arguably
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.model import MODELS
from viz import FIGURE_DIR

ROUTER_VIZ_DIR = os.path.join(FIGURE_DIR, "router_spaces")


@arguably.command()
def router_spaces(
    model_name: str = "olmoe",
    checkpoint_idx: int | None = None,
    device: str = "cpu",
    topk: int = 4,
) -> None:
    """Visualize router spaces directly from model weights.

    This function loads the model weights directly using StandardizedTransformer
    instead of relying on pre-extracted weight files.
    """
    os.makedirs(ROUTER_VIZ_DIR, exist_ok=True)

    model_config = MODELS.get(model_name, None)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

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
    o_proj_weights: dict[int, th.Tensor] = {}

    with th.no_grad():
        for layer_idx in router_layers:
            router_weights[layer_idx] = model.routers[layer_idx].weight.detach().cpu()
            o_proj_weights[layer_idx] = model.attentions[layer_idx].o_proj.weight.detach().cpu()

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
    sorted_expert_vectors = th.cat(sorted_expert_routers, dim=1)

    # next we concatenate the router weights and look at the spectrum
    u, s, vh = th.linalg.svd(sorted_expert_vectors)
    full_expert_spectrum = s
    plt.plot(full_expert_spectrum)
    plt.savefig(os.path.join(ROUTER_VIZ_DIR, "full_expert_spectrum.png"), dpi=300)
    plt.close()

    # next we look at the cosine similarity between the expert vectors
    expert_cosine_similarities = th.nn.functional.cosine_similarity(
        sorted_expert_vectors, sorted_expert_vectors, dim=0
    )
    plt.plot(expert_cosine_similarities)
    plt.savefig(os.path.join(ROUTER_VIZ_DIR, "expert_cosine_similarities.png"), dpi=300)
    plt.close()

    # next we want to find circuits by taking the top left singular vectors
    # and getting the cosine similarity with all expert vectors
    # Use only available singular vectors to avoid shape errors
    num_circuits = min(100, u.shape[1])
    circuit_vectors = u[:, :num_circuits]
    circuit_logits = (circuit_vectors.T @ sorted_expert_vectors).view(
        num_circuits, num_router_layers, -1
    )
    circuit_probs = th.nn.functional.softmax(circuit_logits, dim=2)
    circuit_topk = th.topk(circuit_probs, k=topk, dim=2)
    circuit_topk_mask = th.zeros_like(circuit_probs)
    circuit_topk_mask.scatter_(2, circuit_topk.indices, 1)

    for circuit_idx in tqdm(
        range(num_circuits), desc="Plotting circuits", leave=False, total=num_circuits
    ):
        plt.plot(circuit_topk_mask[circuit_idx])
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"circuit_topk_mask_{circuit_idx}.png"),
            dpi=300,
        )
        plt.close()

        plt.plot(circuit_probs[circuit_idx])
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"circuit_probs_{circuit_idx}.png"), dpi=300
        )
        plt.close()

        plt.plot(circuit_logits[circuit_idx])
        plt.savefig(
            os.path.join(ROUTER_VIZ_DIR, f"circuit_logits_{circuit_idx}.png"), dpi=300
        )
        plt.close()


if __name__ == "__main__":
    arguably.run()
