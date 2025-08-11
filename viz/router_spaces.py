import os

import arguably
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

from exp.get_weights import WEIGHT_DIR
from viz import FIGURE_DIR

ROUTER_VIZ_DIR = os.path.join(FIGURE_DIR, "router_spaces")


@arguably.command()
def router_spaces() -> None:
    os.makedirs(ROUTER_VIZ_DIR, exist_ok=True)

    router_weight_path = os.path.join(WEIGHT_DIR, "router.pt")
    down_proj_weight_path = os.path.join(WEIGHT_DIR, "down_proj.pt")
    o_proj_weight_path = os.path.join(WEIGHT_DIR, "o_proj.pt")

    router_weights_data = th.load(router_weight_path)
    down_proj_weights_data = th.load(down_proj_weight_path)
    o_proj_weights_data = th.load(o_proj_weight_path)

    router_weights = router_weights_data["weights"]
    _down_proj_weights = down_proj_weights_data["weights"]
    o_proj_weights = o_proj_weights_data["weights"]
    _num_layers = len(o_proj_weights)
    router_layers = list(router_weights.keys())
    num_router_layers = len(router_layers)
    topk = router_weights_data["topk"]

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
    num_circuits = 100
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
