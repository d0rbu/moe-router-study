from itertools import count
import os

import torch as th
from tqdm import tqdm

from exp.get_router_activations import ROUTER_LOGITS_DIR


def load_activations_and_indices_and_topk(device: str = "cuda") -> tuple[th.Tensor, th.Tensor, int]:
    activated_expert_indices_collection: list[th.Tensor] = []
    activated_experts_collection: list[th.Tensor] = []

    for file_idx in tqdm(count(), desc="Loading router logits"):
        file_path = os.path.join(ROUTER_LOGITS_DIR, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break

        output = th.load(file_path)
        top_k = output["topk"]
        router_logits = output["router_logits"].to(device)

        # (B, L, E) -> (B, L, topk)
        num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
        topk_indices = th.topk(router_logits, k=top_k, dim=2).indices

        # (B, L, topk) -> (B, L, E)
        expert_activations = th.zeros_like(router_logits, device=device).bool()
        expert_activations.scatter_(2, topk_indices, True)

        activated_expert_indices_collection.append(topk_indices)
        activated_experts_collection.append(expert_activations)

    if top_k is None:
        raise ValueError("No data files found")

    # (B, L, E)
    activated_experts = th.cat(activated_experts_collection, dim=0)
    # (B, L, topk)
    activated_expert_indices = th.cat(activated_expert_indices_collection, dim=0)

    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(device: str = "cuda") -> tuple[th.Tensor, int]:
    activated_experts, _, top_k = load_activations_and_indices_and_topk(device=device)
    return activated_experts, top_k


def load_activations(device: str = "cuda") -> th.Tensor:
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


if __name__ == "__main__":
    load_activations_and_topk()
