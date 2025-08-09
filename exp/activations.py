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
    """Return boolean activation mask and top_k, discarding tokens.

    Delegates to load_activations_tokens_and_topk and drops tokens
    to keep a single source of truth for loading.
    """
    activated_experts, _tokens, top_k = load_activations_tokens_and_topk(device=device)
    return activated_experts, top_k


def load_activations(device: str = "cuda") -> th.Tensor:
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


def load_activations_tokens_and_topk(device: str = "cuda") -> tuple[th.Tensor, List[list[str]], int]:
    activated_experts_collection: list[th.Tensor] = []
    tokens: List[list[str]] = []
    top_k: int | None = None

    for file_idx in tqdm(count(), desc="Loading router logits+tokens"):
        file_path = os.path.join(ROUTER_LOGITS_DIR, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break
        output = th.load(file_path)
        if top_k is None:
            top_k = int(output["topk"])  # saved during collection
        router_logits: th.Tensor = output["router_logits"].to(device)
        file_tokens: list[list[str]] = output.get("tokens", [])
        if file_tokens:
            tokens.extend(file_tokens)

        # Build boolean activation mask via topk + scatter
        topk_indices = th.topk(router_logits, k=top_k, dim=2).indices
        expert_activations = th.zeros_like(router_logits, device=device).bool()
        expert_activations.scatter_(2, topk_indices, True)

        activated_experts_collection.append(expert_activations)

    if top_k is None or not activated_experts_collection:
        raise ValueError("No data files found; ensure exp.get_router_activations has been run")

    activated_experts = th.cat(activated_experts_collection, dim=0)
    return activated_experts, tokens, top_k


if __name__ == "__main__":
    load_activations_and_topk()
