from itertools import count
import os

import torch as th
from tqdm import tqdm

# Import directly from the module
from exp.get_router_activations import ROUTER_LOGITS_DIR


def load_activations_indices_tokens_and_topk(
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor, list[list[str]], int]:
    """
    Load the router logits and tokens from the router_logits directory.
    Returns a tuple of (activated_experts, indices, tokens, top_k).
    activated_experts is a boolean tensor of shape (B, L, E) where B is the batch size,
    L is the number of layers, and E is the number of experts.
    indices is a tensor of shape (B, L, top_k) where top_k is the number of experts
    activated per token.
    tokens is a list of lists of strings, where each inner list is the tokens for a batch.
    top_k is the number of experts activated per token.
    """
    activated_experts_list = []
    indices_list = []
    tokens_list = []
    top_k: int | None = None

    for file_idx in tqdm(count(), desc="Loading router logits+tokens"):
        file_path = os.path.join(ROUTER_LOGITS_DIR, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break
        output = th.load(file_path)
        router_logits = output["router_logits"]
        tokens = output["tokens"]
        if top_k is None:
            top_k = output["topk"]
        else:
            assert top_k == output["topk"], "top_k must be the same for all files"

        # router_logits: (B, L, E)
        # indices: (B, L, top_k)
        indices = th.topk(router_logits, k=top_k, dim=2).indices
        # activated_experts: (B, L, E)
        activated_experts = th.zeros_like(router_logits, dtype=th.bool)
        activated_experts.scatter_(2, indices, True)

        activated_experts_list.append(activated_experts)
        indices_list.append(indices)
        tokens_list.extend(tokens)

    if not activated_experts_list:
        raise ValueError(f"No router logits found in {ROUTER_LOGITS_DIR}")

    activated_experts = th.cat(activated_experts_list, dim=0).to(device)
    indices = th.cat(indices_list, dim=0).to(device)

    return activated_experts, indices, tokens_list, top_k


def load_activations_and_topk(
    device: str = "cuda",
) -> tuple[th.Tensor, int]:
    """
    Load the router logits from the router_logits directory.
    Returns a tuple of (activated_experts, top_k).
    activated_experts is a boolean tensor of shape (B, L, E) where B is the batch size,
    L is the number of layers, and E is the number of experts.
    top_k is the number of experts activated per token.
    """
    activated_experts, _, _, top_k = load_activations_indices_tokens_and_topk(
        device=device
    )
    return activated_experts, top_k


def load_activations(
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor, int]:
    """
    Load the router logits from the router_logits directory.
    Returns a tuple of (activated_experts, indices, top_k).
    activated_experts is a boolean tensor of shape (B, L, E) where B is the batch size,
    L is the number of layers, and E is the number of experts.
    indices is a tensor of shape (B, L, top_k) where top_k is the number of experts
    activated per token.
    top_k is the number of experts activated per token.
    """
    activated_experts, indices, _, top_k = load_activations_indices_tokens_and_topk(
        device=device
    )
    return activated_experts, indices, top_k
