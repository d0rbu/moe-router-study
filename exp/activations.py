from itertools import count
import os

import torch as th
from tqdm import tqdm

# Import module to allow runtime access to exp.ROUTER_LOGITS_DIR (monkeypatchable)
import exp


def load_activations_indices_tokens_and_topk(
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor, list[list[str]], int]:
    """Load boolean activation mask, top-k indices, tokens, and top_k.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - activated_expert_indices: (B, L, topk) long indices of selected experts
      - tokens: list[list[str]] tokenized sequences aligned to batch concatenation
      - top_k: int top-k used during collection
    """
    activated_expert_indices_collection: list[th.Tensor] = []
    activated_experts_collection: list[th.Tensor] = []
    tokens: list[list[str]] = []
    top_k: int | None = None  # handle case of no files

    # Merge resolution: use getattr(exp, "ROUTER_LOGITS_DIR") and support tokens from files
    for file_idx in tqdm(count(), desc="Loading router logits+tokens"):
        dir_path = getattr(exp, "ROUTER_LOGITS_DIR", "router_logits")
        file_path = os.path.join(dir_path, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break

        try:
            output = th.load(file_path)
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load router logits file: {file_path}") from e

        # Required keys
        if "topk" not in output or "router_logits" not in output:
            missing = [k for k in ("topk", "router_logits") if k not in output]
            raise KeyError(f"Missing keys in logits file: {missing}")

        if top_k is None:
            top_k = int(output["topk"])  # normalize to python int
        router_logits: th.Tensor = output["router_logits"].to(device)

        # Validate shape
        if router_logits.ndim != 3:
            raise RuntimeError(
                f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
            )

        # Optional tokens list for alignment in viz
        file_tokens: list[list[str]] = output.get("tokens", [])
        tokens.extend(file_tokens)

        # Build top-k indices and boolean mask via topk + scatter
        topk_indices = th.topk(router_logits, k=top_k, dim=2).indices  # (B, L, topk)
        expert_activations = th.zeros_like(
            router_logits, device=device
        ).bool()  # (B, L, E)
        expert_activations.scatter_(2, topk_indices, True)

        activated_expert_indices_collection.append(topk_indices)
        activated_experts_collection.append(expert_activations)

    if top_k is None or not activated_experts_collection:
        raise ValueError(
            "No data files found; ensure exp.get_router_activations has been run"
        )

    # (B, L, E)
    activated_experts = th.cat(activated_experts_collection, dim=0)
    # (B, L, topk)
    activated_expert_indices = th.cat(activated_expert_indices_collection, dim=0)
    return activated_experts, activated_expert_indices, tokens, top_k


def load_activations_and_indices_and_topk(
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor, int]:
    activated_experts, activated_expert_indices, _tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(device: str = "cuda") -> tuple[th.Tensor, int]:
    activated_experts, _indices, top_k = load_activations_and_indices_and_topk(
        device=device
    )
    return activated_experts, top_k


def load_activations(device: str = "cuda") -> th.Tensor:
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


def load_activations_tokens_and_topk(
    device: str = "cuda",
) -> tuple[th.Tensor, list[list[str]], int]:
    activated_experts, _indices, tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, tokens, top_k


if __name__ == "__main__":
    load_activations_and_topk()
