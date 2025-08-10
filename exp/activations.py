from itertools import count
import os

import torch as th
from tqdm import tqdm

# Import module to allow runtime access to exp.ROUTER_LOGITS_DIR (monkeypatchable)
import exp


def load_activations_and_indices_and_topk(device: str = "cuda") -> tuple[th.Tensor, th.Tensor, int]:
    activated_expert_indices_collection: list[th.Tensor] = []
    activated_experts_collection: list[th.Tensor] = []

    top_k: int | None = None  # handle case of no files

    for file_idx in tqdm(count(), desc="Loading router logits"):
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

        top_k = int(output["topk"])  # normalize to python int
        router_logits = output["router_logits"].to(device)

        # Validate shape
        if router_logits.ndim != 3:
            raise RuntimeError(
                f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
            )

        # (B, L, E) -> (B, L, topk)
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
