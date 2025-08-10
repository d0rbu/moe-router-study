from itertools import count
import os

import torch as th
from tqdm import tqdm

import exp  # use module so exp.ROUTER_LOGITS_DIR patches are honored

# Provide a module-local alias that tests can patch directly as well
try:  # pragma: no cover - trivial aliasing
    ROUTER_LOGITS_DIR = exp.ROUTER_LOGITS_DIR
except Exception:  # noqa: BLE001 - defensive default
    ROUTER_LOGITS_DIR = "router_logits"


def load_activations_and_indices_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, th.Tensor, int]:
    activated_expert_indices_collection: list[th.Tensor] = []
    activated_experts_collection: list[th.Tensor] = []

    top_k: int | None = None  # initialize to avoid UnboundLocalError when no files
    last_indices_k: int | None = None
    # Resolve directory: prefer exp module (monkeypatchable), then module-level, then default
    dir_path = getattr(exp, "ROUTER_LOGITS_DIR", globals().get("ROUTER_LOGITS_DIR", "router_logits"))
    for file_idx in tqdm(count(), desc="Loading router logits"):
        file_path = os.path.join(dir_path, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break

        try:
            output = th.load(file_path)
        except Exception as e:  # noqa: BLE001 - surface as runtimeerror for tests/CI
            raise RuntimeError(f"Failed to load router logits file: {file_path}") from e

        # Required keys
        if "topk" not in output or "router_logits" not in output:
            missing = [k for k in ("topk", "router_logits") if k not in output]
            raise KeyError(f"Missing keys in logits file: {missing}")

        top_k = int(output["topk"])  # ensure python int for validation
        router_logits = output["router_logits"].to(device)

        # Validate tensor shape
        if router_logits.ndim != 3:
            raise RuntimeError(
                f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
            )
        _, _, num_experts = router_logits.shape

        # Validate top_k
        if top_k <= 0 or top_k > num_experts:
            raise RuntimeError(
                f"Invalid topk {top_k}; must be in [1, {num_experts}] for file {file_idx}"
            )

        # (B, L, E) -> (B, L, topk)
        topk_indices = th.topk(router_logits, k=top_k, dim=2).indices

        # If k changed across files, reset indices collection to ensure concat works
        if last_indices_k is not None and top_k != last_indices_k:
            activated_expert_indices_collection = []
        last_indices_k = top_k

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


def load_activations_and_topk(device: str = "cpu") -> tuple[th.Tensor, int]:
    activated_experts, _, top_k = load_activations_and_indices_and_topk(device=device)
    return activated_experts, top_k


def load_activations(device: str = "cpu") -> th.Tensor:
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


if __name__ == "__main__":
    load_activations_and_topk()
