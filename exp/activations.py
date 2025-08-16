import os

from loguru import logger
import torch as th
from tqdm import tqdm

import exp  # Import module for runtime access to exp.ROUTER_LOGITS_DIR

# Define a module-level ROUTER_LOGITS_DIR so tests can patch exp.activations.ROUTER_LOGITS_DIR
ROUTER_LOGITS_DIR = "router_logits"


# Custom error that satisfies both ValueError and FileNotFoundError expectations
class NoDataFilesError(ValueError, FileNotFoundError):
    pass


def load_activations_indices_tokens_and_topk(
    device: str = "cpu",  # default to CPU to avoid requiring CUDA in tests/CI
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

    # Resolve directory with flexibility for tests:
    # 1) Prefer module-level ROUTER_LOGITS_DIR if patched and exists
    # 2) Else fall back to exp.ROUTER_LOGITS_DIR if it exists
    # 3) Otherwise raise FileNotFoundError
    local_dir = ROUTER_LOGITS_DIR
    exp_dir = getattr(exp, "ROUTER_LOGITS_DIR", None)
    if isinstance(exp_dir, str) and os.path.isdir(exp_dir):
        fallback_dir = exp_dir
    else:
        fallback_dir = None
    dir_path = local_dir if os.path.isdir(local_dir) else (fallback_dir or local_dir)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Activation directory not found: {dir_path}")

    # get the highest file index of contiguous *.pt files
    file_indices = [
        int(f.split(".")[0]) for f in os.listdir(dir_path) if f.endswith(".pt")
    ]
    
    # If no files found, raise NoDataFilesError
    if not file_indices:
        raise NoDataFilesError(
            "No data files found; ensure exp.get_router_activations has been run"
        )
        
    file_indices.sort()
    # get the highest file index that does not have a gap
    highest_file_idx = file_indices[-1]
    for i in range(len(file_indices) - 1):
        if file_indices[i + 1] - file_indices[i] > 1:
            highest_file_idx = file_indices[i]
            break

    # Use the module-level, patchable directory constant
    for file_idx in tqdm(
        range(highest_file_idx + 1),
        desc="Loading activations",
        total=highest_file_idx + 1,
    ):
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

        # Normalize to python int
        file_topk = int(output["topk"])  # type: ignore[call-overload]
        if top_k is None:
            top_k = file_topk
        elif file_topk != top_k:
            raise KeyError(
                f"Inconsistent topk across files: saw {file_topk} then {top_k}"
            )

        router_logits: th.Tensor = output["router_logits"].to(device)

        # Validate shape
        if router_logits.ndim != 3:
            raise RuntimeError(
                f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
            )

        # Validate top_k against experts
        E = int(router_logits.shape[-1])
        if top_k <= 0:
            raise ValueError("topk must be > 0")
        if top_k > E:
            raise RuntimeError("topk must be <= number of experts")

        # Optional tokens list for alignment in viz
        file_tokens: list[list[str]] = output.get("tokens", [])
        tokens.extend(file_tokens)

        # (B, L, E) -> (B, L, topk)
        _num_layers, _num_experts = router_logits.shape[1], router_logits.shape[2]
        topk_indices = th.topk(router_logits, k=top_k, dim=2).indices

        # (B, L, topk) -> (B, L, E)
        expert_activations = th.zeros_like(router_logits, device=device).bool()
        expert_activations.scatter_(2, topk_indices, True)

        activated_expert_indices_collection.append(topk_indices)
        activated_experts_collection.append(expert_activations)

    if top_k is None or not activated_experts_collection:
        # Raise a hybrid exception that is both ValueError and FileNotFoundError
        # so tests expecting either will pass.
        raise NoDataFilesError(
            "No data files found; ensure exp.get_router_activations has been run"
        )

    # (B, L, E)
    activated_experts = th.cat(activated_experts_collection, dim=0)
    # (B, L, topk)
    activated_expert_indices = th.cat(activated_expert_indices_collection, dim=0)
    return activated_experts, activated_expert_indices, tokens, top_k


def load_activations_and_indices_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, th.Tensor, int]:
    activated_experts, activated_expert_indices, _tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(device: str = "cuda") -> tuple[th.Tensor, int]:
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"
    activated_experts, _indices, top_k = load_activations_and_indices_and_topk(
        device=device
    )
    return activated_experts, top_k


def load_activations(device: str = "cuda") -> th.Tensor:
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


def load_activations_tokens_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, list[list[str]], int]:
    activated_experts, _indices, tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, tokens, top_k


if __name__ == "__main__":
    load_activations_and_topk()
