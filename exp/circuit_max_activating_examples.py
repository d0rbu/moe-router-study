from itertools import pairwise

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import torch as th

# Use the topk+scatter-based loader that builds a boolean activation mask
from exp.activations import (
    load_activations_and_topk,
)


def get_circuit_activations(
    circuits: th.Tensor,
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor]:
    """Compute circuit activations for every token from top-k activation mask.

    Steps:
    1) Load boolean top-k activation mask (B, L, E) via topk + scatter (from exp.activations)
    2) Compute activations = einsum("ble,cle->bc", mask.float(), circuits)
    """
    # Build top-k activation mask (B, L, E) via topk + scatter over last dim
    token_topk_mask, _topk = load_activations_and_topk(device=device)

    # Ensure circuits is on the same device and dtype
    circuits = circuits.to(device=device, dtype=th.float32)

    # Compute per-token, per-circuit activation: dot product over (L,E)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    return activations, token_topk_mask


def _color_for_value(
    val: float, vmin: float = 0.0, vmax: float = 1.0
) -> tuple[float, float, float]:
    # Map activation value to color (Blues colormap)
    normalized = 0.0 if vmax <= vmin else (val - vmin) / (vmax - vmin)
    cmap = plt.get_cmap("Blues")
    r, g, b, _ = cmap(normalized)
    return (r, g, b)


def _render_circuit(ax: Axes, circuit: np.ndarray) -> tuple:
    ax.clear()
    im = ax.imshow(circuit, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_title("Circuit (L x E)")
    ax.set_xlabel("Experts")
    ax.set_ylabel("Layers")

    # Return image to allow overlay updates
    return (im,)


def _ensure_token_alignment(
    token_topk_mask: th.Tensor, sequences: list[list[str]]
) -> None:
    # Best-effort sanity check: make sure token count matches
    total_tokens = sum(len(s) for s in sequences)
    if token_topk_mask.shape[0] != total_tokens:
        raise ValueError(
            f"Token count mismatch: activations B={token_topk_mask.shape[0]} vs total tokens {total_tokens}. "
            "Ensure you use the same dataset and ordering as when collecting router logits."
        )


def build_sequence_id_tensor(
    sequences: list[list[str]],
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Map each token to its sequence index and compute lengths/offsets.

    Returns:
      - seq_ids_per_token: (B,) long tensor mapping token index -> seq index
      - seq_lengths: (S,) long tensor of token counts per sequence
      - seq_offsets: (S+1,) long tensor of prefix sums, start index per sequence
    """
    lengths = [len(seq) for seq in sequences]
    S = len(lengths)
    seq_lengths = th.tensor(lengths, dtype=th.long)
    # Prefix sums with initial 0 so we have S+1 entries
    seq_offsets = th.empty(S + 1, dtype=th.long)
    seq_offsets[0] = 0
    if S > 0:
        seq_offsets[1:] = th.cumsum(seq_lengths, dim=0)
    B = int(seq_offsets[-1].item()) if S > 0 else 0
    if B == 0:
        return th.empty(0, dtype=th.long), seq_lengths, seq_offsets

    seq_ids = th.empty(B, dtype=th.long)
    # Use itertools.pairwise for successive pairs; operate on Python ints for slicing
    offsets_list = seq_offsets.tolist()
    for s, (start, end) in enumerate(pairwise(offsets_list)):
        seq_ids[start:end] = s
    return seq_ids, seq_lengths, seq_offsets


def _gather_top_sequences_by_max(
    token_scores: th.Tensor,
    seq_ids: th.Tensor,
    top_n: int,
) -> th.Tensor:
    device = token_scores.device
    B = int(token_scores.shape[0])
    order = th.argsort(token_scores, descending=True)
    seq_sorted = seq_ids[order]

    S = int(seq_ids.max().item()) + 1 if seq_ids.numel() > 0 else 0
    assert S > 0, "No sequences present"

    earliest = th.full((S,), B, device=device)
    idx_src = th.arange(B, device=device)
    earliest = earliest.scatter_reduce(
        0, seq_sorted, idx_src, reduce="amin", include_self=True
    )

    assert (earliest < B).all(), "Every sequence must have at least one token"

    topk = th.argsort(earliest)[:top_n]
    return topk


def _gather_top_sequences_by_mean(
    token_scores: th.Tensor,
    seq_ids: th.Tensor,
    seq_lengths: th.Tensor,
    top_n: int,
) -> th.Tensor:
    S = int(seq_lengths.shape[0])
    sums = th.zeros(S, dtype=token_scores.dtype, device=token_scores.device)
    sums = sums.index_add(0, seq_ids.to(sums.device), token_scores)
    means = sums / seq_lengths.to(sums.device)
    order = th.argsort(means, descending=True)
    return order[:top_n]


def _validate_seq_mapping(seq_ids: th.Tensor, seq_lengths: th.Tensor) -> None:
    S = int(seq_lengths.shape[0])
    counts = th.bincount(seq_ids.cpu(), minlength=S)
    if counts.shape[0] < S:
        counts = th.nn.functional.pad(counts, (0, S - counts.shape[0]))
    if not th.equal(counts.to(seq_lengths.dtype), seq_lengths.cpu()):
        raise ValueError(
            "Sequence ID mapping mismatch: per-sequence token counts do not match provided lengths."
        )
    assert (seq_lengths > 0).all(), "Every sequence must have at least one token"
