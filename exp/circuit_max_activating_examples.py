from itertools import count
from typing import Callable, Dict, List, Tuple

import arguably
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

# Use the topk+scatter-based loader that builds a boolean activation mask
from exp.activations import (
    load_activations_and_topk,
    load_activations_tokens_and_topk,
)


def get_circuit_activations(
    circuits: th.Tensor,
    device: str = "cuda",
) -> Tuple[th.Tensor, th.Tensor]:
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


def _color_for_value(val: float, vmin: float = 0.0, vmax: float = 1.0) -> Tuple[float, float, float]:
    # Map activation value to color (Blues colormap)
    normalized = 0.0 if vmax <= vmin else (val - vmin) / (vmax - vmin)
    cmap = plt.get_cmap("Blues")
    r, g, b, _ = cmap(normalized)
    return (r, g, b)


def _render_circuit(ax: Axes, circuit: np.ndarray) -> Tuple:
    ax.clear()
    im = ax.imshow(circuit, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_title("Circuit (L x E)")
    ax.set_xlabel("Experts")
    ax.set_ylabel("Layers")

    # Return image to allow overlay updates
    return (im,)


def _ensure_token_alignment(token_topk_mask: th.Tensor, sequences: List[List[str]]) -> None:
    # Best-effort sanity check: make sure token count matches
    total_tokens = sum(len(s) for s in sequences)
    if token_topk_mask.shape[0] != total_tokens:
        raise ValueError(
            f"Token count mismatch: activations B={token_topk_mask.shape[0]} vs total tokens {total_tokens}. "
            "Ensure you use the same dataset and ordering as when collecting router logits."
        )


def build_sequence_id_tensor(sequences: List[List[str]]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
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
    for s, (start, end) in enumerate(zip(seq_offsets[:-1], seq_offsets[1:])):
        seq_ids[start:end] = s
    return seq_ids, seq_lengths, seq_offsets


def _gather_top_sequences_by_max(
    token_scores: th.Tensor,
    seq_ids: th.Tensor,
    top_n: int,
) -> List[int]:
    device = token_scores.device
    B = int(token_scores.shape[0])
    order = th.argsort(token_scores, descending=True)
    seq_sorted = seq_ids[order]

    S = int(seq_ids.max().item()) + 1 if seq_ids.numel() > 0 else 0
    assert S > 0, "No sequences present"

    earliest = th.full((S,), B, device=device)
    idx_src = th.arange(B, device=device)
    earliest = earliest.scatter_reduce(0, seq_sorted, idx_src, reduce="amin", include_self=True)

    assert (earliest < B).all(), "Every sequence must have at least one token"

    topk = th.argsort(earliest)[:top_n]
    return topk.tolist()


def _gather_top_sequences_by_mean(
    token_scores: th.Tensor,
    seq_ids: th.Tensor,
    seq_lengths: th.Tensor,
    top_n: int,
) -> List[int]:
    S = int(seq_lengths.shape[0])
    sums = th.zeros(S, dtype=token_scores.dtype, device=token_scores.device)
    sums = sums.index_add(0, seq_ids.to(sums.device), token_scores)
    means = sums / seq_lengths.to(sums.device)
    order = th.argsort(means, descending=True)
    return order[:top_n].tolist()


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


def _render_sequences_panel(
    ax: Axes,
    sequences: List[List[str]],
    seq_indices: List[int],
    seq_offsets: th.Tensor,
    token_scores_for_circuit: th.Tensor,
    vmin: float,
    vmax: float,
) -> Dict:
    """Render selected sequences with token coloring based on scores.

    Returns a dictionary with mapping from artist -> global_token_idx for picking.
    """
    ax.clear()
    ax.set_title("Top sequences (click token to overlay mask)")
    ax.set_axis_off()

    artist_to_token: Dict = {}
    y = 0.95  # normalized axes coords
    line_height = 0.035

    token_scores_np = token_scores_for_circuit.detach().cpu().numpy()
    for seq_idx in seq_indices:
        tokens = sequences[seq_idx]
        start = int(seq_offsets[seq_idx].item())
        ax.text(0.01, y, f"S{seq_idx}", transform=ax.transAxes, fontsize=8, ha="left", va="top", color="black")
        x = 0.07
        for i, tok in enumerate(tokens):
            global_idx = start + i
            score = float(token_scores_np[global_idx])
            color = _color_for_value(score, vmin, vmax)
            t = ax.text(
                x,
                y,
                tok,
                transform=ax.transAxes,
                fontsize=8,
                ha="left",
                va="top",
                color=color,
                picker=True,
            )
            artist_to_token[t] = global_idx
            x += 0.012 + 0.006 * max(1, len(tok))
            if x > 0.98:
                break
        y -= line_height
        if y < 0.02:
            break

    return artist_to_token


def _viz_ui(
    circuits: th.Tensor,
    activations: th.Tensor,
    token_topk_mask: th.Tensor,
    sequences: List[List[str]],
    seq_ids: th.Tensor,
    seq_lengths: th.Tensor,
    seq_offsets: th.Tensor,
    top_n: int,
    select_sequences: Callable[[th.Tensor, th.Tensor, th.Tensor, int], List[int]],
) -> None:
    """Shared UI for both max- and mean-based selection.

    select_sequences(token_scores, seq_ids, seq_lengths, top_n) -> List[int]
    """
    B, C = activations.shape
    L, E = token_topk_mask.shape[1], token_topk_mask.shape[2]

    fig: Figure
    ax_left: Axes
    ax_right: Axes
    ax_slider: Axes
    fig = plt.figure(figsize=(16, 10))
    ax_left = fig.add_axes([0.05, 0.15, 0.6, 0.8])
    ax_right = fig.add_axes([0.7, 0.3, 0.28, 0.65])
    ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])

    current_c = 0
    per_c_vmin = activations.min(dim=0).values.detach().cpu().numpy()
    per_c_vmax = activations.max(dim=0).values.detach().cpu().numpy()

    circuit_np = circuits[current_c].detach().cpu().numpy()
    (im_circuit,) = _render_circuit(ax_right, circuit_np)
    overlay = ax_right.imshow(
        np.zeros((L, E)), cmap="Reds", alpha=0.0, aspect="auto", interpolation="nearest"
    )

    token_scores_for_c = activations[:, current_c]
    selected_sequences = select_sequences(token_scores_for_c, seq_ids, seq_lengths, top_n)
    artist_map = _render_sequences_panel(
        ax_left,
        sequences,
        selected_sequences,
        seq_offsets,
        token_scores_for_c,
        vmin=float(per_c_vmin[current_c]),
        vmax=float(per_c_vmax[current_c]),
    )

    slider = Slider(ax=ax_slider, label="Circuit", valmin=0, valmax=max(0, C - 1), valinit=current_c, valstep=1)

    def update_overlay_for_token(token_idx: int | None) -> None:
        if token_idx is None:
            overlay.set_alpha(0.0)
            fig.canvas.draw_idle()
            return
        mask = token_topk_mask[token_idx].detach().cpu().numpy().astype(np.float32)
        overlay.set_data(mask)
        overlay.set_alpha(0.35)
        fig.canvas.draw_idle()

    last_hovered = {"token": None}

    def on_hover(event) -> None:  # type: ignore[no-redef]
        nonlocal artist_map
        if event.inaxes != ax_left:
            if last_hovered["token"] is not None:
                last_hovered["token"] = None
                update_overlay_for_token(None)
            return
        for artist, token_idx in artist_map.items():
            contains, _ = artist.contains(event)
            if contains:
                if last_hovered["token"] != token_idx:
                    last_hovered["token"] = token_idx
                    update_overlay_for_token(token_idx)
                return
        if last_hovered["token"] is not None:
            last_hovered["token"] = None
            update_overlay_for_token(None)

    def on_slider_change(val) -> None:  # type: ignore[no-redef]
        nonlocal current_c, artist_map
        current_c = int(slider.val)
        im_circuit.set_array(circuits[current_c].detach().cpu().numpy())
        ax_right.set_title(f"Circuit {current_c} (L x E)")
        token_scores = activations[:, current_c]
        seqs = select_sequences(token_scores, seq_ids, seq_lengths, top_n)
        artist_map = _render_sequences_panel(
            ax_left,
            sequences,
            seqs,
            seq_offsets,
            token_scores,
            vmin=float(per_c_vmin[current_c]),
            vmax=float(per_c_vmax[current_c]),
        )
        update_overlay_for_token(None)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    slider.on_changed(on_slider_change)
    ax_left.set_title("Top sequences (hover token to overlay mask)")
    plt.show()


def _viz_common(
    circuits: th.Tensor,
    top_n: int = 64,
    device: str = "cuda",
) -> None:
    # Compute activations and load dataset tokens via data loader
    token_topk_mask, tokens, _topk = load_activations_tokens_and_topk(device=device)
    sequences = tokens

    # Now compute activations from the mask
    circuits = circuits.to(device=device, dtype=th.float32)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)

    _ensure_token_alignment(token_topk_mask, sequences)

    seq_ids, seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)
    _validate_seq_mapping(seq_ids, seq_lengths)

    # Move ids to same device as activations for quick ops
    seq_ids = seq_ids.to(activations.device)
    seq_lengths = seq_lengths.to(activations.device)

    # Select max-containing sequences by default
    def select_max(scores: th.Tensor, ids: th.Tensor, lengths: th.Tensor, k: int) -> List[int]:
        return _gather_top_sequences_by_max(scores, ids, k)

    _viz_ui(
        circuits,
        activations,
        token_topk_mask,
        sequences,
        seq_ids,
        seq_lengths,
        seq_offsets,
        top_n,
        select_sequences=select_max,
    )


@arguably.command()
def viz_max_containing_tokens(
    circuits_path: str,
    top_n: int = 64,
    device: str = "cuda",
) -> None:
    """Visualize sequences containing the most activating tokens for a circuit.

    circuits_path should be a .pt file containing a tensor of shape (C, L, E)
    with entries in [0, 1].
    """
    circuits = th.load(circuits_path)
    if isinstance(circuits, dict) and "circuits" in circuits:
        circuits = circuits["circuits"]
    circuits = th.as_tensor(circuits, dtype=th.float32)
    _viz_common(circuits, top_n=top_n, device=device)


@arguably.command()
def viz_mean_activating_tokens(
    circuits_path: str,
    top_n: int = 64,
    device: str = "cuda",
) -> None:
    """Visualize sequences whose tokens are, on average, most activated by a circuit.

    circuits_path should be a .pt file containing a tensor of shape (C, L, E)
    with entries in [0, 1].
    """
    circuits = th.load(circuits_path)
    if isinstance(circuits, dict) and "circuits" in circuits:
        circuits = circuits["circuits"]
    circuits = th.as_tensor(circuits, dtype=th.float32)

    # Compute mask, tokens, activations
    token_topk_mask, tokens, _ = load_activations_tokens_and_topk(device=device)
    sequences = tokens
    circuits = circuits.to(device=device, dtype=th.float32)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)

    # Build ids and validate
    seq_ids, seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)
    _validate_seq_mapping(seq_ids, seq_lengths)
    seq_ids = seq_ids.to(activations.device)
    seq_lengths = seq_lengths.to(activations.device)

    def select_mean(scores: th.Tensor, ids: th.Tensor, lengths: th.Tensor, k: int) -> List[int]:
        return _gather_top_sequences_by_mean(scores, ids, lengths, k)

    _viz_ui(
        circuits,
        activations,
        token_topk_mask,
        sequences,
        seq_ids,
        seq_lengths,
        seq_offsets,
        top_n,
        select_sequences=select_mean,
    )


if __name__ == "__main__":
    arguably.run()
