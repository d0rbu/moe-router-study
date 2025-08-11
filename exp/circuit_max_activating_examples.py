from matplotlib.axes import Axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch as th

from exp.activations import (
    load_activations_and_topk,
    load_activations_tokens_and_topk,
)


def get_circuit_activations(
    _circuits: th.Tensor,
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor]:
    """Compute circuit activations for every token from top-k activation mask.

    Steps:
    1) Load boolean top-k activation mask (B, L, E)
    2) Compute activations = einsum("ble,cle->bc", mask.float(), circuits)
    Returns (activations, token_topk_mask).
    """
    # Resolve merge artifact: only need the mask; ignore top_k value here
    token_topk_mask, _ = load_activations_and_topk(device=device)
    circuits = circuits.to(device=device, dtype=th.float32)
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


def _ensure_token_alignment(
    token_topk_mask: th.Tensor, sequences: list[list[str]]
) -> None:
    # Sanity check: ensure token dimension lines up with flattened tokens across sequences
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
    """Select sequences ordered by presence of highest-scoring tokens.

    We sort tokens by descending score, then take the earliest token index per sequence.
    Sorting those earliest indices yields the prioritized sequence order. Finally, take top_n.
    """
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
    """Select sequences ordered by mean token score."""
    S = int(seq_lengths.shape[0])
    sums = th.zeros(S, dtype=token_scores.dtype, device=token_scores.device)
    sums = sums.index_add(0, seq_ids.to(sums.device), token_scores)
    means = sums / seq_lengths.to(sums.device)
    order = th.argsort(means, descending=True)
    return order[:top_n]


def _make_token_color(val: float) -> tuple[float, float, float, float]:
    # Return RGBA with some baseline alpha
    r, g, b = _color_for_value(float(np.clip(val, 0.0, 1.0)))
    return (r, g, b, 0.9)


def _render_sequences_panel(
    axes: list[Axes],
    sequences: list[list[str]],
    seq_indices: list[int],
    token_scores: np.ndarray,
    seq_offsets: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Render the left panel with selected sequences and colored tokens.

    Returns a mapping from token slot id to (axis_idx, global_token_idx) for hover lookup.
    """
    token_slot_to_global: dict[int, tuple[int, int]] = {}
    slot_id = 0

    for ax in axes:
        ax.clear()

    for ax_idx, (ax, seq_id) in enumerate(zip(axes, seq_indices, strict=False)):
        tokens_in_seq = sequences[seq_id]
        n_tok = len(tokens_in_seq)
        ax.set_xlim(0, n_tok)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Compute global start token index for this sequence
        start = int(seq_offsets[seq_id])
        # draw rectangles for tokens with colors based on token_scores
        for i, tok in enumerate(tokens_in_seq):
            global_token_idx = start + i
            score = float(token_scores[global_token_idx])
            rect = patches.Rectangle(
                (i, 0.0),
                1.0,
                1.0,
                facecolor=_make_token_color(score),
                edgecolor="white",
            )
            ax.add_patch(rect)
            ax.text(
                i + 0.5,
                0.5,
                tok,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                clip_on=True,
            )
            token_slot_to_global[slot_id] = (ax_idx, global_token_idx)
            slot_id += 1

        ax.set_title(f"Sequence {seq_id}")

    return token_slot_to_global


def _viz_render_precomputed(
    circuits: th.Tensor,
    sequences: list[list[str]],
    norm_scores: th.Tensor,  # (B, C) in [0,1]
    order_per_circuit: list[list[int]],  # len C, ordered seq ids per circuit
    top_n: int = 10,
    token_topk_mask: th.Tensor | None = None,
    device: str = "cuda",
) -> None:
    """Render interactive view using precomputed scores and ordering.

    - Left: tokens highlighted by activation (0..1) for current circuit
    - Right: circuit grid with slider to switch circuits
    - Hover: overlay actual top-k activation mask for hovered token
    """
    # Load token-level top-k mask if not provided (used for hover overlay)
    if token_topk_mask is None:
        token_topk_mask, _ = load_activations_and_topk(device=device)
    circuits = circuits.to(device=device, dtype=th.float32)

    # Validate alignment with provided sequences
    _ensure_token_alignment(token_topk_mask, sequences)

    # Build lengths/offsets for hover mapping
    _seq_ids, _seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)

    # Setup figure: left side stacked sequences, right side circuit grid + slider
    C = int(circuits.shape[0])
    L, E = int(circuits.shape[-2]), int(circuits.shape[-1])

    # Figure layout
    n_rows = top_n
    fig = plt.figure(figsize=(14, 1.5 * n_rows + 4))
    gs = fig.add_gridspec(
        n_rows + 1, 2, width_ratios=[3, 2], height_ratios=[*([1] * n_rows), 0.2]
    )

    seq_axes: list[Axes] = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
    circuit_ax: Axes = fig.add_subplot(gs[:n_rows, 1])
    slider_ax: Axes = fig.add_subplot(gs[n_rows, :])

    # Initial circuit index
    circuit_idx = 0

    # Initialize circuit image and overlay
    circuit_im = circuit_ax.imshow(
        circuits[circuit_idx].detach().cpu().numpy(),
        cmap="Greys",
        aspect="auto",
        interpolation="nearest",
    )
    circuit_ax.set_title(f"Circuit {circuit_idx + 1}/{C} (L={L}, E={E})")
    circuit_ax.set_xlabel("Experts")
    circuit_ax.set_ylabel("Layers")

    overlay_im = circuit_ax.imshow(
        np.zeros((L, E), dtype=float),
        cmap="Reds",
        aspect="auto",
        interpolation="nearest",
        alpha=0.0,
    )

    # Initial render
    current_norm_scores = norm_scores[:, circuit_idx]
    top_seq_ids = order_per_circuit[circuit_idx][:top_n]
    _render_sequences_panel(
        seq_axes,
        sequences,
        [int(s) for s in top_seq_ids],
        current_norm_scores.detach().cpu().numpy(),
        seq_offsets.detach().cpu().numpy(),
    )

    # Slider for circuit selection
    slider = Slider(slider_ax, "Circuit", 0, C - 1, valinit=circuit_idx, valstep=1)

    # Hover handling mapping from axes to current displayed sequence IDs
    ax_to_seq = dict(zip(seq_axes, top_seq_ids, strict=False))

    def on_slider_change(val: float) -> None:
        # Resolve merge artifact: remove unused token_mapping from nonlocal
        nonlocal circuit_idx, top_seq_ids, current_norm_scores, ax_to_seq
        circuit_idx = int(val)
        circuit_im.set_array(circuits[circuit_idx].detach().cpu().numpy())
        circuit_ax.set_title(f"Circuit {circuit_idx + 1}/{C} (L={L}, E={E})")
        current_norm_scores = norm_scores[:, circuit_idx]
        top_seq_ids = order_per_circuit[circuit_idx][:top_n]
        _render_sequences_panel(
            seq_axes,
            sequences,
            [int(s) for s in top_seq_ids],
            current_norm_scores.detach().cpu().numpy(),
            seq_offsets.detach().cpu().numpy(),
        )
        # Rebuild axis->sequence mapping for hover after re-render
        ax_to_seq = dict(zip(seq_axes, top_seq_ids, strict=False))
        overlay_im.set_alpha(0.0)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    # Hover handling: update overlay based on token under cursor
    def on_motion(event) -> None:
        if event.inaxes not in seq_axes:
            return
        ax = event.inaxes
        seq_id = ax_to_seq.get(ax)
        if seq_id is None or event.xdata is None:
            return
        # Determine token index in this sequence from x coordinate
        i = int(event.xdata)
        if i < 0:
            return
        if seq_id >= len(sequences) or i >= len(sequences[seq_id]):
            return
        # Compute global token index
        start = int(seq_offsets[seq_id].item())
        global_token_idx = start + i
        # Update overlay to show this token's top-k mask
        mask = (
            cast("th.Tensor", token_topk_mask)[global_token_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        overlay_im.set_array(mask)
        overlay_im.set_alpha(0.35)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    fig.suptitle("Activating tokens viewer")
    plt.tight_layout()
    plt.show()


def viz_max_activating_tokens(
    circuits: th.Tensor, top_n: int = 10, device: str = "cuda"
) -> None:
    """Visualize top-N sequences by containing highest-activating tokens.

    - Left: tokens highlighted by activation (0..1) for current circuit.
    - Right: circuit grid with slider to switch circuits.
    - Hover a token to see its actual top-k router activation mask overlaid as transparency.
    """
    # Load tokens and mask for alignment + scoring
    token_topk_mask, _ = load_activations_and_topk(device=device)
    _activated_experts, tokens, _ = load_activations_tokens_and_topk(device=device)
    _ensure_token_alignment(token_topk_mask, tokens)
    circuits = circuits.to(device=device, dtype=th.float32)

    # Compute per-token, per-circuit activations (B, C)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    C = int(activations.shape[1])

    # Map tokens to sequences
    seq_ids, _seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens)

    # Normalize by theoretical max: top_k * num_layers
    _, top_k = load_activations_and_topk(device=device)
    L = int(circuits.shape[-2])
    denom = th.tensor(
        float(top_k * L), device=activations.device, dtype=activations.dtype
    )
    norm_scores = (activations / denom).clamp(0, 1)

    # Build ordered sequence list per circuit by max token criteria
    order_per_circuit: list[list[int]] = []
    for c in range(C):
        order = _gather_top_sequences_by_max(norm_scores[:, c], seq_ids, top_n=10**9)
        order_per_circuit.append([int(s.item()) for s in order])

    _viz_render_precomputed(
        circuits,
        tokens,
        norm_scores,
        order_per_circuit,
        top_n=top_n,
        token_topk_mask=token_topk_mask,
        device=device,
    )


def viz_mean_activating_tokens(
    circuits: th.Tensor, top_n: int = 10, device: str = "cuda"
) -> None:
    """Visualize top-N sequences by highest mean token activation.

    Same as viz_max_activating_tokens, but selecting sequences by mean token score.
    """
    token_topk_mask, _ = load_activations_and_topk(device=device)
    _activated_experts, tokens, _ = load_activations_tokens_and_topk(device=device)
    _ensure_token_alignment(token_topk_mask, tokens)
    circuits = circuits.to(device=device, dtype=th.float32)

    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    C = int(activations.shape[1])

    seq_ids, seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens)

    # Normalize by theoretical max: top_k * num_layers
    _, top_k = load_activations_and_topk(device=device)
    L = int(circuits.shape[-2])
    denom = th.tensor(
        float(top_k * L), device=activations.device, dtype=activations.dtype
    )
    norm_scores = (activations / denom).clamp(0, 1)

    # Order sequences per circuit using mean per sequence
    order_per_circuit: list[list[int]] = []
    for c in range(C):
        order = _gather_top_sequences_by_mean(
            norm_scores[:, c], seq_ids, seq_lengths, top_n=10**9
        )
        order_per_circuit.append([int(s.item()) for s in order])

    _viz_render_precomputed(
        circuits,
        tokens,
        norm_scores,
        order_per_circuit,
        top_n=top_n,
        token_topk_mask=token_topk_mask,
        device=device,
    )
