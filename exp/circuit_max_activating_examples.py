from collections.abc import Callable
from itertools import pairwise

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

    Returns a mapping from axis id + token slot to global token index for hover lookup.
    Mapping: key is a monotonically increasing token slot id, value is (axis_idx, global_token_idx)
    """
    token_slot_to_global: dict[int, tuple[int, int]] = {}
    slot_id = 0

    for ax in axes:
        ax.clear()

    for ax_idx, (ax, seq_id) in enumerate(zip(axes, seq_indices, strict=False)):
        tokens = sequences[seq_id]
        n_tok = len(tokens)
        ax.set_xlim(0, n_tok)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Compute global start and end token indices for this sequence
        start = int(seq_offsets[seq_id])
        # draw rectangles for tokens with colors based on token_scores
        for i, tok in enumerate(tokens):
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


def _viz_common(
    circuits: th.Tensor,
    sequences: list[list[str]],
<<<<<<< HEAD
    seq_order_per_circuit: list[list[int]],
    token_scores_per_circuit: th.Tensor,  # (C, B) normalized 0..1
    *,
=======
    select_sequences_for_circuit: Callable[[int], tuple[list[int], th.Tensor]],
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
    top_n: int = 10,
    device: str = "cuda",
    title: str | None = None,
) -> None:
    """Render interactive view given precomputed selection and token scores.

    Args:
      circuits: (C, L, E)
      sequences: list of token lists
      seq_order_per_circuit: list with length C; each element is a list of sequence ids ordered by preference
      token_scores_per_circuit: (C, B) token scores for coloring, normalized to [0,1]
      top_n: number of sequences to display per circuit
      device: device for loading overlays
      title: window title
    """
    # Load token-level top-k mask for hover overlay and validate alignment
    token_topk_mask, _topk = load_activations_and_topk(device=device)
    circuits = circuits.to(device=device, dtype=th.float32)
    _ensure_token_alignment(token_topk_mask, sequences)

    # Build seq id mapping for offsets
    _seq_ids, seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)
    _validate_seq_mapping(_seq_ids, seq_lengths)

<<<<<<< HEAD
=======
    # Compute activations per circuit per token: (B, C) — left here for alignment checks if needed
    _ = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)

    # Build seq id mapping and lengths/offsets
    seq_ids, seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)
    _validate_seq_mapping(seq_ids, seq_lengths)

    # Setup figure: left side stacked sequences, right side circuit grid + slider
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
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

    circuit_idx = 0

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

<<<<<<< HEAD
    def render_for_circuit(c_idx: int) -> tuple[list[int], dict[int, tuple[int, int]]]:
        seq_ids = seq_order_per_circuit[c_idx][:top_n]
        token_scores = token_scores_per_circuit[c_idx].detach().cpu().numpy()
        mapping = _render_sequences_panel(
            seq_axes,
            sequences,
            [int(s) for s in seq_ids],
            token_scores,
            seq_offsets.detach().cpu().numpy(),
        )
        return [int(s) for s in seq_ids], mapping

    top_seq_ids, token_mapping = render_for_circuit(circuit_idx)
=======
    # Wrapper that renders using externally-provided selection for each circuit
    def render_for_circuit(
        c_idx: int,
    ) -> tuple[list[int], th.Tensor, dict[int, tuple[int, int]]]:
        top_seq_ids, norm_scores = select_sequences_for_circuit(c_idx)
        mapping = _render_sequences_panel(
            seq_axes,
            sequences,
            [int(s) for s in top_seq_ids],
            norm_scores.detach().cpu().numpy(),
            seq_offsets.detach().cpu().numpy(),
        )
        return [int(s) for s in top_seq_ids], norm_scores, mapping

    # Initial render
    top_seq_ids, current_norm_scores, token_mapping = render_for_circuit(circuit_idx)
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)

    slider = Slider(slider_ax, "Circuit", 0, C - 1, valinit=circuit_idx, valstep=1)

    ax_to_seq = dict(zip(seq_axes, top_seq_ids, strict=False))

    def on_slider_change(val: float) -> None:
        nonlocal circuit_idx, top_seq_ids, token_mapping, ax_to_seq
        circuit_idx = int(val)
        circuit_im.set_array(circuits[circuit_idx].detach().cpu().numpy())
        circuit_ax.set_title(f"Circuit {circuit_idx + 1}/{C} (L={L}, E={E})")
<<<<<<< HEAD
        top_seq_ids, token_mapping = render_for_circuit(circuit_idx)
=======
        top_seq_ids, current_norm_scores, token_mapping = render_for_circuit(
            circuit_idx
        )
        # Rebuild axis->sequence mapping for hover after re-render
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
        ax_to_seq = dict(zip(seq_axes, top_seq_ids, strict=False))
        overlay_im.set_alpha(0.0)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    def on_motion(event) -> None:
        if event.inaxes not in seq_axes:
            return
        ax = event.inaxes
        seq_id = ax_to_seq.get(ax)
        if seq_id is None or event.xdata is None:
            return
        i = int(event.xdata)
        if i < 0:
            return
        if seq_id >= len(sequences) or i >= len(sequences[seq_id]):
            return
        start = int(seq_offsets[seq_id].item())
        global_token_idx = start + i
        mask = token_topk_mask[global_token_idx].detach().cpu().numpy().astype(float)
        overlay_im.set_array(mask)
        overlay_im.set_alpha(0.35)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)

<<<<<<< HEAD
    fig.suptitle(title or "Activating tokens")
=======
    fig.suptitle("Activating tokens viewer")
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
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
<<<<<<< HEAD
    # Load mask and tokens for alignment + scoring
    token_topk_mask, _topk = load_activations_and_topk(device=device)
    _activated_experts, tokens, _ = load_activations_tokens_and_topk(device=device)
    _ensure_token_alignment(token_topk_mask, tokens)
    circuits = circuits.to(device=device, dtype=th.float32)

    # Compute per-token, per-circuit activations (B, C)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    B, C = int(activations.shape[0]), int(activations.shape[1])

    # Map tokens to sequences and normalize scores per-circuit for coloring
    seq_ids, seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens)
    _validate_seq_mapping(seq_ids, seq_lengths)

    # Build ordered sequence list per circuit by max token criteria
    seq_order_per_circuit: list[list[int]] = []
    token_scores_per_circuit = th.empty(
        C, B, dtype=th.float32, device=activations.device
    )
    for c in range(C):
        token_scores = activations[:, c]
        # Normalize 0..1
=======
    # Also fetch tokens to ensure alignment; if not provided externally, they are loaded within _viz_common
    activated_experts, tokens, _topk = load_activations_tokens_and_topk(device=device)
    # Precompute token->circuit scores, sequence ids/lengths/offsets
    token_topk_mask, _ = load_activations_and_topk(device=device)
    circuits = circuits.to(device=device, dtype=th.float32)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    seq_ids, seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens)

    def selector(c_idx: int) -> tuple[list[int], th.Tensor]:
        token_scores = activations[:, c_idx]
        # Normalize to 0..1 for coloring
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
        min_v = float(token_scores.min().item())
        max_v = float(token_scores.max().item())
        denom = max(max_v - min_v, 1e-6)
        norm_scores = ((token_scores - min_v) / denom).clamp(0, 1)
<<<<<<< HEAD
        token_scores_per_circuit[c] = norm_scores
        top_seq_ids = _gather_top_sequences_by_max(norm_scores, seq_ids, top_n)
        seq_order_per_circuit.append([int(s.item()) for s in top_seq_ids])

    _viz_common(
        circuits,
        tokens,
        seq_order_per_circuit,
        token_scores_per_circuit,
        top_n=top_n,
        device=device,
        title="Max-activating tokens",
    )
=======
        top_seq_ids = _gather_top_sequences_by_max(norm_scores, seq_ids, top_n)
        return [int(s.item()) for s in top_seq_ids], norm_scores

    _viz_common(circuits, tokens, selector, top_n=top_n, device=device)
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)


def viz_mean_activating_tokens(
    circuits: th.Tensor, top_n: int = 10, device: str = "cuda"
) -> None:
    """Visualize top-N sequences by highest mean token activation.

    DRY implementation shared with viz_max_activating_tokens via _viz_common.
    """
<<<<<<< HEAD
    token_topk_mask, _topk = load_activations_and_topk(device=device)
    _activated_experts, tokens, _ = load_activations_tokens_and_topk(device=device)
    _ensure_token_alignment(token_topk_mask, tokens)
    circuits = circuits.to(device=device, dtype=th.float32)

    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    B, C = int(activations.shape[0]), int(activations.shape[1])

    seq_ids, seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens)
    _validate_seq_mapping(seq_ids, seq_lengths)

    seq_order_per_circuit: list[list[int]] = []
    token_scores_per_circuit = th.empty(
        C, B, dtype=th.float32, device=activations.device
    )
    for c in range(C):
        token_scores = activations[:, c]
=======
    activated_experts, tokens, _topk = load_activations_tokens_and_topk(device=device)
    token_topk_mask, _ = load_activations_and_topk(device=device)
    circuits = circuits.to(device=device, dtype=th.float32)
    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuits)
    seq_ids, seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens)

    def selector(c_idx: int) -> tuple[list[int], th.Tensor]:
        token_scores = activations[:, c_idx]
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
        min_v = float(token_scores.min().item())
        max_v = float(token_scores.max().item())
        denom = max(max_v - min_v, 1e-6)
        norm_scores = ((token_scores - min_v) / denom).clamp(0, 1)
<<<<<<< HEAD
        token_scores_per_circuit[c] = norm_scores
        top_seq_ids = _gather_top_sequences_by_mean(
            norm_scores, seq_ids, seq_lengths, top_n
        )
        seq_order_per_circuit.append([int(s.item()) for s in top_seq_ids])

    _viz_common(
        circuits,
        tokens,
        seq_order_per_circuit,
        token_scores_per_circuit,
        top_n=top_n,
        device=device,
        title="Mean-activating tokens",
    )
=======
        top_seq_ids = _gather_top_sequences_by_mean(
            norm_scores, seq_ids, seq_lengths, top_n
        )
        return [int(s.item()) for s in top_seq_ids], norm_scores

    _viz_common(circuits, tokens, selector, top_n=top_n, device=device)
>>>>>>> 36ea0c3 (refactor(viz): remove selection_mode from _viz_common; accept external selector callback and normalized token scores\n\n- _viz_common now renders using a selector(c_idx)->(seq_ids, norm_scores) so selection happens upstream\n- viz_max_activating_tokens and viz_mean_activating_tokens precompute activations and provide selectors\n- keep hover overlay behavior and alignment, clean up imports, satisfy ruff/ty/tests\n\nCo-authored-by: Henry Castillo <hacperu2010@gmail.com>)
