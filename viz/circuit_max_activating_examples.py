from itertools import pairwise
import os
from typing import cast

from loguru import logger
import matplotlib
from tqdm import tqdm

matplotlib.use("WebAgg")  # Use GTK3Agg backend for interactive plots on Pop!_OS
import arguably
from matplotlib.axes import Axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch as th

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys


def _load_activations_data(
    device: str = "cuda",
    model_name: str = "switch-base-8",
    dataset_name: str = "c4",
    tokens_per_file: int = 1000,
    context_length: int = 512,
) -> tuple[th.Tensor, th.Tensor, list[list[str]], int]:
    """Load activations data using the proper Activations class.

    Returns:
        Tuple of (token_topk_mask, activated_expert_indices, tokens, top_k)
    """
    import asyncio

    # Load activations using the same pattern as kmeans.py
    activations, activation_dims, gpu_process_group = asyncio.run(
        load_activations_and_init_dist(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=tokens_per_file,
            submodule_names=[ActivationKeys.ROUTER_LOGITS],
            context_length=context_length,
            num_workers=1,
            debug=False,
        )
    )

    # Collect all data
    all_router_logits = []
    all_tokens = []
    top_k = None

    for batch in activations(batch_size=4096):
        router_logits = batch[ActivationKeys.ROUTER_LOGITS]
        all_router_logits.append(router_logits)

        # Extract tokens if available
        if "tokens" in batch:
            all_tokens.extend(batch["tokens"])

        # Extract top_k if available
        if "top_k" in batch and top_k is None:
            top_k = batch["top_k"]

    # Concatenate router logits
    token_topk_mask = th.cat(all_router_logits, dim=0).to(device)

    # Create dummy activated expert indices (same shape as token_topk_mask)
    activated_expert_indices = th.zeros_like(token_topk_mask, dtype=th.long)

    # Use dummy top_k if not found
    if top_k is None:
        top_k = 8  # Default value, adjust based on your model

    return token_topk_mask, activated_expert_indices, all_tokens, top_k


def _load_circuits_tensor(
    circuits_path: str,
    device: str = "cuda",
    token_topk_mask: th.Tensor | None = None,
) -> th.Tensor:
    """Load circuits tensor and reshape to (C, L, E) if needed.

    Accepts a path to a torch-saved object which can be either a dict with a
    "circuits" key or a raw tensor. If the tensor is shape (C, L*E), it will be
    reshaped using (L, E) inferred from activations metadata on disk.
    """
    if not circuits_path:
        # Try common defaults in priority order
        candidates = [
            os.path.join(OUTPUT_DIR, name)
            for name in (
                "optimized_circuits.pt",
                "kmeans_circuits.pt",
                "svd_circuits.pt",
            )
        ]
        circuits_path = next(
            (p for p in candidates if os.path.exists(p)), candidates[0]
        )

    obj = th.load(circuits_path, map_location=device)
    circuits: th.Tensor = (
        obj["circuits"] if isinstance(obj, dict) else cast("th.Tensor", obj)
    )
    circuits = circuits.to(device=device, dtype=th.float32)

    if circuits.ndim == 3:
        return circuits

    if circuits.ndim == 2:
        # Infer (L, E) from activations - use provided data if available
        if token_topk_mask is None:
            token_topk_mask, _indices, _tokens, _top_k = _load_activations_data(
                device=device
            )
        _B, L, E = token_topk_mask.shape
        C = int(circuits.shape[0])
        assert circuits.shape[1] == L * E, (
            f"Circuits second dim {circuits.shape[1]} does not match L*E={L * E}"
        )
        return circuits.view(C, L, E)

    raise ValueError(f"Unsupported circuits tensor ndim: {circuits.ndim}")


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
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Map each token to its sequence index and compute lengths/offsets.

    Returns:
      - seq_ids_per_token: (B,) long tensor mapping token index -> seq index
      - seq_lengths: (S,) long tensor of token counts per sequence
      - seq_offsets: (S+1,) long tensor of prefix sums, start index per sequence
    """
    lengths = [len(seq) for seq in sequences]
    S = len(lengths)
    seq_lengths = th.tensor(lengths, dtype=th.long, device=device)
    # Prefix sums with initial 0 so we have S+1 entries
    seq_offsets = th.empty(S + 1, dtype=th.long, device=device)
    seq_offsets[0] = 0
    if S > 0:
        seq_offsets[1:] = th.cumsum(seq_lengths, dim=0)
    B = int(seq_offsets[-1].item()) if S > 0 else 0
    if B == 0:
        return th.empty(0, dtype=th.long, device=device), seq_lengths, seq_offsets

    seq_ids = th.empty(B, dtype=th.long, device=device)
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
    token_topk_mask: th.Tensor,
    top_n: int = 10,  # noqa: ARG001
    device: str = "cuda",
) -> None:
    """Render interactive view showing a single sequence as plaintext with token highlighting.

    - Left: one sequence rendered inline; each token has a background color based on activation (0..1).
    - Right: circuit grid with slider to switch circuits.
    - Sliders: one for circuit index, one for sequence index.
    - Hover a token to see its actual top-k router activation mask overlaid as transparency.
    """
    circuits = circuits.to(device=device, dtype=th.float32)

    # Validate alignment with provided sequences
    _ensure_token_alignment(token_topk_mask, sequences)

    # Build lengths/offsets
    _seq_ids, _seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)

    # Setup figure: left single sequence, right circuit grid + two sliders
    C = int(circuits.shape[0])
    L, E = int(circuits.shape[-2]), int(circuits.shape[-1])

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 2], height_ratios=[8, 0.9, 0.9])

    seq_ax: Axes = fig.add_subplot(gs[0, 0])
    circuit_ax: Axes = fig.add_subplot(gs[0, 1])
    circuit_slider_ax: Axes = fig.add_subplot(gs[1, :])
    seq_slider_ax: Axes = fig.add_subplot(gs[2, :])

    # Helper: top-K sequences for given circuit (duplicate last if fewer exist)
    TOP_K_SEQS = 16

    def topk_seq_ids_for_circuit(c_idx: int, k: int = TOP_K_SEQS) -> list[int]:
        order = order_per_circuit[c_idx] if c_idx < len(order_per_circuit) else []
        if not order:
            return [0] * k
        trimmed = [int(s) for s in order[:k]]
        if len(trimmed) < k:
            trimmed += [trimmed[-1]] * (k - len(trimmed))
        return trimmed

    # Initial circuit and sequence index (default to top-1 for circuit 0)
    circuit_idx = 0
    allowed_seq_ids = topk_seq_ids_for_circuit(circuit_idx)
    seq_slider_idx = 0  # 0..15 - index into allowed top-K sequences
    seq_id = int(allowed_seq_ids[seq_slider_idx])

    # Circuit image and overlay
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
        cmap="Greys",
        aspect="auto",
        interpolation="nearest",
        alpha=0.0,
    )
    zeros_alpha = np.zeros((L, E), dtype=float)

    # Helper to render a single sequence as HTML-like flow with wrapping and backgrounds
    def render_sequence(
        seq_id_local: int, current_scores: th.Tensor
    ) -> tuple[list[tuple[float, float, float, float, int]], int, float]:
        seq_ax.clear()
        seq_ax.axis("off")
        tokens_in_seq = sequences[seq_id_local]
        n_tok = len(tokens_in_seq)
        global_start = int(seq_offsets[seq_id_local].item())

        # Ensure accurate axis pixel extent
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        fig.canvas.draw()
        bbox = seq_ax.get_window_extent()
        max_width_px = float(bbox.width)

        # Slice the scores once
        if n_tok > 0:
            seq_scores_np = (
                current_scores[global_start : global_start + n_tok]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            seq_scores_np = np.zeros((0,), dtype=float)

        # Token display normalization (replace tokenizer artifact 'Ġ' with real space)
        def _display_token(tok: str) -> str:
            return tok.replace("Ġ", " ")

        # Measure token widths using TextRenderer for more reliable text measurement
        from matplotlib.font_manager import FontProperties

        font_size = 12
        pad_px = 4.0
        line_gap_px = 2.0
        fp = FontProperties(size=font_size)
        dpi = float(fig.dpi)

        def _text_width_px(s: str) -> float:
            if len(s) == 0:
                return 0.0
            try:
                # Use TextPath for accurate text measurement
                from matplotlib.textpath import TextPath

                tp = TextPath((0, 0), s, prop=fp, size=font_size)
                # TextPath extents are in points; convert to pixels
                return float(tp.get_extents().width) * dpi / 72.0
            except Exception as e:
                logger.warning(
                    f"TextPath measurement failed for '{s}': {e}, using fallback"
                )
                # Fallback to character count approximation
                char_width_approx = font_size * 0.6
                return len(s) * char_width_approx

        # Approximate line height in pixels
        line_height_px = (font_size * 1.6) * dpi / 72.0

        # First pass: compute layout boxes
        boxes: list[
            tuple[float, float, float, float, int]
        ] = []  # (x0,y0,x1,y1,local_idx)
        x_px = 0.0
        y_line = 0
        for i, tok in enumerate(tokens_in_seq):
            label = _display_token(tok)
            w_px = _text_width_px(label) + 2 * pad_px
            if x_px + w_px > max_width_px and x_px > 0.0:
                # wrap
                y_line += 1
                x_px = 0.0
            x0 = x_px
            y0 = float(y_line) * (line_height_px + line_gap_px)
            x1 = x0 + w_px
            y1 = y0 + line_height_px
            boxes.append((x0, y0, x1, y1, i))
            x_px = x1

        total_height_px = (float(y_line) + 1.0) * (line_height_px + line_gap_px)
        total_height_px = max(total_height_px, line_height_px)

        # Configure axes in pixel-space
        seq_ax.set_xlim(0, max(1.0, max_width_px))
        seq_ax.set_ylim(0, total_height_px)

        # Draw tokens with backgrounds
        for (x0, y0, x1, y1, i), score in zip(boxes, seq_scores_np, strict=False):
            rect = patches.Rectangle(
                (x0, total_height_px - y1),
                (x1 - x0),
                (y1 - y0),
                facecolor=_make_token_color(float(score)),
                edgecolor="none",
            )
            seq_ax.add_patch(rect)
            seq_ax.text(
                x0 + pad_px,
                total_height_px - (y0 + (y1 - y0) / 2.0),
                _display_token(tokens_in_seq[i]),
                ha="left",
                va="center",
                fontsize=font_size,
                color="black",
                clip_on=True,
            )

        seq_ax.set_title(f"Sequence {seq_id_local}")
        return boxes, global_start, total_height_px

    # Initial render
    current_norm_scores = norm_scores[:, circuit_idx]
    token_boxes, global_seq_start, total_height_px = render_sequence(
        seq_id, current_norm_scores
    )

    # Sliders: circuit (0..C-1), sequence limited to top-16 choices (index 0..15)
    circuit_slider = Slider(
        circuit_slider_ax, "Circuit", 0, C - 1, valinit=circuit_idx, valstep=1
    )
    seq_slider = Slider(
        seq_slider_ax,
        "Sequence (top-16)",
        0,
        TOP_K_SEQS - 1,
        valinit=seq_slider_idx,
        valstep=1,
    )

    # Create a hover outline rectangle on the sequence axis
    hover_outline = patches.Rectangle(
        (0, 0), 0, 0, fill=False, edgecolor="red", linewidth=2.0, zorder=200
    )
    seq_ax.add_patch(hover_outline)
    hover_outline.set_visible(False)

    # Slider callbacks
    def on_circuit_change(val: float) -> None:
        nonlocal \
            circuit_idx, \
            current_norm_scores, \
            token_boxes, \
            global_seq_start, \
            total_height_px, \
            allowed_seq_ids, \
            seq_id

        circuit_idx = int(val)

        circuit_im.set_array(circuits[circuit_idx].detach().cpu().numpy())
        circuit_ax.set_title(f"Circuit {circuit_idx + 1}/{C} (L={L}, E={E})")
        current_norm_scores = norm_scores[:, circuit_idx]
        allowed_seq_ids = topk_seq_ids_for_circuit(circuit_idx)
        # Keep current slider index (0..15), map to actual sequence id (clamped)
        idx = int(np.clip(int(seq_slider.val), 0, len(allowed_seq_ids) - 1))
        seq_id = int(allowed_seq_ids[idx])
        token_boxes, global_seq_start, total_height_px = render_sequence(
            seq_id, current_norm_scores
        )
        hover_outline.set_visible(False)
        overlay_im.set_alpha(zeros_alpha)
        fig.canvas.draw_idle()

    def on_seq_change(val: float) -> None:
        nonlocal seq_slider_idx, seq_id, token_boxes, global_seq_start, total_height_px

        seq_slider_idx = int(val)
        # Map slider index to actual sequence id for current circuit (clamped)
        # Use the already computed allowed_seq_ids instead of recomputing
        local_allowed = allowed_seq_ids
        idx = int(np.clip(seq_slider_idx, 0, len(local_allowed) - 1))
        seq_id = int(local_allowed[idx])

        token_boxes, global_seq_start, total_height_px = render_sequence(
            seq_id, current_norm_scores
        )
        hover_outline.set_visible(False)
        overlay_im.set_alpha(zeros_alpha)
        fig.canvas.draw_idle()

    circuit_slider.on_changed(on_circuit_change)
    seq_slider.on_changed(on_seq_change)

    # Helper to clear hover state (outline + overlay) safely
    def _clear_hover() -> None:
        hover_outline.set_visible(False)
        overlay_im.set_alpha(zeros_alpha)
        fig.canvas.draw_idle()

    # Hover handling: update overlay based on token under cursor
    def on_motion(event) -> None:
        if event.inaxes is not seq_ax or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        # Find token by box hit-test (pixel-space in data coords)
        hit_idx = None
        ymin, ymax = seq_ax.get_ylim()
        total_h = max(ymin, ymax)
        y_inv = total_h - y
        hit_box = None
        for x0, y0, x1, y1, i in token_boxes:
            if (x0 <= x <= x1) and (y0 <= y_inv <= y1):
                hit_idx = i
                hit_box = (x0, y0, x1, y1)
                break
        if hit_idx is None:
            _clear_hover()
            return
        global_token_idx = global_seq_start + int(hit_idx)
        mask = token_topk_mask[global_token_idx].detach().cpu().numpy().astype(float)
        # Show only activated experts by using the mask as per-pixel alpha
        overlay_im.set_array(np.zeros_like(mask, dtype=float))
        overlay_im.set_alpha(np.clip(1.0 - mask, 0.0, 1.0))

        # Update hover outline location/size
        if hit_box is not None:
            x0, y0, x1, y1 = hit_box
            hover_outline.set_xy((x0, total_h - y1))
            hover_outline.set_width(x1 - x0)
            hover_outline.set_height(y1 - y0)
            hover_outline.set_visible(True)

        fig.canvas.draw_idle()

    # Clear hover when leaving the axes/figure
    def on_axes_leave(event) -> None:
        if event.inaxes is seq_ax:
            _clear_hover()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("axes_leave_event", on_axes_leave)

    fig.suptitle("Activating tokens viewer")
    # Avoid tight_layout here since it can alter axes size after we compute pixel-based layout
    # plt.tight_layout()
    plt.show()


def viz_max_activating_tokens(
    circuits: th.Tensor,
    token_topk_mask: th.Tensor,
    tokens: list[list[str]],
    top_k: int,
    top_n: int = 10,
    device: str = "cuda",
    minibatch_size: int | None = None,
) -> None:
    """Visualize top-N sequences by containing highest-activating tokens.

    - Left: tokens highlighted by activation (0..1) for current circuit.
    - Right: circuit grid with slider to switch circuits.
    - Hover a token to see its actual top-k router activation mask overlaid as transparency.
    """
    _ensure_token_alignment(token_topk_mask, tokens)
    circuits = circuits.to(device=device, dtype=th.float32)

    batch_size, _num_layers, _num_experts = token_topk_mask.shape

    if minibatch_size is None:
        minibatch_size = batch_size
    else:
        assert minibatch_size > 0 and minibatch_size <= batch_size, (
            "Batch size must be > 0 and <= batch_size"
        )

        if (num_leftover_samples := batch_size % minibatch_size) > 0:
            logger.warning(
                f"Batch size {batch_size} is not divisible by minibatch size {minibatch_size}, {num_leftover_samples} samples will be discarded"
            )
            token_topk_mask = token_topk_mask[:-num_leftover_samples]
            batch_size -= num_leftover_samples

    num_circuits = int(circuits.shape[0])

    # Compute per-token, per-circuit activations (B, C)
    activations = th.empty(batch_size, num_circuits, device=device)
    for start_idx in tqdm(
        range(0, batch_size, minibatch_size),
        desc="Computing activations",
        leave=False,
        total=batch_size // minibatch_size,
    ):
        stop_idx = min(start_idx + minibatch_size, batch_size)
        activations[start_idx:stop_idx] = th.einsum(
            "ble,cle->bc", token_topk_mask[start_idx:stop_idx].float(), circuits
        )

    # Map tokens to sequences
    seq_ids, _seq_lengths, _seq_offsets = build_sequence_id_tensor(
        tokens, device=device
    )

    # Normalize by theoretical max: top_k * num_layers
    L = int(circuits.shape[-2])
    denom = th.tensor(
        float(top_k * L), device=activations.device, dtype=activations.dtype
    )
    norm_scores = (activations / denom).clamp(0, 1)

    # Build ordered sequence list per circuit by max token criteria
    order_per_circuit: list[list[int]] = []
    for circuit_idx in tqdm(
        range(num_circuits),
        desc="Building ordered sequence list",
        leave=False,
        total=num_circuits,
    ):
        order = _gather_top_sequences_by_max(
            norm_scores[:, circuit_idx], seq_ids, top_n=10**9
        )
        order_per_circuit.append([int(s.item()) for s in order])

    _viz_render_precomputed(
        circuits,
        tokens,
        norm_scores,
        order_per_circuit,
        token_topk_mask,
        top_n=top_n,
        device=device,
    )


def viz_mean_activating_tokens(
    circuits: th.Tensor,
    token_topk_mask: th.Tensor,
    tokens: list[list[str]],
    top_k: int,
    top_n: int = 10,
    device: str = "cuda",
    minibatch_size: int | None = None,
) -> None:
    """Visualize top-N sequences by highest mean token activation.

    Same as viz_max_activating_tokens, but selecting sequences by mean token score.
    """
    _ensure_token_alignment(token_topk_mask, tokens)
    circuits = circuits.to(device=device, dtype=th.float32)

    batch_size, _num_layers, _num_experts = token_topk_mask.shape

    if minibatch_size is None:
        minibatch_size = batch_size
    else:
        assert minibatch_size > 0 and minibatch_size <= batch_size, (
            "Batch size must be > 0 and <= batch_size"
        )

        if (num_leftover_samples := batch_size % minibatch_size) > 0:
            logger.warning(
                f"Batch size {batch_size} is not divisible by minibatch size {minibatch_size}, {num_leftover_samples} samples will be discarded"
            )
            token_topk_mask = token_topk_mask[:-num_leftover_samples]
            batch_size -= num_leftover_samples

    num_circuits = int(circuits.shape[0])

    # Compute per-token, per-circuit activations (B, C)
    activations = th.empty(batch_size, num_circuits, device=device)
    for start_idx in tqdm(
        range(0, batch_size, minibatch_size),
        desc="Computing activations",
        leave=False,
        total=batch_size // minibatch_size,
    ):
        stop_idx = min(start_idx + minibatch_size, batch_size)
        activations[start_idx:stop_idx] = th.einsum(
            "ble,cle->bc", token_topk_mask[start_idx:stop_idx].float(), circuits
        )

    seq_ids, seq_lengths, _seq_offsets = build_sequence_id_tensor(tokens, device=device)

    # Normalize by theoretical max: top_k * num_layers
    L = int(circuits.shape[-2])
    denom = th.tensor(
        float(top_k * L), device=activations.device, dtype=activations.dtype
    )
    norm_scores = (activations / denom).clamp(0, 1)

    # Order sequences per circuit using mean per sequence
    order_per_circuit: list[list[int]] = []
    for circuit_idx in tqdm(
        range(num_circuits),
        desc="Building ordered sequence list",
        leave=False,
        total=num_circuits,
    ):
        order = _gather_top_sequences_by_mean(
            norm_scores[:, circuit_idx], seq_ids, seq_lengths, top_n=10**9
        )
        order_per_circuit.append([int(s.item()) for s in order])

    _viz_render_precomputed(
        circuits,
        tokens,
        norm_scores,
        order_per_circuit,
        token_topk_mask,
        top_n=top_n,
        device=device,
    )


@arguably.command()
def viz_max_cli(
    circuits_path: str = "",
    top_n: int = 10,
    *_args,
    device: str = "cuda",
    minibatch_size: int | None = None,
) -> None:
    """Run the max-activating tokens visualization from the command line.

    Args:
        circuits_path: Path to a .pt file containing a dict with key "circuits" or a raw tensor.
        top_n: Number of sequences to display.
        device: Torch device for computation (e.g., "cuda" or "cpu").
    """
    # Load all data once at the top level
    token_topk_mask, _activated_expert_indices, tokens, top_k = _load_activations_data(
        device=device
    )
    circuits = _load_circuits_tensor(
        circuits_path, device=device, token_topk_mask=token_topk_mask
    )
    viz_max_activating_tokens(
        circuits,
        token_topk_mask,
        tokens,
        top_k,
        top_n=top_n,
        device=device,
        minibatch_size=minibatch_size,
    )


@arguably.command()
def viz_mean_cli(
    circuits_path: str = "",
    top_n: int = 10,
    *_args,
    device: str = "cuda",
    minibatch_size: int | None = None,
) -> None:
    """Run the mean-activating tokens visualization from the command line.

    Args:
        circuits_path: Path to a .pt file containing a dict with key "circuits" or a raw tensor.
        top_n: Number of sequences to display.
        device: Torch device for computation (e.g., "cuda" or "cpu").
    """
    # Load all data once at the top level
    token_topk_mask, _activated_expert_indices, tokens, top_k = _load_activations_data(
        device=device
    )
    circuits = _load_circuits_tensor(
        circuits_path, device=device, token_topk_mask=token_topk_mask
    )
    viz_mean_activating_tokens(
        circuits,
        token_topk_mask,
        tokens,
        top_k,
        top_n=top_n,
        device=device,
        minibatch_size=minibatch_size,
    )


if __name__ == "__main__":
    arguably.run()
