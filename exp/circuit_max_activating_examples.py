from itertools import count
from typing import Dict, List, Tuple

import arguably
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from exp.activations import load_activations
from exp.get_router_activations import ROUTER_LOGITS_DIR


def get_circuit_activations(
    circuits: th.Tensor,
    device: str = "cuda",
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute circuit activations for every token.

    Args:
        circuits: Tensor of shape (C, L, E) with entries in [0, 1]
        device: device for computation

    Returns:
        activations: Tensor of shape (B, C) where B is the number of tokens
        token_topk_mask: Boolean tensor of shape (B, L, E) representing top-k router activations
    """
    # Load top-k activation mask for tokens: (B, L, E) bool
    token_topk_mask: th.Tensor = load_activations(device=device)

    if token_topk_mask.dtype != th.bool:  # safety
        token_topk_mask = token_topk_mask.bool()

    # Ensure circuits is on the same device and dtype
    circuits = circuits.to(device=device, dtype=th.float32)

    # Compute per-token, per-circuit activation: dot product over (L,E)
    # (B, L, E) bool -> float
    token_mask_f = token_topk_mask.float()
    # (B, L, E) · (C, L, E) -> (B, C)
    activations = th.einsum("ble,cle->bc", token_mask_f, circuits)

    return activations, token_topk_mask


def load_tokenized_sequences() -> List[List[str]]:
    """Load tokenized sequences saved alongside router logits.

    Each file at ROUTER_LOGITS_DIR/{i}.pt contains list[list[str]] under key "tokens".
    Returns a single concatenated list across files, preserving order.
    """
    sequences: List[List[str]] = []
    for file_idx in count():
        path = f"{ROUTER_LOGITS_DIR}/{file_idx}.pt"
        try:
            data = th.load(path)
        except FileNotFoundError:
            break
        # Data was saved as list[list[str]]
        toks: List[List[str]] = data.get("tokens", [])
        sequences.extend(toks)
    if not sequences:
        raise ValueError(
            "No tokenized sequences found. Please run exp.get_router_activations first."
        )
    return sequences


def build_sequence_id_tensor(sequences: List[List[str]]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Build tensor mapping each token to its originating sequence index.

    Assumes the data tensors (router logits/activations) contain all tokens concatenated
    in the same order as the provided `sequences` list.

    Args:
        sequences: list of token lists (list[list[str]])

    Returns:
        seq_ids_per_token: (B,) tensor with the sequence index for each token position
        seq_lengths: (S,) tensor with number of tokens per sequence
        seq_offsets: (S+1,) tensor with exclusive prefix sums (start index of each seq)
    """
    lengths = [len(s) for s in sequences]
    seq_lengths = th.tensor(lengths, dtype=th.long)
    # Compute offsets
    seq_offsets = th.zeros(len(sequences) + 1, dtype=th.long)
    if len(sequences) > 0:
        seq_offsets[1:] = th.cumsum(seq_lengths, dim=0)

    total_tokens = int(seq_offsets[-1].item())
    seq_ids_per_token = th.empty(total_tokens, dtype=th.long)
    for idx in range(len(sequences)):
        start = int(seq_offsets[idx].item())
        end = int(seq_offsets[idx + 1].item())
        if end > start:
            seq_ids_per_token[start:end] = idx

    return seq_ids_per_token, seq_lengths, seq_offsets


def _color_for_value(val: float, vmin: float, vmax: float) -> Tuple[float, float, float]:
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


def _gather_top_sequences_by_max(
    token_scores: th.Tensor,
    seq_ids: th.Tensor,
    top_n: int,
) -> List[int]:
    """Select top sequences by their highest-scoring token.

    Args:
        token_scores: (B,) tensor for a fixed circuit
        seq_ids: (B,) tensor mapping token -> sequence index
        top_n: number of unique sequences to return

    Returns:
        List of sequence indices (length <= top_n) sorted by best token score desc.
    """
    # Sort tokens by score descending
    order = th.argsort(token_scores, descending=True)
    selected: List[int] = []
    seen = set()
    for idx in order.tolist():
        s = int(seq_ids[idx].item())
        if s not in seen:
            seen.add(s)
            selected.append(s)
            if len(selected) >= top_n:
                break
    return selected


def _gather_top_sequences_by_mean(
    token_scores: th.Tensor,
    seq_ids: th.Tensor,
    seq_lengths: th.Tensor,
    top_n: int,
) -> List[int]:
    """Select top sequences by mean token score.

    Args:
        token_scores: (B,) tensor for a fixed circuit
        seq_ids: (B,) tensor mapping token -> sequence index
        seq_lengths: (S,) tensor of token counts per sequence
        top_n: number of unique sequences to return

    Returns:
        List of sequence indices (length == top_n) sorted by mean score desc.
    """
    S = int(seq_lengths.shape[0])
    sums = th.zeros(S, dtype=token_scores.dtype, device=token_scores.device)
    sums = sums.index_add(0, seq_ids.to(sums.device), token_scores)
    means = sums / seq_lengths.to(sums.device)
    order = th.argsort(means, descending=True)
    return order[:top_n].tolist()


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
        # Render one sequence as a line of colored tokens
        tokens = sequences[seq_idx]
        start = int(seq_offsets[seq_idx].item())
        # Plot label (sequence id)
        ax.text(
            0.01,
            y,
            f"S{seq_idx}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            color="black",
        )
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
            # advance x a bit; rough estimate based on token length
            x += 0.012 + 0.006 * max(1, len(tok))
            if x > 0.98:
                break  # line overflow; avoid plotting off-canvas
        y -= line_height
        if y < 0.02:
            break

    return artist_to_token


def _viz_common(
    circuits: th.Tensor,
    selector: str = "max",
    top_n: int = 64,
    device: str = "cuda",
) -> None:
    # Compute activations and load dataset tokens
    activations, token_topk_mask = get_circuit_activations(circuits, device=device)
    sequences = load_tokenized_sequences()
    _ensure_token_alignment(token_topk_mask, sequences)

    seq_ids, seq_lengths, seq_offsets = build_sequence_id_tensor(sequences)

    # Move ids to same device as activations for quick ops
    seq_ids = seq_ids.to(activations.device)
    seq_lengths = seq_lengths.to(activations.device)

    B, C = activations.shape
    L, E = token_topk_mask.shape[1], token_topk_mask.shape[2]

    # Matplotlib setup
    fig: Figure
    ax_left: Axes
    ax_right: Axes
    ax_slider: Axes
    fig = plt.figure(figsize=(16, 10))
    ax_left = fig.add_axes([0.05, 0.15, 0.6, 0.8])
    ax_right = fig.add_axes([0.7, 0.3, 0.28, 0.65])
    ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])

    # Initial circuit index
    current_c = 0

    # Precompute min/max for coloring per circuit
    per_c_vmin = activations.min(dim=0).values.detach().cpu().numpy()
    per_c_vmax = activations.max(dim=0).values.detach().cpu().numpy()

    # Right panel: draw initial circuit heatmap and overlay placeholder
    circuit_np = circuits[current_c].detach().cpu().numpy()
    (im_circuit,) = _render_circuit(ax_right, circuit_np)

    overlay = ax_right.imshow(
        np.zeros((L, E)),
        cmap="Reds",
        alpha=0.0,
        aspect="auto",
        interpolation="nearest",
    )

    # Render left panel sequences
    token_scores_for_c = activations[:, current_c]

    if selector == "max":
        selected_sequences = _gather_top_sequences_by_max(
            token_scores_for_c, seq_ids, top_n
        )
    elif selector == "mean":
        selected_sequences = _gather_top_sequences_by_mean(
            token_scores_for_c, seq_ids, seq_lengths, top_n
        )
    else:
        raise ValueError("selector must be 'max' or 'mean'")

    artist_map = _render_sequences_panel(
        ax_left,
        sequences,
        selected_sequences,
        seq_offsets,
        token_scores_for_c,
        vmin=float(per_c_vmin[current_c]),
        vmax=float(per_c_vmax[current_c]),
    )

    # Slider for circuit index
    slider = Slider(
        ax=ax_slider,
        label="Circuit",
        valmin=0,
        valmax=max(0, C - 1),
        valinit=current_c,
        valstep=1,
    )

    # State for current overlay token
    state = {"token_idx": None}

    def update_overlay_for_token(token_idx: int | None) -> None:
        if token_idx is None:
            overlay.set_alpha(0.0)
            fig.canvas.draw_idle()
            return
        mask = token_topk_mask[token_idx].detach().cpu().numpy().astype(np.float32)
        overlay.set_data(mask)
        overlay.set_alpha(0.35)
        fig.canvas.draw_idle()

    def on_pick(event) -> None:  # type: ignore[no-redef]
        artist = event.artist
        if artist in artist_map:
            token_idx = artist_map[artist]
            state["token_idx"] = token_idx
            update_overlay_for_token(token_idx)

    def on_slider_change(val) -> None:  # type: ignore[no-redef]
        nonlocal current_c, artist_map
        current_c = int(slider.val)
        # Update right panel circuit
        im_circuit.set_array(circuits[current_c].detach().cpu().numpy())
        ax_right.set_title(f"Circuit {current_c} (L x E)")

        # Update left panel selection and colors
        token_scores = activations[:, current_c]
        if selector == "max":
            seqs = _gather_top_sequences_by_max(token_scores, seq_ids, top_n)
        else:
            seqs = _gather_top_sequences_by_mean(
                token_scores, seq_ids, seq_lengths, top_n
            )

        artist_map = _render_sequences_panel(
            ax_left,
            sequences,
            seqs,
            seq_offsets,
            token_scores,
            vmin=float(per_c_vmin[current_c]),
            vmax=float(per_c_vmax[current_c]),
        )
        # Clear overlay unless we re-click
        state["token_idx"] = None
        update_overlay_for_token(None)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    slider.on_changed(on_slider_change)

    plt.show()


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
    _viz_common(circuits, selector="max", top_n=top_n, device=device)


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
    _viz_common(circuits, selector="mean", top_n=top_n, device=device)


if __name__ == "__main__":
    arguably.run()
