"""Visualization of expert importances in MoE models."""

import os

import arguably
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch as th

from exp.expert_importance import EXPERT_IMPORTANCE_DIR
from viz import FIGURE_DIR

EXPERT_IMPORTANCES_VIZ_DIR = os.path.join(FIGURE_DIR, "expert_importances")

# Component groupings
READER_COMPONENTS = {"mlp.up_proj", "mlp.gate_proj", "attn.q_proj", "attn.k_proj"}
WRITER_COMPONENTS = ["mlp.down_proj", "attn.o_proj"]

# Colors
READER_CMAP = "Reds"
WRITER_CMAP = "Blues"
HIGHLIGHT_COLOR = "gold"
SELECTED_EXPERT_BORDER_COLOR = "black"
RESIDUAL_STREAM_COLOR = "gray"


@arguably.command()
def expert_importances(
    data_path: str = os.path.join(EXPERT_IMPORTANCE_DIR, "all.pt"),
    model_name: str | None = None,
    checkpoint_idx: int | None = None,
    initial_layer_idx: int = 0,
    initial_expert_idx: int = 0,
    normalize_percentile: float = 95.0,
    figure_width: float = 12.0,
    figure_height: float = 10.0,
) -> None:
    """Visualize expert importances with an interactive plot.

    Args:
        data_path: Path to the expert importance data file
        model_name: Filter by model name (None for no filter)
        checkpoint_idx: Filter by checkpoint index (None for no filter)
        initial_layer_idx: Initial layer index to highlight
        initial_expert_idx: Initial expert index to highlight
        normalize_percentile: Percentile for color normalization (0-100)
        figure_width: Width of the figure in inches
        figure_height: Height of the figure in inches
    """
    os.makedirs(EXPERT_IMPORTANCES_VIZ_DIR, exist_ok=True)

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert importance data not found at {data_path}")

    entries = th.load(data_path)

    # Filter entries if needed
    if model_name is not None:
        entries = [e for e in entries if e["model_name"] == model_name]
    if checkpoint_idx is not None:
        entries = [e for e in entries if e["checkpoint_idx"] == checkpoint_idx]

    if not entries:
        raise ValueError("No entries found with the specified filters")

    # Extract unique layers and number of experts
    layers = sorted({e["layer_idx"] for e in entries})
    num_experts = max(e["expert_idx"] for e in entries) + 1

    # Create lookup dictionary for fast access
    importance_data = {}
    for entry in entries:
        layer_idx = entry["layer_idx"]
        component = entry["component"]
        expert_idx = entry["expert_idx"]
        role = entry["role"]
        l2 = entry["l2"]

        importance_data[(layer_idx, component, expert_idx)] = {"role": role, "l2": l2}

    # Compute color normalization
    all_l2_values = [entry["l2"] for entry in entries]
    vmin = 0
    vmax = float(np.percentile(all_l2_values, normalize_percentile))

    norm = Normalize(vmin=vmin, vmax=vmax)
    reader_cmap = plt.get_cmap(READER_CMAP)
    writer_cmap = plt.get_cmap(WRITER_CMAP)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    plt.subplots_adjust(bottom=0.2)  # Make room for sliders

    # Calculate layout dimensions
    num_layers = len(layers)
    num_reader_components = len(READER_COMPONENTS)
    num_writer_components = len(WRITER_COMPONENTS)

    # Height of each layer band
    layer_height = 1.0
    # Width of each expert block
    expert_width = 0.8
    # Spacing between components
    component_spacing = 0.2

    # Total height needed
    total_height = num_layers * layer_height

    # Set axis limits
    max_width = max(num_reader_components, num_writer_components) * (
        expert_width * num_experts + component_spacing
    )
    ax.set_xlim(-max_width - 1, max_width + 1)
    ax.set_ylim(-1, total_height + 1)

    # Draw residual stream (vertical line at x=0)
    ax.axvline(x=0, color=RESIDUAL_STREAM_COLOR, linestyle="-", linewidth=2, zorder=1)

    # Create highlight rectangle for selected layer (initially hidden)
    layer_highlight = patches.Rectangle(
        (-max_width - 1, 0),
        2 * (max_width + 1),
        layer_height,
        facecolor=HIGHLIGHT_COLOR,
        alpha=0.3,
        zorder=0,
    )
    ax.add_patch(layer_highlight)

    # Dictionary to store all rectangles for fast updates
    all_rectangles = {}

    # Draw all component blocks
    for layer_idx_idx, layer_idx in enumerate(layers):
        y_base = layer_idx_idx * layer_height

        # Draw writer components (left side)
        for comp_idx, component in enumerate(WRITER_COMPONENTS):
            y_offset = y_base + (comp_idx * layer_height / num_writer_components)
            height = layer_height / num_writer_components - 0.05

            # Draw component label
            ax.text(
                -0.5,
                y_offset + height / 2,
                component.split(".")[-1],
                ha="right",
                va="center",
                fontsize=8,
            )

            for expert_idx in range(num_experts):
                x_pos = -(expert_idx + 1) * expert_width

                # Get importance value
                key = (layer_idx, component, expert_idx)
                if key in importance_data:
                    l2 = importance_data[key]["l2"]
                    color = writer_cmap(norm(l2))
                else:
                    l2 = 0
                    color = "lightgray"

                # Create rectangle
                rect = patches.Rectangle(
                    (x_pos, y_offset),
                    expert_width - 0.05,
                    height,
                    facecolor=color,
                    edgecolor="gray",
                    linewidth=0.5,
                    zorder=2,
                )
                ax.add_patch(rect)
                all_rectangles[key] = rect

        # Draw reader components (right side)
        for comp_idx, component in enumerate(READER_COMPONENTS):
            y_offset = y_base + (comp_idx * layer_height / num_reader_components)
            height = layer_height / num_reader_components - 0.05

            # Draw component label
            ax.text(
                0.5,
                y_offset + height / 2,
                component.split(".")[-1],
                ha="left",
                va="center",
                fontsize=8,
            )

            for expert_idx in range(num_experts):
                x_pos = expert_idx * expert_width

                # Get importance value
                key = (layer_idx, component, expert_idx)
                if key in importance_data:
                    l2 = importance_data[key]["l2"]
                    color = reader_cmap(norm(l2))
                else:
                    l2 = 0
                    color = "lightgray"

                # Create rectangle
                rect = patches.Rectangle(
                    (x_pos, y_offset),
                    expert_width - 0.05,
                    height,
                    facecolor=color,
                    edgecolor="gray",
                    linewidth=0.5,
                    zorder=2,
                )
                ax.add_patch(rect)
                all_rectangles[key] = rect

        # Add layer index label
        ax.text(
            0,
            y_base + layer_height / 2,
            f"Layer {layer_idx}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    # Add colorbars
    cax_writer = fig.add_axes((0.15, 0.08, 0.3, 0.03))
    cax_reader = fig.add_axes((0.55, 0.08, 0.3, 0.03))

    writer_sm = ScalarMappable(cmap=writer_cmap, norm=norm)
    reader_sm = ScalarMappable(cmap=reader_cmap, norm=norm)

    writer_sm.set_array([])
    reader_sm.set_array([])

    writer_cbar = fig.colorbar(writer_sm, cax=cax_writer, orientation="horizontal")
    reader_cbar = fig.colorbar(reader_sm, cax=cax_reader, orientation="horizontal")

    writer_cbar.set_label("Writer Importance (L2 Norm)")
    reader_cbar.set_label("Reader Importance (L2 Norm)")

    # Add sliders for layer and expert selection
    ax_layer = fig.add_axes((0.25, 0.02, 0.65, 0.03))
    ax_expert = fig.add_axes((0.25, 0.05, 0.65, 0.03))

    slider_layer = Slider(
        ax=ax_layer,
        label="Layer Index",
        valmin=0,
        valmax=len(layers) - 1,
        valinit=min(initial_layer_idx, len(layers) - 1),
        valstep=1,
    )

    slider_expert = Slider(
        ax=ax_expert,
        label="Expert Index",
        valmin=0,
        valmax=num_experts - 1,
        valinit=min(initial_expert_idx, num_experts - 1),
        valstep=1,
    )

    # Current selection state
    current_layer_idx = min(initial_layer_idx, len(layers) - 1)
    current_expert_idx = min(initial_expert_idx, num_experts - 1)

    # Update function for sliders
    def update(_=None):
        nonlocal current_layer_idx, current_expert_idx

        # Get new values
        new_layer_idx_idx = int(slider_layer.val)
        new_expert_idx = int(slider_expert.val)
        new_layer_idx = layers[new_layer_idx_idx]

        # Update layer highlight
        y_base = new_layer_idx_idx * layer_height
        layer_highlight.set_y(y_base)

        # Reset borders for previous expert
        for key, rect in all_rectangles.items():
            layer, component, expert = key
            if expert == current_expert_idx:
                rect.set_edgecolor("gray")
                rect.set_linewidth(0.5)

        # Set borders for new expert
        for key, rect in all_rectangles.items():
            layer, component, expert = key
            if expert == new_expert_idx:
                rect.set_edgecolor(SELECTED_EXPERT_BORDER_COLOR)
                rect.set_linewidth(2)

        # Update current selection
        current_layer_idx = new_layer_idx
        current_expert_idx = new_expert_idx

        # Update title
        ax.set_title(
            f"Expert Importances - Layer {new_layer_idx}, Expert {new_expert_idx}"
        )

        fig.canvas.draw_idle()

    # Connect sliders to update function
    slider_layer.on_changed(update)
    slider_expert.on_changed(update)

    # Initial update
    update()

    # Set title and labels
    ax.set_title(
        f"Expert Importances - Layer {layers[current_layer_idx]}, Expert {current_expert_idx}"
    )
    ax.set_xlabel("Residual Stream")
    ax.set_ylabel("Layers")

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend for expert selection
    expert_legend = patches.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="none",
        edgecolor=SELECTED_EXPERT_BORDER_COLOR,
        linewidth=2,
    )
    ax.legend([expert_legend], ["Selected Expert"], loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    arguably.run()
