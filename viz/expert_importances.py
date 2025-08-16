"""Visualization of expert importances in MoE models."""

import os

import matplotlib

matplotlib.use("WebAgg")  # Use GTK3Agg backend for interactive plots on Pop!_OS
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

# Constants
READER_COMPONENTS = {"mlp.up_proj", "mlp.gate_proj", "attn.q_proj", "attn.k_proj"}
WRITER_COMPONENTS = ["mlp.down_proj", "attn.o_proj"]

# Components that have multiple experts (MoE components)
MOE_COMPONENTS = {"mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"}
# Components that have only one value (Attention components)
ATTN_COMPONENTS = {"attn.q_proj", "attn.k_proj", "attn.o_proj"}

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
    initial_base_layer_idx: int = 0,
    initial_base_expert_idx: int = 0,
    normalize_percentile: float = 95.0,
    figure_width: float = 16.0,  # Increased width
    figure_height: float = 14.0,  # Increased height
) -> None:
    """Visualize expert importances with an interactive plot.

    Args:
        data_path: Path to the expert importance data file
        model_name: Filter by model name (None for no filter)
        checkpoint_idx: Filter by checkpoint index (None for no filter)
        initial_base_layer_idx: Initial base layer index to highlight
        initial_base_expert_idx: Initial base expert index to highlight
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
    base_layers = sorted({e["base_layer_idx"] for e in entries})
    derived_layers = sorted({e["derived_layer_idx"] for e in entries})
    layers = sorted(set(base_layers) | set(derived_layers))
    num_experts = (
        max(
            max(e.get("base_expert_idx", 0) for e in entries),
            max(
                e.get("derived_expert_idx", 0)
                for e in entries
                if "derived_expert_idx" in e
            ),
        )
        + 1
    )

    # Create lookup dictionary for fast access
    importance_data = {}
    for entry in entries:
        # Handle different entry types (MoE vs Attention)
        param_type = entry.get("param_type")
        if param_type not in ["moe", "attn"]:
            raise ValueError(
                f'Invalid or missing param_type: {param_type}. Must be "moe" or "attn".'
            )
        base_layer_idx = entry["base_layer_idx"]
        base_expert_idx = entry["base_expert_idx"]
        component = entry["component"]
        role = entry["role"]
        l2 = entry["l2"]

        if param_type == "moe":
            # MoE components have both base and derived experts
            derived_layer_idx = entry["derived_layer_idx"]
            derived_expert_idx = entry["derived_expert_idx"]
            key = (
                base_layer_idx,
                base_expert_idx,
                derived_layer_idx,
                component,
                derived_expert_idx,
            )
            importance_data[key] = {"role": role, "l2": l2, "param_type": param_type}
        elif param_type == "attn":
            # Attention components only have base expert
            key = (
                base_layer_idx,
                base_expert_idx,
                entry["derived_layer_idx"],
                component,
                None,
            )
            importance_data[key] = {"role": role, "l2": l2, "param_type": param_type}
        else:  # This should never happen due to the check above
            # Error if param_type is not present or not one of the expected values
            raise ValueError(
                f"Invalid param_type: {param_type}. Must be 'moe' or 'attn'."
            )

    # Compute color normalization
    all_l2_values = [entry["l2"] for entry in entries]
    vmin = 0
    vmax = float(np.percentile(all_l2_values, normalize_percentile))

    norm = Normalize(vmin=vmin, vmax=vmax)
    reader_cmap = plt.get_cmap(READER_CMAP)
    writer_cmap = plt.get_cmap(WRITER_CMAP)

    # Create figure and axes with better layout
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, top=0.95)  # Better spacing

    # Calculate layout dimensions
    num_layers = len(layers)
    num_reader_components = len(READER_COMPONENTS)
    num_writer_components = len(WRITER_COMPONENTS)

    # Height of each layer band - increased spacing
    layer_height = 1.5  # Increased from 1.0
    # Width of each expert block - 4x wider cells
    expert_width = 1.5  # Increased from 2.5 to make cells 4x wider
    # Spacing between components
    component_spacing = 1.0  # Increased from 0.5 for better spacing
    # Space in the middle for labels
    middle_spacing = 30.0  # Increased from 2.0 for better middle spacing

    # Total height needed
    total_height = num_layers * layer_height

    # Set axis limits with middle spacing
    max_width = expert_width * (num_experts + component_spacing)
    ax.set_xlim(-max_width - middle_spacing, max_width + middle_spacing)
    ax.set_ylim(-1, total_height + 1)

    # Draw residual stream (vertical line at x=0)
    ax.axvline(x=0, color=RESIDUAL_STREAM_COLOR, linestyle="-", linewidth=2, zorder=1)

    # Create highlight rectangle for selected layer (initially hidden)
    layer_highlight = patches.Rectangle(
        (-max_width - middle_spacing, 0),
        2 * (max_width + middle_spacing),
        layer_height,
        facecolor=HIGHLIGHT_COLOR,
        alpha=0.3,
        zorder=0,
    )
    ax.add_patch(layer_highlight)

    # Dictionary to store all rectangles for fast updates
    all_rectangles = {}

    # Draw all component blocks
    for layer_idx_idx, derived_layer_idx in enumerate(layers):
        y_base = layer_idx_idx * layer_height

        # Draw writer components (left side)
        for comp_idx, component in enumerate(WRITER_COMPONENTS):
            y_offset = y_base + (comp_idx * layer_height / num_writer_components)
            height = layer_height / num_writer_components - 0.1  # Increased height

            # Draw component label
            ax.text(
                -middle_spacing / 2 - 0.5,
                y_offset + height / 2,
                component.split(".")[-1],
                ha="right",
                va="center",
                fontsize=10,  # Increased font size
            )

            # Check if this is a MoE component or Attention component
            if component in MOE_COMPONENTS:
                # MoE component - draw multiple rectangles for each expert
                for derived_expert_idx in range(num_experts):
                    x_pos = (
                        -(derived_expert_idx + 1) * expert_width - middle_spacing / 2
                    )

                    # Create rectangle
                    rect = plt.Rectangle(
                        (x_pos, y_offset),
                        expert_width,
                        height,
                        edgecolor="gray",
                        facecolor="lightgray",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

                    # Store rectangle for later updates
                    rect_key = (derived_layer_idx, component, derived_expert_idx)
                    all_rectangles[rect_key] = rect
            else:
                # Attention component - draw one wide rectangle
                x_pos = -num_experts * expert_width - middle_spacing / 2

                # Create rectangle
                rect = plt.Rectangle(
                    (x_pos, y_offset),
                    num_experts * expert_width,
                    height,
                    edgecolor="gray",
                    facecolor="lightgray",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

                # Store rectangle for later updates - use None for derived_expert_idx
                rect_key = (derived_layer_idx, component, None)
                all_rectangles[rect_key] = rect

        # Draw reader components (right side)
        for comp_idx, component in enumerate(READER_COMPONENTS):
            y_offset = y_base + (comp_idx * layer_height / num_reader_components)
            height = layer_height / num_reader_components - 0.1  # Increased height

            # Draw component label
            ax.text(
                0,  # Position in the middle space
                y_offset + height / 2,
                component.split(".")[-1],
                ha="left",
                va="center",
                fontsize=10,  # Increased font size
            )

            # Check if this is a MoE component or Attention component
            if component in MOE_COMPONENTS:
                # MoE component - draw multiple rectangles for each expert
                for derived_expert_idx in range(num_experts):
                    x_pos = derived_expert_idx * expert_width + middle_spacing / 2

                    # Create rectangle
                    rect = plt.Rectangle(
                        (x_pos, y_offset),
                        expert_width - 0.1,  # Increased width
                        height,
                        edgecolor="gray",
                        facecolor="lightgray",
                        linewidth=0.5,
                        zorder=2,
                    )
                    ax.add_patch(rect)

                    # Store rectangle for later updates
                    rect_key = (derived_layer_idx, component, derived_expert_idx)
                    all_rectangles[rect_key] = rect
            else:
                # Attention component - draw one wide rectangle
                x_pos = middle_spacing / 2

                # Create rectangle
                rect = plt.Rectangle(
                    (x_pos, y_offset),
                    num_experts * expert_width - 0.1,
                    height,
                    edgecolor="gray",
                    facecolor="lightgray",
                    linewidth=0.5,
                    zorder=2,
                )
                ax.add_patch(rect)

                # Store rectangle for later updates - use None for derived_expert_idx
                rect_key = (derived_layer_idx, component, None)
                all_rectangles[rect_key] = rect

        # Add layer index label
        ax.text(
            0,
            y_base + layer_height / 2,
            f"Layer {derived_layer_idx}",
            ha="center",
            va="center",
            fontsize=12,  # Increased font size
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    # Add colorbars with better positioning
    cax_writer = fig.add_axes((0.15, 0.12, 0.3, 0.03))
    cax_reader = fig.add_axes((0.55, 0.12, 0.3, 0.03))

    writer_sm = ScalarMappable(cmap=writer_cmap, norm=norm)
    reader_sm = ScalarMappable(cmap=reader_cmap, norm=norm)

    writer_sm.set_array([])
    reader_sm.set_array([])

    writer_cbar = fig.colorbar(writer_sm, cax=cax_writer, orientation="horizontal")
    reader_cbar = fig.colorbar(reader_sm, cax=cax_reader, orientation="horizontal")

    writer_cbar.set_label("Writer Importance (L2 Norm)")
    reader_cbar.set_label("Reader Importance (L2 Norm)")

    # Add sliders for base layer and base expert selection with better spacing
    ax_layer = fig.add_axes((0.25, 0.06, 0.65, 0.03))
    ax_expert = fig.add_axes((0.25, 0.02, 0.65, 0.03))

    slider_layer = Slider(
        ax=ax_layer,
        label="Base Layer Index",
        valmin=0,
        valmax=len(base_layers) - 1,
        valinit=min(initial_base_layer_idx, len(base_layers) - 1),
        valstep=1,
    )

    slider_expert = Slider(
        ax=ax_expert,
        label="Base Expert Index",
        valmin=0,
        valmax=num_experts - 1,
        valinit=min(initial_base_expert_idx, num_experts - 1),
        valstep=1,
    )

    # Current selection state
    current_base_layer_idx = base_layers[
        min(initial_base_layer_idx, len(base_layers) - 1)
    ]
    current_base_expert_idx = min(initial_base_expert_idx, num_experts - 1)

    # Update function for sliders
    def update_visualization():
        # Reset all rectangles to default color
        for rect in all_rectangles.values():
            rect.set_facecolor("lightgray")
            rect.set_edgecolor("gray")
            rect.set_linewidth(0.5)

        # Update colors based on current base layer and expert
        for key, data in importance_data.items():
            base_layer, base_expert, derived_layer, component, derived_expert = key

            if (
                base_layer == current_base_layer_idx
                and base_expert == current_base_expert_idx
            ):
                role = data["role"]
                l2 = data["l2"]

                # Get the correct rectangle
                if derived_expert is not None:
                    # MoE component
                    rect_key = (derived_layer, component, derived_expert)
                else:
                    # Attention component
                    rect_key = (
                        derived_layer,
                        component,
                        None,
                    )  # Use None for attention components

                if rect_key in all_rectangles:
                    rect = all_rectangles[rect_key]

                    # Set color based on role
                    if role == "reader":
                        color = reader_cmap(norm(l2))
                    else:  # writer
                        color = writer_cmap(norm(l2))

                    rect.set_facecolor(color)

                    # Highlight the rectangle if it's in the base layer
                    if derived_layer == base_layer:
                        rect.set_edgecolor("black")
                        rect.set_linewidth(2.0)

        # Update layer highlight position
        layer_idx_idx = (
            layers.index(current_base_layer_idx)
            if current_base_layer_idx in layers
            else 0
        )
        y_base = layer_idx_idx * layer_height
        layer_highlight.set_y(y_base)

        # Update title
        ax.set_title(
            f"Expert Importances - Base Layer {current_base_layer_idx}, Base Expert {current_base_expert_idx}",
            fontsize=14,
        )

        # Force redraw
        fig.canvas.draw_idle()

    def update_layer(val):
        nonlocal current_base_layer_idx
        # Get new base layer index
        new_base_layer_idx_idx = int(val)
        new_base_layer_idx = base_layers[new_base_layer_idx_idx]
        # Update current base layer
        current_base_layer_idx = new_base_layer_idx
        # Update visualization
        update_visualization()

    def update_expert(val):
        nonlocal current_base_expert_idx
        # Get new base expert index
        new_base_expert_idx = int(val)
        # Update current base expert
        current_base_expert_idx = new_base_expert_idx
        # Update visualization
        update_visualization()

    # Connect sliders to update functions
    slider_layer.on_changed(update_layer)
    slider_expert.on_changed(update_expert)

    # Initial update to set the starting state
    update_visualization()

    # Set initial title and labels
    ax.set_title(
        f"Expert Importances - Base Layer {current_base_layer_idx}, Base Expert {current_base_expert_idx}",
        fontsize=14,
    )
    ax.set_xlabel("Residual Stream", fontsize=12)
    ax.set_ylabel("Layers", fontsize=12)

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

    plt.show()


if __name__ == "__main__":
    arguably.run()
