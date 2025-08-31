"""Visualization of expert importances in MoE models."""

import os

import arguably
import matplotlib
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch as th
from tqdm import tqdm

matplotlib.use("WebAgg")  # Use WebAgg backend for interactive plots

from exp import get_experiment_dir
from viz import get_figure_dir

# Constants
READER_COMPONENTS = {"mlp.up_proj", "mlp.gate_proj", "attn.q_proj", "attn.k_proj"}
WRITER_COMPONENTS = {"mlp.down_proj", "attn.o_proj"}

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
    data_path: str | None = None,
    model_name: str | None = None,
    checkpoint_idx: int | None = None,
    initial_base_layer_idx: int = 0,
    initial_base_expert_idx: int = 0,
    normalize_percentile: float = 95.0,
    figure_width: float = 16.0,  # Increased width
    figure_height: float = 14.0,  # Increased height
    experiment_name: str | None = None,
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
        experiment_name: Name of the experiment to use for paths
    """
    # Get experiment directory and figure directory
    experiment_dir = get_experiment_dir(name=experiment_name)
    figure_dir = get_figure_dir(experiment_name)
    expert_importances_viz_dir = os.path.join(figure_dir, "expert_importances")
    os.makedirs(expert_importances_viz_dir, exist_ok=True)

    # If data_path is not provided, use the default path in the experiment directory
    if data_path is None:
        data_path = os.path.join(experiment_dir, "expert_importance", "all.pt")

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}...")
    entries = th.load(data_path)
    if not entries:
        raise ValueError("No entries found in data file")

    # Filter entries by model_name and checkpoint_idx if provided
    filtered_entries = []
    for entry in tqdm(entries, desc="Filtering entries"):
        if model_name is not None and entry["model_name"] != model_name:
            continue
        if checkpoint_idx is not None and entry["checkpoint_idx"] != checkpoint_idx:
            continue
        filtered_entries.append(entry)

    if not filtered_entries:
        raise ValueError(
            f"No entries found for model_name={model_name}, checkpoint_idx={checkpoint_idx}"
        )

    # Extract unique layers and experts
    base_layers = sorted({entry["base_layer_idx"] for entry in filtered_entries})
    derived_layers = sorted({entry["derived_layer_idx"] for entry in filtered_entries})
    layers = sorted(set(base_layers + derived_layers))

    # Get the number of experts from the data
    num_experts = (
        max(
            max(entry.get("base_expert_idx", 0) for entry in filtered_entries),
            max(
                entry.get("derived_expert_idx", 0)
                for entry in filtered_entries
                if "derived_expert_idx" in entry
            ),
        )
        + 1
    )

    # Create lookup dictionary for fast access
    importance_data = {}
    for entry in tqdm(filtered_entries, desc="Creating importance data"):
        # Get the param_type to determine if it's MoE or Attention
        param_type = entry.get("param_type")
        if param_type not in ["moe", "attn"]:
            raise ValueError(
                f'Invalid or missing param_type: {param_type}. Must be "moe" or "attn".'
            )

        base_layer = entry["base_layer_idx"]
        base_expert = entry["base_expert_idx"]
        derived_layer = entry["derived_layer_idx"]
        component = entry["component"]

        # For MoE components, we have derived_expert_idx
        derived_expert = entry["derived_expert_idx"] if param_type == "moe" else None

        key = (base_layer, base_expert, derived_layer, component, derived_expert)
        importance_data[key] = {
            "role": entry["role"],
            "l2": entry["l2"],
        }

    # Compute color normalization
    all_l2_values = [entry["l2"] for entry in filtered_entries]
    if not all_l2_values:
        raise ValueError("No L2 values found in filtered data")

    # Use percentile for normalization to avoid outliers
    max_l2 = float(np.percentile(all_l2_values, normalize_percentile))
    norm = mcolors.Normalize(vmin=0, vmax=max_l2)

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
                -0.5,
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
                    x_pos = (-num_experts * expert_width - middle_spacing / 2) + (
                        derived_expert_idx * expert_width
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
                0.5,
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
        # for rect in all_rectangles.values():
        #     rect.set_facecolor("lightgray")
        #     rect.set_edgecolor("gray")
        #     rect.set_linewidth(0.5)

        # Update colors based on current base layer and expert
        for key, data in tqdm(
            importance_data.items(), desc="Updating colors", leave=False
        ):
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

                    # Highlight the selected expert
                    if (
                        derived_layer == base_layer
                        and base_expert == current_base_expert_idx
                    ):
                        rect.set_edgecolor(SELECTED_EXPERT_BORDER_COLOR)
                        rect.set_linewidth(1.0)
                    else:
                        rect.set_edgecolor("gray")
                        rect.set_linewidth(0.5)

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
