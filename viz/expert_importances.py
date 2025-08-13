from dataclasses import dataclass
import os
from typing import Any

import arguably
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch as th

from exp.expert_importance import EXPERT_IMPORTANCE_DIR
from viz import FIGURE_DIR


# Layout configuration
@dataclass
class ComponentInfo:
    """Information about a component in the visualization."""

    key: str
    label: str


# Writers (left side of visualization)
WRITER_COMPONENTS = {
    ComponentInfo("mlp.down_proj", "Down (writer)"),
    ComponentInfo("attn.o_proj", "Attn O (writer)"),
}

# Readers (right side of visualization)
READER_COMPONENTS = {
    ComponentInfo("attn.q_proj", "Attn Q (reader)"),
    ComponentInfo("attn.k_proj", "Attn K (reader)"),
    ComponentInfo("mlp.up_proj", "Up (reader)"),
    ComponentInfo("mlp.gate_proj", "Gate (reader)"),
}

ALL_COMPONENTS = WRITER_COMPONENTS | READER_COMPONENTS

EXPERT_OWNED = {"mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"}


@dataclass
class ImportanceData:
    # maps: layer_idx -> component -> tensor of shape (num_experts,) with l2 values for that layer/component/each expert
    layer_component_l2: dict[int, dict[str, th.Tensor]]
    # maps: layer_idx -> num_experts for that layer
    layer_num_experts: dict[int, int]
    # global per-component max for normalization
    component_max_l2: dict[str, float]
    # sorted list of layer indices for consistent row order
    layers_sorted: list[int]


def load_importance_data(path: str) -> ImportanceData:
    file_path = os.path.join(path, "all.pt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Expert importance data not found: {file_path}. Run exp/expert_importance.py first."
        )

    entries: list[dict[str, Any]] = th.load(file_path)

    layer_component_l2: dict[int, dict[str, list[float]]] = {}
    layer_num_experts: dict[int, int] = {}
    component_max_l2: dict[str, float] = {}

    # initialize dicts for all components we expect
    for e in entries:
        comp = str(e["component"])  # explicit str for type-checkers
        component_max_l2[comp] = 0.0

    # aggregate per layer/component
    for e in entries:
        layer = int(e["layer_idx"])  # type: ignore[call-arg]
        comp = str(e["component"])  # type: ignore[call-arg]
        expert_idx = int(e["expert_idx"])  # type: ignore[call-arg]
        l2 = float(e["l2"])  # type: ignore[call-arg]
        num_experts = int(e["num_experts"])  # type: ignore[call-arg]

        if layer not in layer_component_l2:
            layer_component_l2[layer] = {
                comp_info.key: [] for comp_info in ALL_COMPONENTS
            }
        # ensure lists can be extended up to expert_idx
        comp_list = layer_component_l2[layer].setdefault(comp, [])
        if len(comp_list) <= expert_idx:
            comp_list.extend([0.0] * (expert_idx + 1 - len(comp_list)))
        comp_list[expert_idx] = l2

        layer_num_experts[layer] = num_experts
        component_max_l2[comp] = max(component_max_l2.get(comp, 0.0), l2)

    # convert lists to tensors for convenient indexing
    layer_component_l2_tensor: dict[int, dict[str, th.Tensor]] = {}
    for layer, comp_map in layer_component_l2.items():
        layer_component_l2_tensor[layer] = {
            comp: th.tensor(vals, dtype=th.float32) for comp, vals in comp_map.items()
        }

    layers_sorted = sorted(layer_component_l2_tensor.keys())

    return ImportanceData(
        layer_component_l2=layer_component_l2_tensor,
        layer_num_experts=layer_num_experts,
        component_max_l2=component_max_l2,
        layers_sorted=layers_sorted,
    )


def _norm_intensity(l2: float, max_l2: float, eps: float = 1e-8) -> float:
    if max_l2 <= eps:
        return 0.0
    # clip to [0,1]
    v = max(0.0, min(1.0, l2 / max_l2))
    return float(v)


def _color_for(role: str, intensity: float) -> tuple[float, float, float, float]:
    # blue for writers, red for readers
    if role == "writer":
        r, g, b = (0.2, 0.4, 0.9)
    else:
        r, g, b = (0.9, 0.2, 0.2)
    # use intensity as alpha; keep base color constant
    return (r, g, b, max(0.05, intensity))


@arguably.command()
def expert_importances() -> None:
    """Interactive visualization for expert importances.

    - Rows are router layers
    - Left columns are writers (Down, Attn O), right columns are readers (Attn Q, Attn K, Up, Gate)
    - For the selected (layer, expert) via sliders, the selected row is colored with intensity proportional to expert-importance L2 per component.
    - Residual stream dot at the center column is gold for the selected layer.
    - Expert-owned weights (Up/Gate/Down) have a thicker border on the selected layer.
    """

    os.makedirs(FIGURE_DIR, exist_ok=True)

    data = load_importance_data(EXPERT_IMPORTANCE_DIR)

    # Figure layout
    num_rows = len(data.layers_sorted)
    # columns: writers (2) + center residual (1) + readers (4)
    writer_info_list = sorted(WRITER_COMPONENTS, key=lambda x: x.key)
    reader_info_list = sorted(READER_COMPONENTS, key=lambda x: x.key)

    col_labels = (
        [comp_info.label for comp_info in writer_info_list]
        + ["Residual"]
        + [comp_info.label for comp_info in reader_info_list]
    )
    col_components = (
        [comp_info.key for comp_info in writer_info_list]
        + [None]
        + [comp_info.key for comp_info in reader_info_list]
    )
    num_cols = len(col_labels)

    fig_h = max(4.0, 0.6 * num_rows + 2.5)
    fig_w = 14.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)

    # axis styling
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xticks([i + 0.5 for i in range(num_cols)])
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks([r + 0.5 for r in range(num_rows)])
    ax.set_yticklabels([f"Layer {layer_idx}" for layer_idx in data.layers_sorted])
    ax.invert_yaxis()  # top layer at the top row
    ax.set_title("Expert Importances (blue=writers, red=readers)")

    # grid patches
    cell_height = 0.8
    cell_width = 0.8
    y_offset = 0.1
    x_offset = 0.1

    # build mapping from (row_idx, col_idx) -> patch
    rects: dict[tuple[int, int], Rectangle] = {}
    centers: dict[int, Circle] = {}

    # Draw background cells and center residual dots
    for row_idx, _layer in enumerate(data.layers_sorted):
        for col_idx in range(num_cols):
            # skip center col for rectangles (we'll draw a dot)
            if col_components[col_idx] is None:
                # residual stream dot
                c = Circle(
                    (col_idx + 0.5, row_idx + 0.5),
                    radius=0.12,
                    facecolor=(0.7, 0.7, 0.7, 1.0),
                    edgecolor="black",
                    linewidth=1.0,
                )
                ax.add_patch(c)
                centers[row_idx] = c
                continue

            rect = Rectangle(
                (col_idx + x_offset, row_idx + y_offset),
                width=cell_width,
                height=cell_height,
                facecolor=(0.9, 0.9, 0.9, 1.0),
                edgecolor="black",
                linewidth=1.0,
            )
            ax.add_patch(rect)
            rects[(row_idx, col_idx)] = rect

    # Slider axes
    ax_layer = plt.axes((0.10, 0.08, 0.35, 0.04))  # use tuple for type-checker
    ax_expert = plt.axes((0.55, 0.08, 0.35, 0.04))  # use tuple for type-checker

    # Slider ranges
    layer_slider = Slider(
        ax=ax_layer,
        label="Layer",
        valmin=0,
        valmax=max(data.layers_sorted) if data.layers_sorted else 0,
        valinit=data.layers_sorted[0] if data.layers_sorted else 0,
        valstep=1,
    )
    # initial expert range based on first layer
    init_layer = int(layer_slider.val)
    init_num_experts = data.layer_num_experts.get(init_layer, 1)
    expert_slider = Slider(
        ax=ax_expert,
        label="Expert",
        valmin=0,
        valmax=max(0, init_num_experts - 1),
        valinit=0,
        valstep=1,
    )

    # Function to refresh visualization for chosen (layer, expert)
    def refresh() -> None:
        sel_layer = int(layer_slider.val)
        # clamp expert to layer bounds
        sel_num_experts = data.layer_num_experts.get(sel_layer, 1)
        if expert_slider.val > sel_num_experts - 1:
            expert_slider.set_val(max(0, sel_num_experts - 1))
        sel_expert = int(expert_slider.val)

        # Update each row
        for row_idx, layer in enumerate(data.layers_sorted):
            is_selected_row = layer == sel_layer

            # residual center
            centers[row_idx].set_facecolor(
                (1.0, 0.84, 0.0, 1.0) if is_selected_row else (0.7, 0.7, 0.7, 1.0)
            )

            for col_idx, comp in enumerate(col_components):
                if comp is None:
                    # center column handled above
                    continue

                rect = rects[(row_idx, col_idx)]

                # Default visuals for non-selected rows
                if not is_selected_row:
                    rect.set_facecolor((0.92, 0.92, 0.92, 1.0))
                    rect.set_linewidth(1.0)
                    rect.set_edgecolor("black")
                    continue

                # Selected layer: set intensity color based on role and l2
                # Find role
                role = (
                    "writer"
                    if comp in {comp_info.key for comp_info in WRITER_COMPONENTS}
                    else "reader"
                )

                # get l2 for this layer/comp/expert
                l2_tensor = data.layer_component_l2[layer][comp]
                # bounds check for safety
                if sel_expert < 0 or sel_expert >= l2_tensor.numel():
                    intensity = 0.0
                else:
                    max_l2 = data.component_max_l2.get(comp, 0.0)
                    intensity = _norm_intensity(
                        float(l2_tensor[sel_expert].item()), max_l2
                    )

                rect.set_facecolor(_color_for(role, intensity))

                # Expert-owned highlight
                if comp in EXPERT_OWNED:
                    rect.set_linewidth(3.0)
                    rect.set_edgecolor((1.0, 0.75, 0.0, 1.0))  # orange/gold border
                else:
                    rect.set_linewidth(1.0)
                    rect.set_edgecolor("black")

        fig.canvas.draw_idle()

    # on-change callbacks
    def on_layer_change(val: float) -> None:
        # update expert slider max for this layer
        sel_layer = int(val)
        num_ex = data.layer_num_experts.get(sel_layer, 1)
        # Matplotlib Slider doesn't support dynamic valmax cleanly; emulate by clamping in refresh
        # We still update the displayed max in the label by recreating text
        expert_slider.valmax = max(0, num_ex - 1)
        expert_slider.ax.set_title(
            f"Expert (0..{max(0, num_ex - 1)})", fontsize=9, pad=12
        )
        refresh()

    def on_expert_change(_val: float) -> None:
        refresh()

    # Initialize expert slider title to reflect range
    expert_slider.ax.set_title(
        f"Expert (0..{max(0, init_num_experts - 1)})", fontsize=9, pad=12
    )

    layer_slider.on_changed(on_layer_change)
    expert_slider.on_changed(on_expert_change)

    # Initial draw
    refresh()

    plt.show()


if __name__ == "__main__":
    arguably.run()
