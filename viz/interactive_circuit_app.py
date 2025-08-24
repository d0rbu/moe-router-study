import json
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import torch as th

from exp.activations import (
    load_activations_indices_tokens_and_topk,
)
from viz.circuit_max_activating_examples import (
    _color_for_value,
)

# Constants
CIRCUITS_DIR = "saved_circuits"
DEFAULT_CIRCUIT_PATH = ""  # Will use default paths from _load_circuits_tensor


def ensure_circuits_dir():
    """Ensure the circuits directory exists."""
    os.makedirs(CIRCUITS_DIR, exist_ok=True)


def load_saved_circuit_names() -> list[str]:
    """Load the names of all saved circuits."""
    ensure_circuits_dir()
    return [f for f in os.listdir(CIRCUITS_DIR) if f.endswith(".json")]


def save_circuit(name: str, circuit: np.ndarray):
    """Save a circuit to disk."""
    ensure_circuits_dir()
    if not name.endswith(".json"):
        name = f"{name}.json"

    # Convert to list for JSON serialization
    circuit_list = circuit.tolist()

    with open(os.path.join(CIRCUITS_DIR, name), "w") as f:
        json.dump({"circuit": circuit_list}, f)


def load_circuit(name: str) -> np.ndarray:
    """Load a circuit from disk."""
    with open(os.path.join(CIRCUITS_DIR, name)) as f:
        data = json.load(f)
    return np.array(data["circuit"])


def generate_random_mask(num_layers: int, num_experts: int, k: int) -> np.ndarray:
    """Generate a random mask with k ones per layer."""
    mask = np.zeros((num_layers, num_experts))
    for layer in range(num_layers):
        # Randomly select k experts for this layer
        selected_experts = random.sample(range(num_experts), k)
        mask[layer, selected_experts] = 1.0
    return mask


def compute_max_activating_examples(
    circuit: np.ndarray,
    token_topk_mask: th.Tensor,
    top_k: int,
    device: str = "cpu",
    top_n: int = 10
) -> tuple[th.Tensor, list[int]]:
    """Compute the max activating examples for a given circuit."""
    # Convert circuit to tensor
    circuit_tensor = th.tensor(circuit, device=device, dtype=th.float32).unsqueeze(0)

    # Compute activations

    activations = th.einsum("ble,cle->bc", token_topk_mask.float(), circuit_tensor)

    # Normalize by theoretical max: top_k * num_layers
    L = int(circuit_tensor.shape[-2])
    denom = th.tensor(
        float(top_k * L), device=activations.device, dtype=activations.dtype
    )
    norm_scores = (activations / denom).clamp(0, 1)

    # Get top indices
    top_indices = th.argsort(norm_scores.squeeze(1), descending=True)[:top_n]

    return norm_scores.squeeze(1), top_indices.tolist()


def render_token_sequence(
    tokens: list[str],
    token_scores: th.Tensor,
    selected_token_idx: int | None = None
) -> plt.Figure:
    """Render a sequence of tokens with color highlighting based on scores."""
    # Create a figure for the tokens
    fig, ax = plt.subplots(figsize=(10, 2))

    # Set up the plot
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Draw rectangles for tokens with colors based on token_scores
    for i, tok in enumerate(tokens):
        score = float(token_scores[i])
        r, g, b = _color_for_value(score)

        # Add border for selected token
        edgecolor = "red" if i == selected_token_idx else "white"
        linewidth = 2 if i == selected_token_idx else 1

        rect = plt.Rectangle(
            (i, 0.0),
            1.0,
            1.0,
            facecolor=(r, g, b, 0.9),
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.5,
            0.5,
            tok,
            ha="center",
            va="center",
            fontsize=9,
            color="black" if score < 0.5 else "white",
            clip_on=True,
        )

    return fig


def render_token_activation(
    token_topk_mask: th.Tensor,
    token_idx: int
) -> tuple[plt.Figure, np.ndarray]:
    """Render the activation pattern for a specific token."""
    # Get the token's activation mask
    token_mask = token_topk_mask[token_idx].detach().cpu().numpy()

    # Create a figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Display the activation mask
    im = ax.imshow(token_mask, cmap="Blues", vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_title("Token Activation Pattern")

    # Set ticks
    ax.set_xticks(np.arange(token_mask.shape[1]))
    ax.set_yticks(np.arange(token_mask.shape[0]))

    return fig, token_mask


def render_element_product(token_mask: np.ndarray, circuit: np.ndarray) -> plt.Figure:
    """Render the element-wise product of token activation and circuit."""
    # Compute element-wise product
    product = token_mask * circuit

    # Create a figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Display the product
    im = ax.imshow(product, cmap="Blues", vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_title("Element-wise Product")

    # Set ticks
    ax.set_xticks(np.arange(product.shape[1]))
    ax.set_yticks(np.arange(product.shape[0]))

    return fig


def main():
    st.set_page_config(layout="wide", page_title="Interactive Circuit Visualization")

    # Initialize session state for circuit if not already present
    if "circuit" not in st.session_state:
        # Load data
        try:
            token_topk_mask, _activated_expert_indices, tokens, top_k = (
                load_activations_indices_tokens_and_topk(device="cpu")
            )

            # Get circuit dimensions
            num_layers, num_experts = token_topk_mask.shape[1], token_topk_mask.shape[2]

            # Initialize with zeros
            st.session_state.token_topk_mask = token_topk_mask
            st.session_state.tokens = tokens
            st.session_state.top_k = top_k
            st.session_state.selected_cell = None
            st.session_state.selected_token_idx = None
            st.session_state.computing = False
            st.session_state.norm_scores = None
            st.session_state.top_indices = []
            st.session_state.current_example_idx = 0
            st.session_state.computation_id = 0
        except Exception as e:
            st.error(f"Error loading data: {e!s}")
            st.info("Please make sure you have run the data collection scripts first.")
            # Create dummy data for development
            st.session_state.token_topk_mask = th.zeros((100, 4, 8))  # Dummy data
            st.session_state.tokens = [
                ["token1", "token2", "token3"] for _ in range(10)
            ]
            st.session_state.top_k = 2
            st.session_state.selected_cell = None
            st.session_state.selected_token_idx = None
            st.session_state.computing = False
            st.session_state.norm_scores = None
            st.session_state.top_indices = []
            st.session_state.current_example_idx = 0
            st.session_state.computation_id = 0

    # Page title
    st.title("Interactive Circuit Visualization")

    # Create layout with columns for the main interface
    col1, col2 = st.columns([3, 1])

    with col2:
        # Circuit management section
        st.subheader("Circuit Management")

        # Load circuit dropdown
        saved_circuits = load_saved_circuit_names()
        if saved_circuits:
            selected_circuit = st.selectbox("Select a saved circuit", saved_circuits)

            # Load button
            if st.button("Load Selected Circuit"):
                st.success(f"Loaded circuit: {selected_circuit}")
                # Trigger recomputation
                st.session_state.computation_id += 1
        else:
            st.info("No saved circuits found.")

        # Save circuit section
        st.subheader("Save Current Circuit")
        circuit_name = st.text_input("Circuit name")
        if st.button("Save Circuit"):
            if circuit_name:
                st.success(f"Circuit saved as: {circuit_name}")
            else:
                st.error("Please enter a name for the circuit")

        # Random circuit generation
        st.subheader("Generate Random Circuit")
        k_value = st.slider(
            "K value (experts per layer)",
            1,
            min(st.session_state.token_topk_mask.shape[2], 10),
            st.session_state.top_k,
        )

        if st.button("Generate Random Circuit"):
                num_layers, num_experts, k_value
            )
            st.success("Random circuit generated!")
            # Trigger recomputation
            st.session_state.computation_id += 1

        # Vertical slider for selected cell
        st.subheader("Cell Value Adjustment")
        if st.session_state.selected_cell:
            layer, expert = st.session_state.selected_cell

            new_value = st.slider(
                f"Value for Layer {layer}, Expert {expert}",
                min_value=0.0,
                max_value=1.0,
                value=float(current_value),
                step=0.01,
                key=f"slider_{layer}_{expert}",
            )

            if new_value != current_value:
                # Trigger recomputation
                st.session_state.computation_id += 1
        else:
            st.info("Click a cell in the circuit to adjust its value")

        # Compute button
        if st.button("Compute Max Activating Examples"):
            st.session_state.computing = True
            # Trigger recomputation
            st.session_state.computation_id += 1

    with col1:
        # Circuit visualization
        st.subheader("Interactive Circuit Grid")
        st.write("Click on a cell to select it. Click again to toggle between 0 and 1.")

        # Get circuit dimensions

        # Create a grid of buttons for interactive circuit editing
        for layer in range(num_layers):
            cols = st.columns(num_experts)
            for expert in range(num_experts):
                with cols[expert]:
                    # Create a unique key for each button
                    button_key = f"btn_{layer}_{expert}"

                    # Create a button with the value
                    if st.button(
                        f"{value:.2f}",
                        key=button_key,
                        help=f"Layer {layer}, Expert {expert}",
                        use_container_width=True,
                    ):
                        # Handle button click
                        if st.session_state.selected_cell == (layer, expert):
                            # Toggle value if already selected

                        # Update selected cell
                        st.session_state.selected_cell = (layer, expert)

                        # Trigger recomputation
                        st.session_state.computation_id += 1

                        # Force rerun to update UI
                        st.rerun()

                    # Apply custom styling to the button based on value
                    # Higher values = darker blue
                    blue_intensity = int(255 * value)
                    button_color = (
                        f"rgb({255 - blue_intensity}, {255 - blue_intensity}, 255)"
                    )
                    text_color = "white" if value > 0.5 else "black"

                    # Add border if this cell is selected
                    border_style = (
                        "3px solid red"
                        if st.session_state.selected_cell == (layer, expert)
                        else "1px solid #ddd"
                    )

                    st.markdown(
                        f"""
                        <style>
                        [data-testid="stButton"][aria-describedby="StyledDescription-{button_key}"] button {{
                            background-color: {button_color};
                            color: {text_color};
                            border: {border_style};
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

    # Compute max activating examples if needed
    if st.session_state.computing or st.session_state.computation_id > 0:
        computation_id = st.session_state.computation_id

        with st.spinner("Computing max activating examples..."):
            # Convert circuit to tensor and compute
            norm_scores, top_indices = compute_max_activating_examples(
                st.session_state.token_topk_mask,
                st.session_state.tokens,
                st.session_state.top_k,
            )

            # Check if computation was interrupted
            if computation_id == st.session_state.computation_id:
                st.session_state.norm_scores = norm_scores
                st.session_state.top_indices = top_indices
                st.session_state.computing = False
                st.session_state.current_example_idx = 0 if top_indices else -1

    # Four quadrants display
    if st.session_state.norm_scores is not None and st.session_state.top_indices:
        st.subheader("Max Activating Examples")

        # Example selection
        example_idx = st.selectbox(
            "Select example",
            range(len(st.session_state.top_indices)),
            format_func=lambda i: f"Example {i + 1} (Score: {st.session_state.norm_scores[st.session_state.top_indices[i]]:.4f})",
            index=st.session_state.current_example_idx,
        )

        # Update current example
        st.session_state.current_example_idx = example_idx

        # Get the selected example
        if 0 <= example_idx < len(st.session_state.top_indices):
            token_idx = st.session_state.top_indices[example_idx]

            # Get the sequence for this token
            # Find which sequence this token belongs to
            token_count = 0
            seq_idx = -1
            token_in_seq_idx = -1

            for i, seq in enumerate(st.session_state.tokens):
                if token_count <= token_idx < token_count + len(seq):
                    seq_idx = i
                    token_in_seq_idx = token_idx - token_count
                    break
                token_count += len(seq)

            if seq_idx >= 0:
                # Get the sequence
                sequence = st.session_state.tokens[seq_idx]

                # Create the four quadrants
                q1, q2 = st.columns(2)
                q3, q4 = st.columns(2)

                with q1:
                    st.subheader("Text Display")
                    # Compute token scores for this sequence
                    seq_scores = st.session_state.norm_scores[
                        token_count : token_count + len(sequence)
                    ]

                    # Render the sequence
                    fig = render_token_sequence(sequence, seq_scores, token_in_seq_idx)
                    st.pyplot(fig)

                    # Add token selection functionality
                    st.write("Click on a token to see its activation pattern")
                    token_options = [
                        f"{i}: {token}" for i, token in enumerate(sequence)
                    ]
                    selected_token = st.selectbox(
                        "Select token",
                        range(len(sequence)),
                        format_func=lambda i: token_options[i],
                        index=token_in_seq_idx,
                    )

                    # Update selected token
                    if selected_token != token_in_seq_idx:
                        st.session_state.selected_token_idx = (
                            token_count + selected_token
                        )
                    else:
                        st.session_state.selected_token_idx = token_idx

                with q2:
                    st.subheader("Circuit Visualization")
                    # Display the circuit
                    fig, ax = plt.subplots(figsize=(5, 3))
                    im = ax.imshow(
                    )
                    plt.colorbar(im, ax=ax)
                    ax.set_xlabel("Expert")
                    ax.set_ylabel("Layer")
                    ax.set_title("Circuit Mask")
                    st.pyplot(fig)

                # Get the selected token's global index
                selected_token_global_idx = (
                    st.session_state.selected_token_idx or token_idx
                )

                with q3:
                    st.subheader("Token Activation")
                    # Render the token activation
                    fig, token_mask = render_token_activation(
                        st.session_state.token_topk_mask,
                        selected_token_global_idx,
                    )
                    st.pyplot(fig)

                with q4:
                    st.subheader("Element-wise Product")
                    # Render the element-wise product
                    st.pyplot(fig)


if __name__ == "__main__":
    main()
