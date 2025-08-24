import json
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch as th

from exp.activations import (
    load_activations_indices_tokens_and_topk,
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
    circuit: np.ndarray, token_topk_mask: th.Tensor, top_k: int, device: str = "cpu"
) -> tuple[th.Tensor, list[list[str]], list[int]]:
    """Compute the max activating examples for a given circuit."""
    # Convert circuit to tensor
    circuit_tensor = th.tensor(circuit, device=device, dtype=th.float32).unsqueeze(0)

    # Compute activations
    activations = th.einsum(
        "ble,cle->bc", token_topk_mask.float(), circuit_tensor
    )

    # Normalize by theoretical max: top_k * num_layers
    L = int(circuit_tensor.shape[-2])
    denom = th.tensor(
        float(top_k * L), device=activations.device, dtype=activations.dtype
    )
    norm_scores = (activations / denom).clamp(0, 1)

    return norm_scores.squeeze(1), [], []  # Placeholder for tokens and indices


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
            st.session_state.circuit = np.zeros((num_layers, num_experts))
            st.session_state.token_topk_mask = token_topk_mask
            st.session_state.tokens = tokens
            st.session_state.top_k = top_k
            st.session_state.selected_cell = None
            st.session_state.computing = False
            st.session_state.norm_scores = None
            st.session_state.max_activating_examples = []
        except Exception as e:
            st.error(f"Error loading data: {e!s}")
            st.info("Please make sure you have run the data collection scripts first.")
            # Create dummy data for development
            st.session_state.circuit = np.zeros((4, 8))  # 4 layers, 8 experts
            st.session_state.token_topk_mask = th.zeros((100, 4, 8))  # Dummy data
            st.session_state.tokens = []
            st.session_state.top_k = 2
            st.session_state.selected_cell = None
            st.session_state.computing = False
            st.session_state.norm_scores = None
            st.session_state.max_activating_examples = []

    # Page title
    st.title("Interactive Circuit Visualization")

    # Create layout with columns
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
                st.session_state.circuit = load_circuit(selected_circuit)
                st.success(f"Loaded circuit: {selected_circuit}")
        else:
            st.info("No saved circuits found.")

        # Save circuit section
        st.subheader("Save Current Circuit")
        circuit_name = st.text_input("Circuit name")
        if st.button("Save Circuit"):
            if circuit_name:
                save_circuit(circuit_name, st.session_state.circuit)
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
            num_layers, num_experts = st.session_state.circuit.shape
            st.session_state.circuit = generate_random_mask(
                num_layers, num_experts, k_value
            )
            st.success("Random circuit generated!")

        # Vertical slider for selected cell
        st.subheader("Cell Value Adjustment")
        if st.session_state.selected_cell:
            layer, expert = st.session_state.selected_cell
            current_value = st.session_state.circuit[layer, expert]

            new_value = st.slider(
                f"Value for Layer {layer}, Expert {expert}",
                min_value=0.0,
                max_value=1.0,
                value=float(current_value),
                step=0.01,
                key=f"slider_{layer}_{expert}",
            )

            if new_value != current_value:
                st.session_state.circuit[layer, expert] = new_value
        else:
            st.info("Click a cell in the circuit to adjust its value")

    with col1:
        # Circuit visualization
        st.subheader("Circuit Visualization")
        st.write("Click on a cell to select it. Click again to toggle between 0 and 1.")

        # Get circuit dimensions
        num_layers, num_experts = st.session_state.circuit.shape

        # Create a dataframe for the circuit visualization
        circuit_df = pd.DataFrame(
            st.session_state.circuit,
            index=[f"Layer {i}" for i in range(num_layers)],
            columns=[f"Expert {i}" for i in range(num_experts)],
        )

        # Use Streamlit's data editor for interactive editing
        edited_df = st.data_editor(
            circuit_df,
            key="circuit_editor",
            disabled=False,
            hide_index=False,
            column_config={
                col: st.column_config.NumberColumn(
                    col, min_value=0.0, max_value=1.0, step=0.01, format="%.2f"
                )
                for col in circuit_df.columns
            },
            use_container_width=True,
        )

        # Update the circuit if the dataframe was edited
        if not edited_df.equals(circuit_df):
            st.session_state.circuit = edited_df.values

        # Create a grid of buttons for more intuitive interaction
        st.subheader("Interactive Circuit Grid")
        st.write("Click on a cell to select it or toggle its value")

        # Create a grid of buttons
        for layer in range(num_layers):
            cols = st.columns(num_experts)
            for expert in range(num_experts):
                with cols[expert]:
                    value = st.session_state.circuit[layer, expert]
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
                            st.session_state.circuit[layer, expert] = 1.0 - value

                        # Update selected cell
                        st.session_state.selected_cell = (layer, expert)

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

        # Compute button
        if st.button("Compute Max Activating Examples"):
            st.session_state.computing = True
            with st.spinner("Computing max activating examples..."):
                # Convert circuit to tensor and compute
                norm_scores, _, _ = compute_max_activating_examples(
                    st.session_state.circuit,
                    st.session_state.token_topk_mask,
                    st.session_state.top_k,
                )
                st.session_state.norm_scores = norm_scores
                st.session_state.computing = False
            st.success("Computation complete!")

        # Display the top activating examples if available
        if st.session_state.norm_scores is not None:
            st.subheader("Top Activating Examples")
            # Get top 10 indices
            top_indices = th.argsort(st.session_state.norm_scores, descending=True)[:10]

            # Display scores
            for i, idx in enumerate(top_indices):
                score = st.session_state.norm_scores[idx].item()
                st.write(f"Example {i + 1}: Score = {score:.4f}")


if __name__ == "__main__":
    main()
