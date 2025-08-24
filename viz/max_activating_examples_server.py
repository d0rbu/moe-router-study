import os

import streamlit as st
import torch as th

from exp.activations import load_activations_indices_tokens_and_topk

# Constants
CIRCUITS_PATH = "saved_circuits/circuits.pt"


def save_circuits(circuits_dict: dict):
    """Save all circuits to a single file."""
    os.makedirs(os.path.dirname(CIRCUITS_PATH), exist_ok=True)
    th.save(circuits_dict, CIRCUITS_PATH)


def load_circuits() -> dict:
    """Load all circuits from the single file."""
    if not os.path.exists(CIRCUITS_PATH):
        return {"circuits": th.zeros((0, 0, 0)), "names": []}

    return th.load(CIRCUITS_PATH)


def generate_random_mask(num_layers: int, num_experts: int, top_k: int) -> th.Tensor:
    """Generate a random mask with top_k ones per layer using torch operations."""
    # Create a tensor of zeros
    mask = th.zeros(num_layers, num_experts)

    # For each layer, randomly select top_k experts
    for layer in range(num_layers):
        # Generate random indices for each layer
        perm = th.randperm(num_experts)
        top_k_indices = perm[:top_k]

        # Set the selected indices to 1
        mask[layer].scatter_(0, top_k_indices, 1.0)

    return mask


def compute_max_activating_examples(
    circuit: th.Tensor,
    token_topk_mask: th.Tensor,
    top_k: int,
    device: str = "cpu",
    top_n: int = 10,
) -> tuple[th.Tensor, list[int]]:
    """Compute the max activating examples for a given circuit."""
    # Ensure circuit is a tensor with correct shape
    if not isinstance(circuit, th.Tensor):
        circuit = th.tensor(circuit, device=device, dtype=th.float32)

    # Add batch dimension if needed
    circuit_tensor = circuit.unsqueeze(0) if circuit.dim() == 2 else circuit

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
    _tokens: list[str], _token_scores: th.Tensor, _selected_token_idx: int | None = None
) -> object:
    """Render a sequence of tokens with color highlighting based on scores."""
    # Placeholder function - to be implemented later
    return None


def render_token_activation(
    token_topk_mask: th.Tensor, token_idx: int
) -> tuple[object, th.Tensor]:
    """Render the activation pattern for a specific token."""
    # Placeholder function - to be implemented later
    return None, token_topk_mask[token_idx].detach().cpu()


def render_element_product(_token_mask: th.Tensor, _circuit: th.Tensor) -> object:
    """Render the element-wise product of token mask and circuit."""
    # Placeholder function - to be implemented later
    return None


def max_activating_examples_server(
    _circuits_path: str = "",
    top_n: int = 64,
    *_args,
    device: str = "cuda",
    _minibatch_size: int | None = None,
) -> None:
    """Run the max-activating tokens visualization from the command line.

    Args:
        _circuits_path: Path to a .pt file containing a dict with key "circuits" or a raw tensor.
        top_n: Number of sequences to display.
        device: Torch device for computation (e.g., "cuda" or "cpu").
        _minibatch_size: Size of the minibatch for the computation.
    """
    # Load all data once at the top level
    token_topk_mask, _activated_expert_indices, tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )

    # Get dimensions from token_topk_mask
    num_layers, num_experts = token_topk_mask.shape[1], token_topk_mask.shape[2]

    # Initialize Streamlit app
    st.set_page_config(layout="wide", page_title="Max Activating Examples Server")

    # Initialize session state
    if "circuits_dict" not in st.session_state:
        # Try to load existing circuits
        circuits_dict = load_circuits()

        st.session_state.circuits_dict = circuits_dict
        st.session_state.current_circuit_idx = -1  # Always start with no selection
        st.session_state.current_circuit = th.zeros(
            num_layers, num_experts, device=device
        )
        st.session_state.token_topk_mask = token_topk_mask
        st.session_state.tokens = tokens
        st.session_state.top_k = top_k
        st.session_state.selected_cell = None
        st.session_state.norm_scores = None
        st.session_state.top_indices = []

    # Page title
    st.title("Max Activating Examples Server")

    # Circuit management at the top
    st.subheader("Circuit Management")

    # Circuit selection dropdown
    if len(st.session_state.circuits_dict["names"]) > 0:
        circuit_options = st.session_state.circuits_dict["names"]
        selected_idx = st.selectbox(
            "Select a circuit",
            range(len(circuit_options)),
            format_func=lambda i: circuit_options[i],
            index=st.session_state.current_circuit_idx
            if st.session_state.current_circuit_idx >= 0
            else 0,
        )

        if selected_idx != st.session_state.current_circuit_idx:
            st.session_state.current_circuit_idx = selected_idx
            st.session_state.current_circuit = st.session_state.circuits_dict[
                "circuits"
            ][selected_idx].clone()
            # Auto-compute max activating examples when circuit changes
            st.session_state.norm_scores, st.session_state.top_indices = (
                compute_max_activating_examples(
                    st.session_state.current_circuit,
                    token_topk_mask,
                    top_k,
                    device=device,
                    top_n=top_n,
                )
            )
    else:
        st.info("No circuits available. Create a new circuit below.")

    # Save and Generate buttons
    if st.button("Save Current Circuit"):
        # Use a popup for the name input
        circuit_name = st.text_input("Enter circuit name", key="circuit_name_input")
        if circuit_name:
            # Check if name already exists
            if circuit_name in st.session_state.circuits_dict["names"]:
                # Replace existing circuit
                idx = st.session_state.circuits_dict["names"].index(circuit_name)
                st.session_state.circuits_dict["circuits"][idx] = (
                    st.session_state.current_circuit.clone()
                )
                st.session_state.current_circuit_idx = idx
            else:
                # Add new circuit
                if (
                    "circuits" not in st.session_state.circuits_dict
                    or st.session_state.circuits_dict["circuits"].shape[0] == 0
                ):
                    st.session_state.circuits_dict["circuits"] = (
                        st.session_state.current_circuit.unsqueeze(0)
                    )
                    st.session_state.circuits_dict["names"] = [circuit_name]
                else:
                    # Append to existing circuits
                    st.session_state.circuits_dict["circuits"] = th.cat(
                        [
                            st.session_state.circuits_dict["circuits"],
                            st.session_state.current_circuit.unsqueeze(0),
                        ],
                        dim=0,
                    )
                    st.session_state.circuits_dict["names"].append(circuit_name)

                # Update current index to the newly added circuit
                st.session_state.current_circuit_idx = (
                    len(st.session_state.circuits_dict["names"]) - 1
                )

            # Save to disk
            save_circuits(st.session_state.circuits_dict)

            st.success(f"Circuit saved as: {circuit_name}")
            st.rerun()  # Refresh to update the dropdown

    if st.button("Generate Random Circuit"):
        st.session_state.current_circuit = generate_random_mask(
            num_layers, num_experts, top_k
        )

        # Auto-compute max activating examples
        st.session_state.norm_scores, st.session_state.top_indices = (
            compute_max_activating_examples(
                st.session_state.current_circuit,
                token_topk_mask,
                top_k,
                device=device,
                top_n=top_n,
            )
        )

        st.success("Random circuit generated!")
        st.rerun()  # Refresh to update the UI

    # Interactive circuit editor
    st.subheader("Interactive Circuit Grid")
    st.write(
        "Click on a cell to select it. Click again to toggle between 0 and 1. Right-click to deselect."
    )

    # Get circuit dimensions
    num_layers, num_experts = st.session_state.current_circuit.shape

    # Create a grid of buttons for interactive circuit editing
    for layer in range(num_layers):
        cols = st.columns(num_experts)
        for expert in range(num_experts):
            with cols[expert]:
                # Get current value
                value = float(st.session_state.current_circuit[layer, expert])

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
                        st.session_state.current_circuit[layer, expert] = 1.0 - value
                        # Auto-compute max activating examples
                        st.session_state.norm_scores, st.session_state.top_indices = (
                            compute_max_activating_examples(
                                st.session_state.current_circuit,
                                token_topk_mask,
                                top_k,
                                device=device,
                                top_n=top_n,
                            )
                        )

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

    # Cell value adjustment
    if st.session_state.selected_cell:
        layer, expert = st.session_state.selected_cell
        current_value = float(st.session_state.current_circuit[layer, expert])

        st.subheader("Cell Value Adjustment")
        new_value = st.slider(
            f"Value for Layer {layer}, Expert {expert}",
            min_value=0.0,
            max_value=1.0,
            value=current_value,
            step=0.01,
            key=f"slider_{layer}_{expert}",
        )

        if new_value != current_value:
            st.session_state.current_circuit[layer, expert] = new_value
            # Auto-compute max activating examples
            st.session_state.norm_scores, st.session_state.top_indices = (
                compute_max_activating_examples(
                    st.session_state.current_circuit,
                    token_topk_mask,
                    top_k,
                    device=device,
                    top_n=top_n,
                )
            )
            st.rerun()

        if st.button("Deselect Cell"):
            st.session_state.selected_cell = None
            st.rerun()


if __name__ == "__main__":
    max_activating_examples_server()
