import os

import streamlit as st
import torch as th

from exp.activations import (
    load_activations_indices_tokens_and_topk,
)
from viz.circuit_max_activating_examples import (
    _color_for_value,
    _load_circuits_tensor,
)

# Constants
CIRCUITS_PATH = "saved_circuits/circuits.pt"


def save_circuits(circuits_dict: dict):
    """Save all circuits to a single file."""
    os.makedirs(os.path.dirname(CIRCUITS_PATH), exist_ok=True)
    th.save(circuits_dict, CIRCUITS_PATH)


def load_circuits() -> dict:
    """Load all circuits from the single file."""
    if not os.path.exists(CIRCUITS_PATH):
        # Initialize with empty dictionary if file doesn't exist
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
    tokens: list[str], token_scores: th.Tensor, selected_token_idx: int | None = None
) -> object:
    """Render a sequence of tokens with color highlighting based on scores."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

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

        rect = patches.Rectangle(
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
    token_topk_mask: th.Tensor, token_idx: int
) -> tuple[object, th.Tensor]:
    """Render the activation pattern for a specific token."""
    import matplotlib.pyplot as plt

    # Get the token's activation mask
    token_mask = token_topk_mask[token_idx].detach().cpu()

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
    ax.set_xticks(range(token_mask.shape[1]))
    ax.set_yticks(range(token_mask.shape[0]))

    return fig, token_mask


def render_element_product(token_mask: th.Tensor, circuit: th.Tensor) -> object:
    """Render the element-wise product of token activation and circuit."""
    import matplotlib.pyplot as plt

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
    ax.set_xticks(range(product.shape[1]))
    ax.set_yticks(range(product.shape[0]))

    return fig


def max_activating_examples_server(
    circuits_path: str = "",
    top_n: int = 64,
    *_args,
    device: str = "cuda",
    _minibatch_size: int | None = None,
) -> None:
    """Run the max-activating tokens visualization from the command line.

    Args:
        circuits_path: Path to a .pt file containing a dict with key "circuits" or a raw tensor.
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

    # Load circuits
    circuits = _load_circuits_tensor(
        circuits_path, device=device, token_topk_mask=token_topk_mask
    )

    # Initialize Streamlit app
    st.set_page_config(layout="wide", page_title="Max Activating Examples Server")

    # Initialize session state
    if "circuits_dict" not in st.session_state:
        # Try to load existing circuits
        try:
            circuits_dict = load_circuits()
        except Exception:
            # Initialize with empty dictionary if loading fails
            circuits_dict = {
                "circuits": th.zeros((0, num_layers, num_experts), device=device),
                "names": [],
            }

        # Add the loaded circuit if available and not already in circuits_dict
        if (
            circuits is not None
            and len(circuits) > 0
            and (
                "circuits" not in circuits_dict
                or circuits_dict["circuits"].shape[0] == 0
            )
        ):
            circuits_dict["circuits"] = circuits
            circuits_dict["names"] = [f"Circuit {i + 1}" for i in range(len(circuits))]

        st.session_state.circuits_dict = circuits_dict
        st.session_state.current_circuit_idx = (
            0 if len(circuits_dict["names"]) > 0 else -1
        )
        st.session_state.token_topk_mask = token_topk_mask
        st.session_state.tokens = tokens
        st.session_state.top_k = top_k
        st.session_state.selected_cell = None
        st.session_state.selected_token_idx = None
        st.session_state.norm_scores = None
        st.session_state.top_indices = []
        st.session_state.current_example_idx = 0

    # Page title
    st.title("Interactive Circuit Visualization")

    # Create layout with columns for the main interface
    col1, col2 = st.columns([3, 1])

    with col2:
        # Circuit management section
        st.subheader("Circuit Management")

        # Load circuit dropdown
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
                # Auto-compute max activating examples when circuit changes
                st.session_state.norm_scores, st.session_state.top_indices = (
                    compute_max_activating_examples(
                        st.session_state.circuits_dict["circuits"][selected_idx],
                        token_topk_mask,
                        top_k,
                        device=device,
                        top_n=top_n,
                    )
                )
        else:
            st.info("No circuits available. Create a new circuit below.")
            st.session_state.current_circuit_idx = -1

        # Save circuit section
        st.subheader("Save Current Circuit")
        circuit_name = st.text_input("Circuit name")
        if st.button("Save Circuit"):
            if circuit_name:
                # Get current circuit
                if st.session_state.current_circuit_idx >= 0:
                    current_circuit = st.session_state.circuits_dict["circuits"][
                        st.session_state.current_circuit_idx
                    ]
                else:
                    # Create a new circuit if none is selected
                    current_circuit = th.zeros(num_layers, num_experts, device=device)

                # Add to circuits dictionary
                if (
                    "circuits" not in st.session_state.circuits_dict
                    or st.session_state.circuits_dict["circuits"].shape[0] == 0
                ):
                    st.session_state.circuits_dict["circuits"] = (
                        current_circuit.unsqueeze(0)
                    )
                    st.session_state.circuits_dict["names"] = [circuit_name]
                else:
                    # Append to existing circuits
                    st.session_state.circuits_dict["circuits"] = th.cat(
                        [
                            st.session_state.circuits_dict["circuits"],
                            current_circuit.unsqueeze(0),
                        ],
                        dim=0,
                    )
                    st.session_state.circuits_dict["names"].append(circuit_name)

                # Save to disk
                save_circuits(st.session_state.circuits_dict)

                # Update current index to the newly added circuit
                st.session_state.current_circuit_idx = (
                    len(st.session_state.circuits_dict["names"]) - 1
                )

                st.success(f"Circuit saved as: {circuit_name}")
                st.rerun()  # Refresh to update the dropdown
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
            random_circuit = generate_random_mask(num_layers, num_experts, k_value)

            # Add to circuits dictionary
            if (
                "circuits" not in st.session_state.circuits_dict
                or st.session_state.circuits_dict["circuits"].shape[0] == 0
            ):
                st.session_state.circuits_dict["circuits"] = random_circuit.unsqueeze(0)
                st.session_state.circuits_dict["names"] = ["Random Circuit"]
            else:
                # Append to existing circuits
                st.session_state.circuits_dict["circuits"] = th.cat(
                    [
                        st.session_state.circuits_dict["circuits"],
                        random_circuit.unsqueeze(0),
                    ],
                    dim=0,
                )
                st.session_state.circuits_dict["names"].append(
                    f"Random Circuit {len(st.session_state.circuits_dict['names']) + 1}"
                )

            # Save to disk
            save_circuits(st.session_state.circuits_dict)

            # Update current index to the newly added circuit
            st.session_state.current_circuit_idx = (
                len(st.session_state.circuits_dict["names"]) - 1
            )

            # Auto-compute max activating examples
            st.session_state.norm_scores, st.session_state.top_indices = (
                compute_max_activating_examples(
                    random_circuit,
                    token_topk_mask,
                    top_k,
                    device=device,
                    top_n=top_n,
                )
            )

            st.success("Random circuit generated!")
            st.rerun()  # Refresh to update the dropdown

        # Vertical slider for selected cell
        st.subheader("Cell Value Adjustment")
        if st.session_state.selected_cell:
            layer, expert = st.session_state.selected_cell

            # Get current circuit
            if st.session_state.current_circuit_idx >= 0:
                current_circuit = st.session_state.circuits_dict["circuits"][
                    st.session_state.current_circuit_idx
                ]
                current_value = float(current_circuit[layer, expert])

                new_value = st.slider(
                    f"Value for Layer {layer}, Expert {expert}",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_value,
                    step=0.01,
                    key=f"slider_{layer}_{expert}",
                )

                if new_value != current_value:
                    current_circuit[layer, expert] = new_value
                    # Auto-compute max activating examples
                    st.session_state.norm_scores, st.session_state.top_indices = (
                        compute_max_activating_examples(
                            current_circuit,
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
        else:
            st.info("Click a cell in the circuit to adjust its value")

    with col1:
        # Circuit visualization
        st.subheader("Interactive Circuit Grid")
        st.write("Click on a cell to select it. Click again to toggle between 0 and 1.")

        # Get current circuit
        if st.session_state.current_circuit_idx >= 0:
            current_circuit = st.session_state.circuits_dict["circuits"][
                st.session_state.current_circuit_idx
            ]

            # Get circuit dimensions
            num_layers, num_experts = current_circuit.shape

            # Create a grid of buttons for interactive circuit editing
            for layer in range(num_layers):
                cols = st.columns(num_experts)
                for expert in range(num_experts):
                    with cols[expert]:
                        # Get current value
                        value = float(current_circuit[layer, expert])

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
                                current_circuit[layer, expert] = 1.0 - value
                                # Auto-compute max activating examples
                                (
                                    st.session_state.norm_scores,
                                    st.session_state.top_indices,
                                ) = compute_max_activating_examples(
                                    current_circuit,
                                    token_topk_mask,
                                    top_k,
                                    device=device,
                                    top_n=top_n,
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
        else:
            st.info("No circuit selected. Please select or create a circuit.")

    # Four quadrants display
    if st.session_state.norm_scores is not None and st.session_state.top_indices:
        st.subheader("Max Activating Examples")

        # Example selection
        example_idx = st.selectbox(
            "Select example",
            range(len(st.session_state.top_indices)),
            format_func=lambda i: f"Example {i + 1} (Score: {st.session_state.norm_scores[st.session_state.top_indices[i]]:.4f})",
            index=st.session_state.current_example_idx
            if st.session_state.current_example_idx < len(st.session_state.top_indices)
            else 0,
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
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(5, 3))

                    # Get current circuit
                    if st.session_state.current_circuit_idx >= 0:
                        current_circuit = st.session_state.circuits_dict["circuits"][
                            st.session_state.current_circuit_idx
                        ]
                        im = ax.imshow(
                            current_circuit.cpu(), cmap="Blues", vmin=0, vmax=1
                        )
                        plt.colorbar(im, ax=ax)
                        ax.set_xlabel("Expert")
                        ax.set_ylabel("Layer")
                        ax.set_title("Circuit Mask")
                        ax.set_xticks(range(current_circuit.shape[1]))
                        ax.set_yticks(range(current_circuit.shape[0]))
                        st.pyplot(fig)

                # Get the selected token's global index
                selected_token_global_idx = (
                    st.session_state.selected_token_idx or token_idx
                )

                with q3:
                    st.subheader("Token Activation")
                    # Render the token activation
                    fig, token_mask = render_token_activation(
                        st.session_state.token_topk_mask, selected_token_global_idx
                    )
                    st.pyplot(fig)

                with q4:
                    st.subheader("Element-wise Product")
                    # Get current circuit
                    if st.session_state.current_circuit_idx >= 0:
                        current_circuit = st.session_state.circuits_dict["circuits"][
                            st.session_state.current_circuit_idx
                        ]
                        # Render the element-wise product
                        fig = render_element_product(token_mask, current_circuit.cpu())
                        st.pyplot(fig)


if __name__ == "__main__":
    max_activating_examples_server()
