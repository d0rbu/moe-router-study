"""Analyze expert coactivation using Jaccard distance in MoE models."""

from itertools import count
import os

import arguably
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR


def compute_jaccard_distance_matrix(activated_experts: th.Tensor) -> th.Tensor:
    """Compute pairwise Jaccard distance between expert activations.

    Jaccard distance = |A & B| / |A | B|
    where A and B are the sets of samples activating each expert.

    Args:
        activated_experts: Binary activation matrix of shape (num_experts, num_samples).

    Returns:
        Jaccard distance matrix of shape (num_experts, num_experts).
    """
    # Convert to float for matrix operations
    activated = activated_experts.float()

    # Compute intersection: A & B
    # This is the count of samples where both experts are activated
    intersection = activated @ activated.T

    # Compute union: A | B
    # This is |A| + |B| - |A & B|
    expert_activation_counts = activated.sum(dim=1)
    union = (
        expert_activation_counts.unsqueeze(1)
        + expert_activation_counts.unsqueeze(0)
        - intersection
    )

    # Compute Jaccard distance
    # Avoid division by zero
    jaccard = th.where(union > 0, intersection / union, th.zeros_like(intersection))

    return jaccard


def compute_independent_jaccard_matrix(activated_experts: th.Tensor) -> th.Tensor:
    """Compute expected Jaccard distances if experts were independent.

    For independent experts, Jaccard = (xy) / (x + y - xy)
    where x and y are the activation rates of each expert.

    Args:
        activated_experts: Binary activation matrix of shape (num_experts, num_samples).

    Returns:
        Expected Jaccard distance matrix for independent activations.
    """
    # Compute activation rates for each expert
    activation_rates = activated_experts.float().mean(dim=1)  # Shape: (num_experts,)

    # Compute pairwise independent Jaccard distances
    x = activation_rates.unsqueeze(1)  # Shape: (num_experts, 1)
    y = activation_rates.unsqueeze(0)  # Shape: (1, num_experts)

    # Independent coactivation rate: xy
    independent_coactivation = x * y

    # Independent union rate: x + y - xy
    independent_union = x + y - independent_coactivation

    # Independent Jaccard: xy / (x + y - xy)
    independent_jaccard = th.where(
        independent_union > 0,
        independent_coactivation / independent_union,
        th.zeros_like(independent_union),
    )

    return independent_jaccard


@arguably.command
def router_jaccard_distance(experiment_name: str) -> None:
    """Compute Jaccard distance between expert activations.

    This script:
    1. Loads router activations from stored .pt files
    2. Converts router logits to binary activations via top-k
    3. Computes Jaccard distance for each pair of experts
    4. Generates matrix visualizations and bar plots

    Args:
        experiment_name: Name of the experiment to analyze.
    """
    activated_experts_collection = []
    top_k: int | None = None

    activations_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)

    print(f"Loading router activations from {activations_dir}...")

    for file_idx in tqdm(count(), desc="Loading router activations"):
        file_path = os.path.join(activations_dir, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break

        output = th.load(file_path)
        top_k = output["topk"]
        router_logits = output[str(ActivationKeys.ROUTER_LOGITS)]

        num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
        total_experts = num_layers * num_experts

        # Convert to binary activations (top-k selection)
        top_k_indices = th.topk(router_logits, k=top_k, dim=2).indices
        activated_experts = th.zeros_like(router_logits)
        activated_experts.scatter_(2, top_k_indices, 1)

        # Reshape: (B, L, E) -> (L * E, B)
        activated_experts_collection.append(
            activated_experts.reshape(-1, total_experts).T
        )

    if top_k is None:
        raise ValueError("No data files found")

    # Concatenate all batches: (L * E, total_samples)
    activated_experts = th.cat(activated_experts_collection, dim=-1)
    batch_size = activated_experts.shape[-1]

    print("\nComputing Jaccard distances...")
    print(f"Total samples: {batch_size:,}")
    print(f"Total experts (across all layers): {total_experts}")

    # Compute actual and independent Jaccard distance matrices
    jaccard = compute_jaccard_distance_matrix(activated_experts)
    independent_jaccard = compute_independent_jaccard_matrix(activated_experts)

    # Extract upper triangle (excluding diagonal) for bar plots
    upper_triangular_mask = th.triu(th.ones_like(jaccard).bool(), diagonal=1)
    jaccard_upper = jaccard[upper_triangular_mask]
    independent_upper = independent_jaccard[upper_triangular_mask]

    # Sort distances for bar plots
    sorted_jaccard, _ = th.sort(jaccard_upper, descending=True)
    sorted_independent, _ = th.sort(independent_upper, descending=True)

    # Separate cross-layer distances
    # Create layer indices for each expert
    layer_indices = th.arange(total_experts) // num_experts
    layer_i = layer_indices.unsqueeze(1)  # Shape: (total_experts, 1)
    layer_j = layer_indices.unsqueeze(0)  # Shape: (1, total_experts)
    cross_layer_mask = (layer_i != layer_j) & upper_triangular_mask

    cross_layer_jaccard = jaccard[cross_layer_mask]
    sorted_cross_layer, _ = th.sort(cross_layer_jaccard, descending=True)

    # Print statistics
    print("\nJaccard distance statistics:")
    print("  All pairs:")
    print(f"    Mean: {jaccard_upper.mean().item():.4f}")
    print(f"    Median: {jaccard_upper.median().item():.4f}")
    print(f"    Max: {jaccard_upper.max().item():.4f}")
    print("  Cross-layer pairs:")
    print(f"    Mean: {cross_layer_jaccard.mean().item():.4f}")
    print(f"    Median: {cross_layer_jaccard.median().item():.4f}")
    print(f"    Max: {cross_layer_jaccard.max().item():.4f}")
    print("  Independent baseline:")
    print(f"    Mean: {independent_upper.mean().item():.4f}")
    print(f"    Median: {independent_upper.median().item():.4f}")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (12, 10)

    print("\nGenerating visualizations...")

    # Plot 1: Absolute Jaccard distance matrix
    plt.figure()
    plt.imshow(jaccard.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Jaccard Distance")
    plt.title("Jaccard Distance Matrix (Absolute)")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_matrix_absolute.png"))
    plt.close()

    # Plot 2: Independent Jaccard distance matrix
    plt.figure()
    plt.imshow(independent_jaccard.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Independent Jaccard Distance")
    plt.title("Independent Jaccard Distance Matrix")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_matrix_independent.png"))
    plt.close()

    # Plot 3: Relative Jaccard distance matrix (red-black-blue colormap)
    # Red = 0, Black = independent value, Blue = 2x independent value
    relative_jaccard = jaccard / (independent_jaccard + 1e-8)  # Avoid division by zero

    # Create custom colormap: red (0) -> black (1) -> blue (2)
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["red", "black", "blue"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    plt.figure()
    plt.imshow(relative_jaccard.cpu().numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=2)
    plt.colorbar(label="Jaccard / Independent Jaccard")
    plt.title(
        "Relative Jaccard Distance Matrix\n(Red=0, Black=Independent, Blue=2x Independent)"
    )
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_matrix_relative.png"))
    plt.close()

    # Plot 4: Bar plot of sorted Jaccard distances (upper triangular)
    plt.figure()
    plt.bar(range(len(sorted_jaccard)), sorted_jaccard.cpu().numpy())
    plt.xlabel("Expert Pair Rank")
    plt.ylabel("Jaccard Distance")
    plt.title("Sorted Jaccard Distances (Upper Triangular)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_bar_sorted.png"))
    plt.close()

    # Plot 5: Bar plot of sorted cross-layer Jaccard distances
    plt.figure()
    plt.bar(range(len(sorted_cross_layer)), sorted_cross_layer.cpu().numpy())
    plt.xlabel("Cross-Layer Pair Rank")
    plt.ylabel("Jaccard Distance")
    plt.title("Sorted Cross-Layer Jaccard Distances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_bar_cross_layer.png"))
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}/")
    print("  - router_jaccard_matrix_absolute.png")
    print("  - router_jaccard_matrix_independent.png")
    print("  - router_jaccard_matrix_relative.png")
    print("  - router_jaccard_bar_sorted.png")
    print("  - router_jaccard_bar_cross_layer.png")


if __name__ == "__main__":
    arguably.run()
