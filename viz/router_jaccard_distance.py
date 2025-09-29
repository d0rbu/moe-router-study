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


@arguably.command()
def router_jaccard_distance(experiment_name: str) -> None:
    """Compute Jaccard distance between expert activations.

    This script:
    1. Loads router activations from stored .pt files
    2. Converts router logits to binary activations via top-k
    3. Computes Jaccard distance for each pair of experts
    4. Generates visualizations similar to correlation analysis

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

    # Compute Jaccard distance matrix
    jaccard = compute_jaccard_distance_matrix(activated_experts)

    # Build control by shuffling along the batch dimension per-layer
    # Reconstruct to (B, L, E)
    activated_experts_ble = activated_experts.T.view(
        batch_size, num_layers, num_experts
    )

    # Initialize with a copy and shuffle batch indices independently for each layer
    random_activated_experts_ble = activated_experts_ble.clone()
    for layer_idx in range(num_layers):
        perm = th.randperm(batch_size, device=activated_experts.device)
        random_activated_experts_ble[:, layer_idx, :] = activated_experts_ble[
            perm, layer_idx, :
        ]

    # Reshape back: (B, L, E) -> (L * E, B)
    random_activated_experts = random_activated_experts_ble.reshape(batch_size, -1).T

    # Compute random Jaccard distance matrix
    random_jaccard = compute_jaccard_distance_matrix(random_activated_experts)

    # Extract upper triangle (excluding diagonal) for analysis
    upper_triangular_mask = th.triu(th.ones_like(jaccard).bool(), diagonal=1).view(-1)

    # Flatten and sort
    jaccard_raw, indices_raw = th.sort(jaccard.view(-1))
    random_jaccard_raw, random_indices_raw = th.sort(random_jaccard.view(-1))

    # Filter for upper triangle
    sorted_upper_triangular_mask = upper_triangular_mask[indices_raw]
    jaccard_distances = jaccard_raw[sorted_upper_triangular_mask]
    random_jaccard_distances = random_jaccard_raw[sorted_upper_triangular_mask]

    # Compute indices for layer/expert identification
    indices = indices_raw[sorted_upper_triangular_mask]
    first_layer_indices = (indices // total_experts) // num_experts
    second_layer_indices = (indices % total_experts) // num_experts
    first_expert_indices = (indices % (num_experts * total_experts)) // total_experts
    second_expert_indices = indices % num_experts
    rolled_indices = th.stack(
        [
            first_layer_indices,
            second_layer_indices,
            first_expert_indices,
            second_expert_indices,
        ],
        dim=1,
    )

    # Separate within-layer and cross-layer distances
    within_layer_mask = first_layer_indices == second_layer_indices
    cross_layer_jaccard = jaccard_distances[~within_layer_mask]
    cross_layer_indices = rolled_indices[~within_layer_mask]

    # Random baseline for cross-layer
    random_indices = random_indices_raw[sorted_upper_triangular_mask]
    first_layer_random_indices = (random_indices // total_experts) // num_experts
    second_layer_random_indices = (random_indices % total_experts) // num_experts
    within_layer_random_mask = first_layer_random_indices == second_layer_random_indices
    cross_layer_random_jaccard = random_jaccard_distances[~within_layer_random_mask]

    # Print statistics
    print("\nJaccard distance statistics:")
    print("  All pairs:")
    print(f"    Mean: {jaccard_distances.mean().item():.4f}")
    print(f"    Median: {jaccard_distances.median().item():.4f}")
    print(f"    Max: {jaccard_distances.max().item():.4f}")
    print("  Cross-layer pairs:")
    print(f"    Mean: {cross_layer_jaccard.mean().item():.4f}")
    print(f"    Median: {cross_layer_jaccard.median().item():.4f}")
    print(f"    Max: {cross_layer_jaccard.max().item():.4f}")
    print("  Random baseline (cross-layer):")
    print(f"    Mean: {cross_layer_random_jaccard.mean().item():.4f}")
    print(f"    Median: {cross_layer_random_jaccard.median().item():.4f}")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (16, 12)

    print("\nGenerating visualizations...")

    # Plot 1: All Jaccard distances (sorted)
    plt.figure()
    plt.bar(range(len(jaccard_distances)), jaccard_distances.cpu())
    plt.xlabel("Expert pair rank")
    plt.ylabel("Jaccard distance")
    plt.title("All Pairwise Jaccard Distances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_distances.png"))
    plt.close()

    # Plot 2: Random baseline Jaccard distances
    plt.figure()
    plt.bar(range(len(random_jaccard_distances)), random_jaccard_distances.cpu())
    plt.xlabel("Expert pair rank")
    plt.ylabel("Jaccard distance")
    plt.title("Random Baseline Jaccard Distances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_distances_random.png"))
    plt.close()

    # Plot 3: Cross-layer Jaccard distances
    plt.figure()
    plt.bar(range(len(cross_layer_jaccard)), cross_layer_jaccard.cpu())
    plt.xlabel("Expert pair rank")
    plt.ylabel("Jaccard distance")
    plt.title("Cross-Layer Jaccard Distances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_distances_cross_layer.png"))
    plt.close()

    # Plot 4: Random baseline cross-layer Jaccard distances
    plt.figure()
    plt.bar(range(len(cross_layer_random_jaccard)), cross_layer_random_jaccard.cpu())
    plt.xlabel("Expert pair rank")
    plt.ylabel("Jaccard distance")
    plt.title("Random Baseline Cross-Layer Jaccard Distances")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "router_jaccard_distances_cross_layer_random.png")
    )
    plt.close()

    # Plot 5: Distribution comparison
    plt.figure()
    plt.hist(
        [
            jaccard_distances.cpu().numpy(),
            cross_layer_jaccard.cpu().numpy(),
            cross_layer_random_jaccard.cpu().numpy(),
        ],
        bins=50,
        alpha=0.6,
        label=["All pairs", "Cross-layer", "Random baseline"],
    )
    plt.xlabel("Jaccard distance")
    plt.ylabel("Count")
    plt.title("Distribution of Jaccard Distances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_distribution.png"))
    plt.close()

    # Print top cross-layer coactivations
    print("\nTop 10 cross-layer Jaccard distances:")
    for i in range(min(10, len(cross_layer_jaccard))):
        idx = -i - 1
        first_layer_idx, second_layer_idx, first_expert_idx, second_expert_idx = (
            cross_layer_indices[idx]
        )
        distance = cross_layer_jaccard[idx]
        print(
            f"  layer {first_layer_idx} expert {first_expert_idx} <-> "
            f"layer {second_layer_idx} expert {second_expert_idx}: {distance:.4f}"
        )

    print(f"\nFigures saved to {FIGURE_DIR}/")
    print("  - router_jaccard_distances.png")
    print("  - router_jaccard_distances_random.png")
    print("  - router_jaccard_distances_cross_layer.png")
    print("  - router_jaccard_distances_cross_layer_random.png")
    print("  - router_jaccard_distribution.png")


if __name__ == "__main__":
    arguably.run()
