"""Analyze expert coactivation using Jaccard similarity in MoE models."""

import asyncio
import os

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import torch as th

from exp.activations import Activations
from exp.get_activations import ActivationKeys
from moe import convert_router_logits_to_paths
from viz import FIGURE_DIR


async def _router_jaccard_distance_async(
    experiment_name: str, batch_size: int = 4096
) -> None:
    """Async implementation of Jaccard similarity analysis."""
    logger.info(f"Loading activations for experiment: {experiment_name}")

    # Load activations using the Activations class
    activations = await Activations.load(experiment_name=experiment_name)

    # Initialize running counts for memory-efficient Jaccard computation
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None
    total_samples = 0

    # Running counts for Jaccard similarity computation
    expert_activation_counts: th.Tensor | None = None
    pairwise_coactivation_counts: th.Tensor | None = None

    # Iterate through activation batches
    for batch in activations(batch_size=batch_size):
        router_logits = batch[ActivationKeys.ROUTER_LOGITS]

        if top_k is None:
            top_k = batch["topk"]
            num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
            total_experts = num_layers * num_experts
            logger.info(
                f"Router configuration: {num_layers} layers, {num_experts} experts per layer, top-k={top_k}"
            )

            # Initialize running count tensors
            expert_activation_counts = th.zeros(total_experts, dtype=th.float32)
            pairwise_coactivation_counts = th.zeros(
                total_experts, total_experts, dtype=th.float32
            )

        current_batch_size = router_logits.shape[0]
        total_samples += current_batch_size

        # Convert to binary activations (top-k selection)
        activated_experts = convert_router_logits_to_paths(router_logits, top_k)

        # Reshape: (B, L, E) -> (B, L * E)
        activated_experts_batch = activated_experts.reshape(-1, total_experts).float()

        # Update running counts
        expert_activation_counts += activated_experts_batch.sum(dim=0)
        pairwise_coactivation_counts += (
            activated_experts_batch.T @ activated_experts_batch
        )

    if (
        top_k is None
        or expert_activation_counts is None
        or pairwise_coactivation_counts is None
    ):
        raise ValueError("No activation data found")

    logger.info("Computing Jaccard similarities...")
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Total experts (across all layers): {total_experts}")

    # Compute Jaccard similarity matrix using running counts
    # Jaccard similarity = intersection / union = coactivation / (count_i + count_j - coactivation)
    union_counts = (
        expert_activation_counts.unsqueeze(1)
        + expert_activation_counts.unsqueeze(0)
        - pairwise_coactivation_counts
    )
    jaccard = th.where(
        union_counts > 0,
        pairwise_coactivation_counts / union_counts,
        th.zeros_like(pairwise_coactivation_counts),
    )

    # Compute independent baseline using marginal probabilities
    expert_probabilities = expert_activation_counts / total_samples

    # For independent experts, Jaccard = (p_i * p_j) / (p_i + p_j - p_i * p_j)
    p_i = expert_probabilities.unsqueeze(1)  # Shape: (num_experts, 1)
    p_j = expert_probabilities.unsqueeze(0)  # Shape: (1, num_experts)

    # Independent coactivation probability: p_i * p_j
    independent_coactivation = p_i * p_j

    # Independent union probability: p_i + p_j - p_i * p_j
    independent_union = p_i + p_j - independent_coactivation

    # Independent Jaccard similarity: coactivation / union
    independent_jaccard = th.where(
        independent_union > 0,
        independent_coactivation / independent_union,
        th.zeros_like(independent_union),
    )

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
    logger.info("Jaccard similarity statistics:")
    logger.info("  All pairs:")
    logger.info(f"    Mean: {jaccard_upper.mean().item():.4f}")
    logger.info(f"    Median: {jaccard_upper.median().item():.4f}")
    logger.info(f"    Max: {jaccard_upper.max().item():.4f}")
    logger.info("  Cross-layer pairs:")
    logger.info(f"    Mean: {cross_layer_jaccard.mean().item():.4f}")
    logger.info(f"    Median: {cross_layer_jaccard.median().item():.4f}")
    logger.info(f"    Max: {cross_layer_jaccard.max().item():.4f}")
    logger.info("  Independent baseline:")
    logger.info(f"    Mean: {independent_upper.mean().item():.4f}")
    logger.info(f"    Median: {independent_upper.median().item():.4f}")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (12, 10)

    logger.info("Generating visualizations...")

    # Plot 1: Absolute Jaccard similarity matrix
    plt.figure()
    plt.imshow(jaccard.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Jaccard Similarity")
    plt.title("Jaccard Similarity Matrix (Absolute)")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_matrix_absolute.png"))
    plt.close()

    # Plot 2: Independent Jaccard similarity matrix
    plt.figure()
    plt.imshow(independent_jaccard.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Independent Jaccard Similarity")
    plt.title("Independent Jaccard Similarity Matrix")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_matrix_independent.png"))
    plt.close()

    # Plot 3: Relative Jaccard similarity matrix (red-black-blue colormap)
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
        "Relative Jaccard Similarity Matrix\n(Red=0, Black=Independent, Blue=2x Independent)"
    )
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_matrix_relative.png"))
    plt.close()

    # Plot 4: Bar plot of sorted Jaccard similarities (upper triangular)
    plt.figure()
    plt.bar(range(len(sorted_jaccard)), sorted_jaccard.cpu().numpy())
    plt.xlabel("Expert Pair Rank")
    plt.ylabel("Jaccard Similarity")
    plt.title("Sorted Jaccard Similarities (Upper Triangular)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_bar_sorted.png"))
    plt.close()

    # Plot 5: Bar plot of sorted cross-layer Jaccard similarities
    plt.figure()
    plt.bar(range(len(sorted_cross_layer)), sorted_cross_layer.cpu().numpy())
    plt.xlabel("Cross-Layer Pair Rank")
    plt.ylabel("Jaccard Similarity")
    plt.title("Sorted Cross-Layer Jaccard Similarities")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_jaccard_bar_cross_layer.png"))
    plt.close()

    logger.info(f"Figures saved to {FIGURE_DIR}/")
    logger.info("  - router_jaccard_matrix_absolute.png")
    logger.info("  - router_jaccard_matrix_independent.png")
    logger.info("  - router_jaccard_matrix_relative.png")
    logger.info("  - router_jaccard_bar_sorted.png")
    logger.info("  - router_jaccard_bar_cross_layer.png")


@arguably.command
def router_jaccard_distance(experiment_name: str, batch_size: int = 4096) -> None:
    """Compute Jaccard similarity between expert activations.

    This script:
    1. Loads router activations using the Activations class
    2. Converts router logits to binary activations via top-k
    3. Computes Jaccard similarity for each pair of experts
    4. Generates matrix visualizations and bar plots

    Args:
        experiment_name: Name of the experiment to analyze.
        batch_size: Number of samples to process per batch (default: 4096).
    """
    asyncio.run(_router_jaccard_distance_async(experiment_name, batch_size))


if __name__ == "__main__":
    arguably.run()
