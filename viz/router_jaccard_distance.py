"""Analyze expert coactivation using Jaccard similarity in MoE models."""

import asyncio
import os

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import torch as th

from core.moe import convert_router_logits_to_paths
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR


async def _router_jaccard_distance_async(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 5_000,
    context_length: int = 2048,
    batch_size: int = 4096,
    reshuffled_tokens_per_file: int = 100000,
    num_workers: int = 8,
    debug: bool = False,
    max_samples: int = 0,
) -> None:
    """Async implementation of Jaccard similarity analysis."""
    logger.info(f"Loading activations for model: {model_name}, dataset: {dataset_name}")
    logger.debug(f"Batch size: {batch_size}")

    assert model_name, "Model name cannot be empty"
    assert dataset_name, "Dataset name cannot be empty"
    assert tokens_per_file > 0, (
        f"Tokens per file must be positive, got {tokens_per_file}"
    )
    assert context_length > 0, f"Context length must be positive, got {context_length}"
    assert batch_size > 0, f"Batch size must be positive, got {batch_size}"

    logger.debug("Loading activations and initializing distributed...")
    (
        activations,
        _activation_dims,
    ) = await load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        submodule_names=[ActivationKeys.ROUTER_LOGITS],
        context_length=context_length,
        num_workers=num_workers,
        debug=debug,
    )
    logger.debug("Activations loaded successfully")

    # Initialize running counts for memory-efficient Jaccard computation
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None
    total_samples = 0
    batch_count = 0

    # Running counts for Jaccard similarity computation
    expert_activation_counts: th.Tensor | None = None
    pairwise_coactivation_counts: th.Tensor | None = None

    logger.debug("Starting batch processing...")

    # Add assertion for non-negative max_samples
    assert max_samples >= 0, f"max_samples must be non-negative, got {max_samples}"

    if max_samples > 0:
        logger.info(f"Processing first {max_samples:,} samples")
    else:
        logger.info("Processing all available samples")

    for batch in activations(batch_size=batch_size, max_samples=max_samples):
        batch_count += 1
        logger.trace(f"Processing batch {batch_count}")

        router_logits = batch[ActivationKeys.ROUTER_LOGITS]
        logger.trace(
            f"Router logits shape: {router_logits.shape}, dtype: {router_logits.dtype}"
        )

        assert router_logits.dim() == 3, (
            f"Expected 3D router logits, got {router_logits.dim()}D"
        )
        assert router_logits.shape[0] > 0, "Batch size must be positive"
        assert router_logits.shape[1] > 0, "Number of layers must be positive"
        assert router_logits.shape[2] > 0, "Number of experts must be positive"

        if top_k is None:
            top_k = batch["topk"]
            num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
            total_experts = num_layers * num_experts
            logger.info(
                f"Router configuration: {num_layers} layers, {num_experts} experts per layer, top-k={top_k}"
            )
            logger.debug(f"Total experts across all layers: {total_experts}")

            assert top_k > 0, f"Top-k must be positive, got {top_k}"
            assert top_k <= num_experts, (
                f"Top-k {top_k} cannot exceed number of experts {num_experts}"
            )
            assert num_layers > 0, (
                f"Number of layers must be positive, got {num_layers}"
            )
            assert num_experts > 0, (
                f"Number of experts must be positive, got {num_experts}"
            )

            # Initialize running count tensors
            logger.debug("Initializing running count tensors...")
            expert_activation_counts = th.zeros(total_experts, dtype=th.float32)
            pairwise_coactivation_counts = th.zeros(
                total_experts, total_experts, dtype=th.float32
            )
            logger.trace(
                f"Expert activation counts shape: {expert_activation_counts.shape}"
            )
            logger.trace(
                f"Pairwise coactivation counts shape: {pairwise_coactivation_counts.shape}"
            )

        current_batch_size = router_logits.shape[0]
        total_samples += current_batch_size
        logger.trace(
            f"Current batch size: {current_batch_size}, total samples: {total_samples}"
        )

        # Convert to binary activations (top-k selection)
        logger.trace("Converting router logits to binary activations...")
        activated_experts = convert_router_logits_to_paths(router_logits, top_k).bool()
        logger.trace(
            f"Activated experts shape: {activated_experts.shape}, dtype: {activated_experts.dtype}"
        )

        # Validate activated experts
        assert activated_experts.shape == (
            current_batch_size,
            num_layers,
            num_experts,
        ), f"Unexpected activated experts shape: {activated_experts.shape}"
        assert activated_experts.dtype == th.bool, (
            f"Expected bool tensor, got {activated_experts.dtype}"
        )

        # Check that exactly top_k experts are activated per layer
        experts_per_layer = activated_experts.sum(dim=2)
        logger.trace(f"Experts per layer: {experts_per_layer}")
        assert th.all(experts_per_layer == top_k), (
            f"Expected {top_k} experts per layer, got {experts_per_layer}"
        )

        # Reshape: (B, L, E) -> (B, L * E)
        activated_experts_batch = activated_experts.reshape(-1, total_experts).float()
        logger.trace(f"Reshaped activated experts: {activated_experts_batch.shape}")

        # Validate reshaped tensor
        assert activated_experts_batch.shape == (current_batch_size, total_experts), (
            f"Unexpected reshaped shape: {activated_experts_batch.shape}"
        )
        assert th.all(
            (activated_experts_batch == 0) | (activated_experts_batch == 1)
        ), "Binary activations must be 0 or 1"

        # Update running counts
        logger.trace("Updating running counts...")
        batch_expert_counts = activated_experts_batch.sum(dim=0)
        batch_total_activations = batch_expert_counts.sum().item()
        expected_batch_activations = current_batch_size * num_layers * top_k

        logger.trace(
            f"Batch activations: actual={batch_total_activations:.1f}, "
            f"expected={expected_batch_activations}, "
            f"diff={abs(batch_total_activations - expected_batch_activations):.1f}"
        )

        expert_activation_counts += batch_expert_counts
        pairwise_coactivation_counts += (
            activated_experts_batch.T @ activated_experts_batch
        )

        logger.trace(
            f"Expert activation counts sum: {expert_activation_counts.sum().item():.6f}"
        )
        logger.trace(
            f"Pairwise coactivation counts sum: {pairwise_coactivation_counts.sum().item():.6f}"
        )

        # Validate running counts
        expected_expert_sum = total_samples * num_layers * top_k
        actual_expert_sum = expert_activation_counts.sum().item()
        expert_sum_diff = abs(actual_expert_sum - expected_expert_sum)
        expert_sum_rel_error = (
            expert_sum_diff / expected_expert_sum if expected_expert_sum > 0 else 0
        )

        logger.trace(
            f"Expert sum validation: actual={actual_expert_sum:.1f}, "
            f"expected={expected_expert_sum}, diff={expert_sum_diff:.1f}, "
            f"rel_error={expert_sum_rel_error:.2e}"
        )

        assert expert_sum_rel_error < 1e-4, (
            f"Expert activation sum validation failed: "
            f"actual={actual_expert_sum:.1f}, expected={expected_expert_sum}, "
            f"diff={expert_sum_diff:.1f}, rel_error={expert_sum_rel_error:.2e}"
        )

        logger.debug(
            f"Batch {batch_count} complete: {current_batch_size} samples processed, {total_samples} total"
        )

    if (
        top_k is None
        or expert_activation_counts is None
        or pairwise_coactivation_counts is None
    ):
        raise ValueError("No activation data found")

    logger.info("Computing Jaccard similarities...")
    logger.info(f"Total batches processed: {batch_count}")
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Total experts (across all layers): {total_experts}")

    assert total_samples > 0, "No samples were processed"
    assert expert_activation_counts is not None, (
        "Expert activation counts not initialized"
    )
    assert pairwise_coactivation_counts is not None, (
        "Pairwise coactivation counts not initialized"
    )
    assert expert_activation_counts.shape == (total_experts,), (
        f"Unexpected expert counts shape: {expert_activation_counts.shape}"
    )
    assert pairwise_coactivation_counts.shape == (total_experts, total_experts), (
        f"Unexpected pairwise counts shape: {pairwise_coactivation_counts.shape}"
    )

    # Compute Jaccard similarity matrix using running counts
    logger.debug("Computing Jaccard similarity matrix...")
    logger.trace(
        f"Expert activation counts range: min={expert_activation_counts.min().item():.2f}, max={expert_activation_counts.max().item():.2f}"
    )
    logger.trace(
        f"Pairwise coactivation counts range: min={pairwise_coactivation_counts.min().item():.2f}, max={pairwise_coactivation_counts.max().item():.2f}"
    )

    # Jaccard similarity = intersection / union = coactivation / (count_i + count_j - coactivation)
    union_counts = (
        expert_activation_counts.unsqueeze(1)
        + expert_activation_counts.unsqueeze(0)
        - pairwise_coactivation_counts
    )
    logger.trace(
        f"Union counts range: min={union_counts.min().item():.2f}, max={union_counts.max().item():.2f}"
    )

    # Validate union counts
    assert th.all(union_counts >= 0), "Union counts must be non-negative"
    assert th.all(union_counts >= pairwise_coactivation_counts), (
        "Union counts must be >= coactivation counts"
    )

    jaccard = th.where(
        union_counts > 0,
        pairwise_coactivation_counts / union_counts,
        th.zeros_like(pairwise_coactivation_counts),
    )
    logger.trace(
        f"Jaccard similarity range: min={jaccard.min().item():.6f}, max={jaccard.max().item():.6f}"
    )

    # Validate Jaccard similarities
    assert th.all(jaccard >= 0), "Jaccard similarities must be non-negative"
    assert th.all(jaccard <= 1), "Jaccard similarities must be <= 1"
    assert th.allclose(jaccard, jaccard.T), "Jaccard matrix should be symmetric"

    # Compute independent baseline using marginal probabilities
    logger.debug("Computing independent baseline...")
    # Independent marginal probability that each expert gets activated across all samples
    # This is NOT the router's output probabilities, but the frequency of activation
    independent_expert_probabilities = expert_activation_counts / total_samples
    logger.trace(
        f"Independent expert probabilities range: min={independent_expert_probabilities.min().item():.6f}, max={independent_expert_probabilities.max().item():.6f}"
    )
    logger.trace(
        f"Independent expert probabilities sum: {independent_expert_probabilities.sum().item():.6f}"
    )

    # Validate probabilities
    assert th.all(independent_expert_probabilities >= 0), (
        "Independent expert probabilities must be non-negative"
    )
    assert th.all(independent_expert_probabilities <= 1), (
        "Independent expert probabilities must be <= 1"
    )

    # Validate probability sum with relative error tolerance using th.allclose
    # Expected sum = num_layers * top_k because top_k experts are selected per layer
    expected_prob_sum = num_layers * top_k
    actual_prob_sum = independent_expert_probabilities.sum()

    assert th.allclose(
        actual_prob_sum,
        th.tensor(
            expected_prob_sum,
            dtype=actual_prob_sum.dtype,
            device=actual_prob_sum.device,
        ),
        rtol=1e-4,
    ), (
        f"Independent expert probabilities sum validation failed: "
        f"actual={actual_prob_sum.item():.8f}, expected={expected_prob_sum}"
    )

    # For independent experts, Jaccard = (p_i * p_j) / (p_i + p_j - p_i * p_j)
    p_i = independent_expert_probabilities.unsqueeze(1)  # Shape: (num_experts, 1)
    p_j = independent_expert_probabilities.unsqueeze(0)  # Shape: (1, num_experts)
    logger.trace(f"p_i shape: {p_i.shape}, p_j shape: {p_j.shape}")

    # Independent coactivation probability: p_i * p_j
    independent_coactivation = p_i * p_j
    logger.trace(
        f"Independent coactivation range: min={independent_coactivation.min().item():.8f}, max={independent_coactivation.max().item():.8f}"
    )

    # Independent union probability: p_i + p_j - p_i * p_j
    independent_union = p_i + p_j - independent_coactivation
    logger.trace(
        f"Independent union range: min={independent_union.min().item():.6f}, max={independent_union.max().item():.6f}"
    )

    # Validate independent calculations
    assert th.all(independent_coactivation >= 0), (
        "Independent coactivation must be non-negative"
    )
    assert th.all(independent_coactivation <= 1), (
        "Independent coactivation must be <= 1"
    )
    assert th.all(independent_union >= 0), "Independent union must be non-negative"
    assert th.all(independent_union >= independent_coactivation), (
        "Independent union must be >= coactivation"
    )

    # Independent Jaccard similarity: coactivation / union
    independent_jaccard = th.where(
        independent_union > 0,
        independent_coactivation / independent_union,
        th.zeros_like(independent_union),
    )
    logger.trace(
        f"Independent Jaccard range: min={independent_jaccard.min().item():.8f}, max={independent_jaccard.max().item():.8f}"
    )

    # Validate independent Jaccard
    assert th.all(independent_jaccard >= 0), "Independent Jaccard must be non-negative"
    assert th.all(independent_jaccard <= 1), "Independent Jaccard must be <= 1"
    assert th.allclose(independent_jaccard, independent_jaccard.T), (
        "Independent Jaccard matrix should be symmetric"
    )

    # Extract upper triangle (excluding diagonal) for bar plots
    logger.debug("Extracting upper triangular matrices...")
    upper_triangular_mask = th.triu(th.ones_like(jaccard).bool(), diagonal=1)
    jaccard_upper = jaccard[upper_triangular_mask]
    independent_upper = independent_jaccard[upper_triangular_mask]

    logger.trace(f"Upper triangular mask shape: {upper_triangular_mask.shape}")
    logger.trace(
        f"Upper triangular elements count: {upper_triangular_mask.sum().item()}"
    )
    logger.trace(
        f"Jaccard upper range: min={jaccard_upper.min().item():.6f}, max={jaccard_upper.max().item():.6f}"
    )
    logger.trace(
        f"Independent upper range: min={independent_upper.min().item():.8f}, max={independent_upper.max().item():.8f}"
    )

    # Validate upper triangular extraction
    expected_upper_count = total_experts * (total_experts - 1) // 2
    assert len(jaccard_upper) == expected_upper_count, (
        f"Unexpected upper triangular count: {len(jaccard_upper)} != {expected_upper_count}"
    )
    assert len(independent_upper) == expected_upper_count, (
        f"Unexpected independent upper count: {len(independent_upper)} != {expected_upper_count}"
    )

    # Sort distances for bar plots
    logger.debug("Sorting similarities for visualization...")
    sorted_jaccard, _ = th.sort(jaccard_upper, descending=True)
    sorted_independent, _ = th.sort(independent_upper, descending=True)

    # Validate sorting
    assert th.all(sorted_jaccard[:-1] >= sorted_jaccard[1:]), (
        "Jaccard similarities not properly sorted"
    )
    assert th.all(sorted_independent[:-1] >= sorted_independent[1:]), (
        "Independent similarities not properly sorted"
    )

    # Separate cross-layer distances
    logger.debug("Computing cross-layer similarities...")
    # Create layer indices for each expert
    layer_indices = th.arange(total_experts) // num_experts
    layer_i = layer_indices.unsqueeze(1)  # Shape: (total_experts, 1)
    layer_j = layer_indices.unsqueeze(0)  # Shape: (1, total_experts)
    cross_layer_mask = (layer_i != layer_j) & upper_triangular_mask

    logger.trace(f"Layer indices: {layer_indices}")
    logger.trace(f"Cross-layer mask count: {cross_layer_mask.sum().item()}")

    # Validate cross-layer mask
    assert num_layers is not None, "num_layers should not be None at this point"
    expected_cross_layer_count = (
        num_layers * (num_layers - 1) * num_experts * num_experts // 2
    )
    assert cross_layer_mask.sum().item() == expected_cross_layer_count, (
        f"Unexpected cross-layer count: {cross_layer_mask.sum().item()} != {expected_cross_layer_count}"
    )

    cross_layer_jaccard = jaccard[cross_layer_mask]
    sorted_cross_layer, _ = th.sort(cross_layer_jaccard, descending=True)

    logger.trace(
        f"Cross-layer Jaccard range: min={cross_layer_jaccard.min().item():.6f}, max={cross_layer_jaccard.max().item():.6f}"
    )
    assert th.all(sorted_cross_layer[:-1] >= sorted_cross_layer[1:]), (
        "Cross-layer similarities not properly sorted"
    )

    # Print statistics
    logger.info("Jaccard similarity statistics:")
    logger.info("  All pairs:")
    logger.info(f"    Mean: {jaccard_upper.mean().item():.4f}")
    logger.info(f"    Median: {jaccard_upper.median().item():.4f}")
    logger.info(f"    Max: {jaccard_upper.max().item():.4f}")
    logger.info(f"    Std: {jaccard_upper.std().item():.4f}")
    logger.info("  Cross-layer pairs:")
    logger.info(f"    Mean: {cross_layer_jaccard.mean().item():.4f}")
    logger.info(f"    Median: {cross_layer_jaccard.median().item():.4f}")
    logger.info(f"    Max: {cross_layer_jaccard.max().item():.4f}")
    logger.info(f"    Std: {cross_layer_jaccard.std().item():.4f}")
    logger.info("  Independent baseline:")
    logger.info(f"    Mean: {independent_upper.mean().item():.4f}")
    logger.info(f"    Median: {independent_upper.median().item():.4f}")
    logger.info(f"    Std: {independent_upper.std().item():.4f}")

    # Additional validation statistics
    logger.debug("Validation statistics:")
    logger.debug(
        f"  Diagonal Jaccard (should be 1.0): {th.diag(jaccard).mean().item():.6f}"
    )
    logger.debug(
        f"  Diagonal Independent (should be 1.0): {th.diag(independent_jaccard).mean().item():.6f}"
    )
    logger.debug(f"  Zero Jaccard count: {(jaccard_upper == 0).sum().item()}")
    logger.debug(f"  Zero Independent count: {(independent_upper == 0).sum().item()}")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (12, 10)

    logger.info("Generating visualizations...")

    # Ensure output directory exists
    os.makedirs(FIGURE_DIR, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {FIGURE_DIR}")

    # Plot 1: Absolute Jaccard similarity matrix
    logger.debug("Creating absolute Jaccard similarity matrix plot...")
    plt.figure()
    jaccard_numpy = jaccard.cpu().numpy()
    logger.trace(
        f"Plotting Jaccard matrix: {jaccard_numpy.shape}, range=[{jaccard_numpy.min():.6f}, {jaccard_numpy.max():.6f}]"
    )

    plt.imshow(jaccard_numpy, cmap="viridis", aspect="auto")
    plt.colorbar(label="Jaccard Similarity")
    plt.title("Jaccard Similarity Matrix (Absolute)")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()

    abs_plot_path = os.path.join(FIGURE_DIR, "router_jaccard_matrix_absolute.png")
    logger.debug(f"Saving absolute matrix plot to: {abs_plot_path}")
    plt.savefig(abs_plot_path)
    plt.close()
    logger.trace("Absolute matrix plot saved and closed")

    # Plot 2: Independent Jaccard similarity matrix
    logger.debug("Creating independent Jaccard similarity matrix plot...")
    plt.figure()
    independent_numpy = independent_jaccard.cpu().numpy()
    logger.trace(
        f"Plotting Independent matrix: {independent_numpy.shape}, range=[{independent_numpy.min():.8f}, {independent_numpy.max():.8f}]"
    )

    plt.imshow(independent_numpy, cmap="viridis", aspect="auto")
    plt.colorbar(label="Independent Jaccard Similarity")
    plt.title("Independent Jaccard Similarity Matrix")
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()

    indep_plot_path = os.path.join(FIGURE_DIR, "router_jaccard_matrix_independent.png")
    logger.debug(f"Saving independent matrix plot to: {indep_plot_path}")
    plt.savefig(indep_plot_path)
    plt.close()
    logger.trace("Independent matrix plot saved and closed")

    # Plot 3: Relative Jaccard similarity matrix (red-black-blue colormap)
    logger.debug("Creating relative Jaccard similarity matrix plot...")
    # Red = 0, Black = independent value, Blue = 2x independent value
    relative_jaccard = jaccard / (independent_jaccard + 1e-8)  # Avoid division by zero
    relative_numpy = relative_jaccard.cpu().numpy()
    logger.trace(
        f"Plotting Relative matrix: {relative_numpy.shape}, range=[{relative_numpy.min():.6f}, {relative_numpy.max():.6f}]"
    )

    # Validate relative Jaccard
    assert th.all(relative_jaccard >= 0), "Relative Jaccard must be non-negative"
    logger.trace(
        f"Relative Jaccard statistics: mean={relative_jaccard.mean().item():.4f}, std={relative_jaccard.std().item():.4f}"
    )

    # Create custom colormap: red (0) -> black (1) -> blue (2)
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["red", "black", "blue"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    plt.figure()
    plt.imshow(relative_numpy, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    plt.colorbar(label="Jaccard / Independent Jaccard")
    plt.title(
        "Relative Jaccard Similarity Matrix\n(Red=0, Black=Independent, Blue=2x Independent)"
    )
    plt.xlabel("Expert Index")
    plt.ylabel("Expert Index")
    plt.tight_layout()

    rel_plot_path = os.path.join(FIGURE_DIR, "router_jaccard_matrix_relative.png")
    logger.debug(f"Saving relative matrix plot to: {rel_plot_path}")
    plt.savefig(rel_plot_path)
    plt.close()
    logger.trace("Relative matrix plot saved and closed")

    # Plot 4: Bar plot of sorted Jaccard similarities (upper triangular)
    logger.debug("Creating sorted Jaccard similarities bar plot...")
    plt.figure()
    sorted_jaccard_numpy = sorted_jaccard.cpu().numpy()
    logger.trace(f"Plotting {len(sorted_jaccard_numpy)} sorted Jaccard similarities")
    logger.trace(
        f"Sorted Jaccard range: [{sorted_jaccard_numpy.min():.6f}, {sorted_jaccard_numpy.max():.6f}]"
    )

    plt.bar(range(len(sorted_jaccard_numpy)), sorted_jaccard_numpy)
    plt.xlabel("Expert Pair Rank")
    plt.ylabel("Jaccard Similarity")
    plt.title("Sorted Jaccard Similarities (Upper Triangular)")
    plt.tight_layout()

    sorted_plot_path = os.path.join(FIGURE_DIR, "router_jaccard_bar_sorted.png")
    logger.debug(f"Saving sorted bar plot to: {sorted_plot_path}")
    plt.savefig(sorted_plot_path)
    plt.close()
    logger.trace("Sorted bar plot saved and closed")

    # Plot 5: Bar plot of sorted cross-layer Jaccard similarities
    logger.debug("Creating cross-layer Jaccard similarities bar plot...")
    plt.figure()
    sorted_cross_layer_numpy = sorted_cross_layer.cpu().numpy()
    logger.trace(
        f"Plotting {len(sorted_cross_layer_numpy)} sorted cross-layer similarities"
    )
    logger.trace(
        f"Cross-layer range: [{sorted_cross_layer_numpy.min():.6f}, {sorted_cross_layer_numpy.max():.6f}]"
    )

    plt.bar(range(len(sorted_cross_layer_numpy)), sorted_cross_layer_numpy)
    plt.xlabel("Cross-Layer Pair Rank")
    plt.ylabel("Jaccard Similarity")
    plt.title("Sorted Cross-Layer Jaccard Similarities")
    plt.tight_layout()

    cross_layer_plot_path = os.path.join(
        FIGURE_DIR, "router_jaccard_bar_cross_layer.png"
    )
    logger.debug(f"Saving cross-layer bar plot to: {cross_layer_plot_path}")
    plt.savefig(cross_layer_plot_path)
    plt.close()
    logger.trace("Cross-layer bar plot saved and closed")

    logger.info(f"Figures saved to {FIGURE_DIR}/")
    logger.info("  - router_jaccard_matrix_absolute.png")
    logger.info("  - router_jaccard_matrix_independent.png")
    logger.info("  - router_jaccard_matrix_relative.png")
    logger.info("  - router_jaccard_bar_sorted.png")
    logger.info("  - router_jaccard_bar_cross_layer.png")

    # Validate all files were created
    plot_paths = [
        abs_plot_path,
        indep_plot_path,
        rel_plot_path,
        sorted_plot_path,
        cross_layer_plot_path,
    ]
    for plot_path in plot_paths:
        assert os.path.exists(plot_path), f"Plot file not created: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 0, f"Plot file is empty: {plot_path}"
        logger.trace(f"Plot file {plot_path}: {file_size} bytes")


@arguably.command
def router_jaccard_distance(
    *,
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 5_000,
    context_length: int = 2048,
    batch_size: int = 4096,
    reshuffled_tokens_per_file: int = 100000,
    num_workers: int = 8,
    debug: bool = False,
    max_samples: int = 0,
) -> None:
    """Compute Jaccard similarity between expert activations.

    This script:
    1. Loads router activations using load_activations_and_init_dist
    2. Converts router logits to binary activations via top-k
    3. Computes Jaccard similarity for each pair of experts
    4. Generates matrix visualizations and bar plots

    Args:
        model_name: Name of the model (e.g., "olmoe-i").
        dataset_name: Name of the dataset (e.g., "lmsys").
        tokens_per_file: Number of tokens per activation file.
        context_length: Context length used during activation collection.
        batch_size: Number of samples to process per batch (default: 4096).
        reshuffled_tokens_per_file: Number of tokens per reshuffled file (default: 100000).
        num_workers: Number of worker processes for data loading (default: 8).
        debug: Enable debug logging (default: False).
        max_samples: Maximum number of samples to process. 0 = all samples, >0 = first N samples, <0 = all but last N samples (default: 0).
    """
    asyncio.run(
        _router_jaccard_distance_async(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            context_length=context_length,
            batch_size=batch_size,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            num_workers=num_workers,
            debug=debug,
            max_samples=max_samples,
        )
    )


if __name__ == "__main__":
    arguably.run()
