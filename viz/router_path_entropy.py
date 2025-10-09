"""Analyze the entropy and distribution of routing paths in MoE models."""

import asyncio
from collections import Counter
import os

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import torch as th

from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR


def compute_entropy(frequencies: th.Tensor) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        frequencies: Tensor of frequencies (will be normalized to probabilities).

    Returns:
        Shannon entropy in bits.
    """
    logger.debug(f"Computing entropy for {len(frequencies)} frequencies")
    logger.trace(
        f"Frequency tensor shape: {frequencies.shape}, dtype: {frequencies.dtype}"
    )

    assert frequencies.dim() == 1, f"Expected 1D tensor, got {frequencies.dim()}D"
    assert len(frequencies) > 0, "Cannot compute entropy of empty tensor"
    assert th.all(frequencies >= 0), "Frequencies must be non-negative"
    assert th.any(frequencies > 0), "At least one frequency must be positive"

    total_freq = frequencies.sum()
    logger.trace(f"Total frequency sum: {total_freq.item()}")
    assert total_freq > 0, "Total frequency must be positive"

    # Normalize to probabilities
    probabilities = frequencies / total_freq
    logger.trace(f"Probability sum: {probabilities.sum().item():.10f} (should be ~1.0)")
    assert th.allclose(probabilities.sum(), th.tensor(1.0)), (
        f"Probabilities don't sum to 1: {probabilities.sum().item()}"
    )

    # Filter out zero probabilities to avoid log(0)
    nonzero_mask = probabilities > 0
    nonzero_probabilities = probabilities[nonzero_mask]
    zero_count = len(probabilities) - len(nonzero_probabilities)

    logger.debug(
        f"Non-zero probabilities: {len(nonzero_probabilities)}/{len(probabilities)}"
    )
    logger.trace(f"Zero probability count: {zero_count}")

    # Emit warning if we filtered out zero probabilities
    if zero_count > 0:
        logger.warning(f"Filtered out {zero_count} zero probabilities")

    # Compute Shannon entropy
    entropy = -th.sum(nonzero_probabilities * th.log2(nonzero_probabilities)).item()
    logger.debug(f"Computed entropy: {entropy:.6f} bits")

    # Validation
    assert entropy >= 0, f"Entropy should be non-negative, got {entropy}"
    max_entropy = th.log2(th.tensor(float(len(nonzero_probabilities)))).item()
    assert entropy <= max_entropy * (1 + 1e-4), (
        f"Entropy {entropy} exceeds maximum {max_entropy}"
    )

    return entropy


def compute_gini_coefficient(frequencies: th.Tensor) -> float:
    """Compute Gini coefficient of a distribution.

    Args:
        frequencies: Tensor of frequencies.

    Returns:
        Gini coefficient between 0 (perfect equality) and 1 (perfect inequality).
    """
    logger.debug(f"Computing Gini coefficient for {len(frequencies)} frequencies")
    logger.trace(
        f"Frequency tensor shape: {frequencies.shape}, dtype: {frequencies.dtype}"
    )

    # Input validation
    assert frequencies.dim() == 1, f"Expected 1D tensor, got {frequencies.dim()}D"
    assert len(frequencies) > 0, "Cannot compute Gini coefficient of empty tensor"
    assert th.all(frequencies >= 0), "Frequencies must be non-negative"
    assert th.any(frequencies > 0), "At least one frequency must be positive"

    sorted_freqs = th.sort(frequencies).values
    n = len(sorted_freqs)
    logger.trace(
        f"Sorted frequencies: min={sorted_freqs.min().item():.6f}, max={sorted_freqs.max().item():.6f}"
    )

    cumsum = th.cumsum(sorted_freqs, dim=0)
    logger.trace(f"Cumulative sum: final={cumsum[-1].item():.6f}")

    indices = th.arange(1, n + 1, dtype=sorted_freqs.dtype, device=sorted_freqs.device)
    weighted_sum = th.sum(indices * sorted_freqs)
    total_occurrences = cumsum[-1]
    numerator = 2 * weighted_sum - (n + 1) * total_occurrences
    denominator = n * total_occurrences

    logger.trace(
        f"Gini calculation: n={n}, weighted_sum={weighted_sum.item():.6f}, total={total_occurrences.item():.6f}"
    )
    logger.trace(
        f"Numerator: {numerator.item():.6f}, Denominator: {denominator.item():.6f}"
    )

    # Add assertions for safety
    assert denominator > 0, (
        f"Denominator should be positive for valid frequencies, got {denominator.item()}"
    )
    assert numerator >= 0, (
        f"Numerator should be non-negative for valid Gini calculation, got {numerator.item()}"
    )
    assert denominator > numerator, (
        f"Denominator {denominator.item()} should be > numerator {numerator.item()}"
    )

    gini = (numerator / denominator).item()
    logger.debug(f"Computed Gini coefficient: {gini:.6f}")

    # Validation
    assert 0 <= gini <= 1, f"Gini coefficient should be in [0,1], got {gini}"

    return gini


async def _router_path_entropy_async(
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
    """Async implementation of router path entropy analysis."""
    logger.info(f"Loading activations for model: {model_name}, dataset: {dataset_name}")
    logger.debug(f"Batch size: {batch_size}")

    # Input validation
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
        activation_dims,
        _gpu_process_group,
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

    path_counter: Counter[tuple[int, ...]] = Counter()
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None
    total_tokens = 0
    batch_count = 0

    logger.debug("Starting batch processing...")

    # Add assertion for non-negative max_samples
    assert max_samples >= 0, f"max_samples must be non-negative, got {max_samples}"

    if max_samples > 0:
        logger.info(f"Processing first {max_samples:,} samples")
    else:
        logger.info("Processing all available samples")

    # Iterate through activation batches with max_samples limit
    for batch in activations(batch_size=batch_size, max_samples=max_samples):
        batch_count += 1
        logger.trace(f"Processing batch {batch_count}")

        router_logits = batch[ActivationKeys.ROUTER_LOGITS]
        logger.trace(
            f"Router logits shape: {router_logits.shape}, dtype: {router_logits.dtype}"
        )

        # Validate router logits
        assert router_logits.dim() == 3, (
            f"Expected 3D router logits, got {router_logits.dim()}D"
        )
        assert router_logits.shape[0] > 0, "Batch size must be positive"
        assert router_logits.shape[1] > 0, "Number of layers must be positive"
        assert router_logits.shape[2] > 0, "Number of experts must be positive"

        if top_k is None:
            top_k = batch["topk"]
            num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
            logger.info(
                f"Router configuration: {num_layers} layers, {num_experts} experts per layer, top-k={top_k}"
            )
            logger.debug(f"Total experts across all layers: {num_layers * num_experts}")

            # Validate configuration
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

        # Get dimensions
        current_batch_size = router_logits.shape[0]
        logger.trace(f"Current batch size: {current_batch_size}")

        # Convert to binary activations (top-k selection)
        logger.trace("Computing top-k indices...")
        top_k_indices = th.topk(router_logits, k=top_k, dim=2).indices
        logger.trace(f"Top-k indices shape: {top_k_indices.shape}")

        # Validate top-k indices
        assert top_k_indices.shape == (current_batch_size, num_layers, top_k), (
            f"Unexpected top-k indices shape: {top_k_indices.shape}"
        )
        assert th.all(top_k_indices >= 0), "Expert indices must be non-negative"
        assert th.all(top_k_indices < num_experts), (
            f"Expert indices must be < {num_experts}"
        )

        # For each token in the batch, create a path tuple
        # Path is the concatenation of activated expert indices across all layers
        logger.trace(f"Processing {current_batch_size} tokens in batch...")

        for token_idx in range(current_batch_size):
            # Get activated experts for this token across all layers
            # Shape: (num_layers, top_k)
            token_experts = top_k_indices[token_idx]
            logger.trace(f"Token {token_idx} experts before sorting: {token_experts}")

            # Sort experts within each layer to create a canonical representation
            token_experts_sorted = th.sort(token_experts, dim=1).values
            logger.trace(
                f"Token {token_idx} experts after sorting: {token_experts_sorted}"
            )

            # Flatten to create path tuple
            path = tuple(token_experts_sorted.flatten().tolist())
            logger.trace(f"Token {token_idx} path: {path}")

            # Validate path
            assert len(path) == num_layers * top_k, (
                f"Path length {len(path)} != expected {num_layers * top_k}"
            )
            assert all(isinstance(x, int) for x in path), (
                "Path elements must be integers"
            )
            assert all(0 <= x < num_experts for x in path), (
                f"Path elements must be in [0, {num_experts})"
            )

            path_counter[path] += 1
            total_tokens += 1

        logger.debug(
            f"Batch {batch_count} complete: {current_batch_size} tokens processed, {len(path_counter)} unique paths so far"
        )

    if top_k is None:
        raise ValueError("No activation data found")

    logger.info("Analysis complete!")
    logger.info(f"Total batches processed: {batch_count}")
    logger.info(f"Total tokens processed: {total_tokens:,}")
    logger.info(f"Unique paths: {len(path_counter):,}")
    logger.info(f"Path length (experts per path): {num_layers * top_k}")

    # Validate final state
    assert total_tokens > 0, "No tokens were processed"
    assert len(path_counter) > 0, "No paths were found"
    assert total_tokens == sum(path_counter.values()), (
        "Token count mismatch in path counter"
    )

    # Convert to frequency tensor
    logger.debug("Converting path counter to frequency tensor...")
    path_frequencies = th.tensor(list(path_counter.values()), dtype=th.float32)
    logger.trace(
        f"Path frequencies shape: {path_frequencies.shape}, dtype: {path_frequencies.dtype}"
    )
    logger.trace(
        f"Frequency sum: {path_frequencies.sum().item():.6f} (should equal {total_tokens})"
    )

    # Validate frequency tensor
    assert path_frequencies.sum().item() == total_tokens, (
        f"Frequency sum {path_frequencies.sum().item()} != total tokens {total_tokens}"
    )
    assert th.all(path_frequencies > 0), "All path frequencies must be positive"
    assert len(path_frequencies) == len(path_counter), (
        "Frequency tensor length mismatch"
    )

    # Compute metrics
    logger.debug("Computing entropy metrics...")
    entropy = compute_entropy(path_frequencies)
    max_entropy = th.log2(
        th.tensor(float(total_tokens))
    ).item()  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    gini = compute_gini_coefficient(path_frequencies)

    logger.info("Entropy metrics:")
    logger.info(f"  Shannon entropy: {entropy:.2f} bits")
    logger.info(f"  Maximum entropy: {max_entropy:.2f} bits")
    logger.info(f"  Normalized entropy: {normalized_entropy:.4f}")
    logger.info(f"  Gini coefficient: {gini:.4f}")

    # Validate metrics
    assert 0 <= entropy <= max_entropy, (
        f"Entropy {entropy} not in valid range [0, {max_entropy}]"
    )
    assert 0 <= normalized_entropy <= 1, (
        f"Normalized entropy {normalized_entropy} not in [0,1]"
    )
    assert 0 <= gini <= 1, f"Gini coefficient {gini} not in [0,1]"

    # Compute top-k path coverage
    logger.debug("Computing path coverage statistics...")
    sorted_frequencies = th.sort(path_frequencies, descending=True).values
    cumulative = th.cumsum(sorted_frequencies, dim=0)

    # Validate sorted frequencies
    assert th.all(sorted_frequencies[:-1] >= sorted_frequencies[1:]), (
        "Frequencies not properly sorted"
    )
    assert cumulative[-1].item() == total_tokens, "Cumulative sum mismatch"

    for k in [10, 100, 1000, 10000]:
        if k <= len(sorted_frequencies):
            coverage = cumulative[k - 1] / total_tokens
            logger.info(
                f"  Top {k} paths cover: {coverage.item() * 100:.2f}% of tokens"
            )
            logger.trace(f"Top {k} coverage: {coverage.item():.6f}")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (16, 12)

    # Plot 1: Path frequency distribution (sorted)
    logger.info("Generating visualizations...")

    # Ensure output directory exists
    os.makedirs(FIGURE_DIR, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {FIGURE_DIR}")

    logger.debug("Creating path frequency distribution plot...")
    plt.figure()
    sorted_freq_numpy = sorted_frequencies.cpu().numpy()
    logger.trace(f"Plotting {len(sorted_freq_numpy)} sorted frequencies")
    logger.trace(
        f"Frequency range: min={sorted_freq_numpy.min():.2f}, max={sorted_freq_numpy.max():.2f}"
    )

    plt.plot(sorted_freq_numpy)
    plt.xlabel("Path rank")
    plt.ylabel("Frequency")
    plt.title(
        f"Path Frequency Distribution\n(Entropy: {entropy:.2f} bits, Gini: {gini:.4f})"
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    freq_plot_path = os.path.join(FIGURE_DIR, "router_path_frequency.png")
    logger.debug(f"Saving frequency plot to: {freq_plot_path}")
    plt.savefig(freq_plot_path)
    plt.close()
    logger.trace("Frequency plot saved and closed")

    # Plot 2: Cumulative path coverage
    logger.debug("Creating cumulative path coverage plot...")
    plt.figure()
    cumulative_normalized = cumulative / total_tokens
    cum_norm_numpy = cumulative_normalized.cpu().numpy()
    logger.trace(f"Plotting cumulative coverage: {len(cum_norm_numpy)} points")
    logger.trace(
        f"Coverage range: min={cum_norm_numpy.min():.6f}, max={cum_norm_numpy.max():.6f}"
    )

    # Validate cumulative coverage
    assert th.allclose(th.tensor(cum_norm_numpy[-1]), th.tensor(1.0)), (
        f"Final coverage {cum_norm_numpy[-1]} != 1.0"
    )
    assert th.all(cumulative_normalized >= 0), (
        "Cumulative coverage must be non-negative"
    )
    assert th.all(cumulative_normalized[1:] >= cumulative_normalized[:-1]), (
        "Cumulative coverage must be non-decreasing"
    )

    plt.plot(cum_norm_numpy)
    plt.xlabel("Number of paths")
    plt.ylabel("Cumulative coverage")
    plt.title("Cumulative Path Coverage")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    coverage_plot_path = os.path.join(FIGURE_DIR, "router_path_coverage.png")
    logger.debug(f"Saving coverage plot to: {coverage_plot_path}")
    plt.savefig(coverage_plot_path)
    plt.close()
    logger.trace("Coverage plot saved and closed")

    # Plot 3: Histogram of path frequencies
    logger.debug("Creating path frequency histogram...")
    plt.figure()
    path_freq_numpy = path_frequencies.cpu().numpy()
    logger.trace(f"Creating histogram for {len(path_freq_numpy)} frequencies")
    logger.trace(
        f"Frequency range: min={path_freq_numpy.min():.2f}, max={path_freq_numpy.max():.2f}"
    )

    bins = th.logspace(0, th.log10(path_frequencies.max()), 50).cpu().numpy()
    logger.trace(f"Using {len(bins)} bins for histogram")

    hist_counts, hist_bins, _ = plt.hist(
        path_freq_numpy, bins=bins, edgecolor="black", alpha=0.7
    )
    logger.trace(f"Histogram: {len(hist_counts)} bins with counts {hist_counts}")

    plt.xlabel("Path frequency")
    plt.ylabel("Number of paths")
    plt.title("Distribution of Path Frequencies")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    hist_plot_path = os.path.join(FIGURE_DIR, "router_path_histogram.png")
    logger.debug(f"Saving histogram plot to: {hist_plot_path}")
    plt.savefig(hist_plot_path)
    plt.close()
    logger.trace("Histogram plot saved and closed")

    logger.info(f"Figures saved to {FIGURE_DIR}/")
    logger.info("  - router_path_frequency.png")
    logger.info("  - router_path_coverage.png")
    logger.info("  - router_path_histogram.png")

    # Validate all files were created
    for plot_path in [freq_plot_path, coverage_plot_path, hist_plot_path]:
        assert os.path.exists(plot_path), f"Plot file not created: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 0, f"Plot file is empty: {plot_path}"
        logger.trace(f"Plot file {plot_path}: {file_size} bytes")


@arguably.command
def router_path_entropy(
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
    """Analyze routing path entropy and distribution for an experiment.

    This script:
    1. Loads router activations using load_activations_and_init_dist
    2. Converts router logits to binary activations via top-k
    3. Hashes the complete routing path for each token across all layers
    4. Measures the entropy and non-uniformity of the path distribution

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
        _router_path_entropy_async(
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
