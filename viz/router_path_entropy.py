"""Analyze the entropy and distribution of routing paths in MoE models."""

import asyncio
from collections import Counter
import os

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import torch as th

from exp.activations import Activations
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR


def compute_entropy(frequencies: th.Tensor) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        frequencies: Tensor of frequencies (will be normalized to probabilities).

    Returns:
        Shannon entropy in bits.
    """
    # Normalize to probabilities
    probabilities = frequencies / frequencies.sum()
    # Filter out zero probabilities to avoid log(0)
    nonzero_probabilities = probabilities[probabilities > 0]

    # Emit warning if we filtered out zero probabilities
    if len(nonzero_probabilities) < len(probabilities):
        logger.warning(
            f"Filtered out {len(probabilities) - len(nonzero_probabilities)} zero probabilities"
        )

    # Compute Shannon entropy
    return -th.sum(nonzero_probabilities * th.log2(nonzero_probabilities)).item()


def compute_gini_coefficient(frequencies: th.Tensor) -> float:
    """Compute Gini coefficient of a distribution.

    Args:
        frequencies: Tensor of frequencies.

    Returns:
        Gini coefficient between 0 (perfect equality) and 1 (perfect inequality).
    """
    sorted_freqs = th.sort(frequencies).values
    n = len(sorted_freqs)
    cumsum = th.cumsum(sorted_freqs, dim=0)

    indices = th.arange(1, n + 1, dtype=sorted_freqs.dtype, device=sorted_freqs.device)
    weighted_sum = th.sum(indices * sorted_freqs)
    total_occurrences = cumsum[-1]
    numerator = 2 * weighted_sum - (n + 1) * total_occurrences
    denominator = n * total_occurrences

    # Add assertions for safety
    assert denominator > 0, "Denominator should be positive for valid frequencies"
    assert numerator >= 0, "Numerator should be non-negative for valid Gini calculation"

    return (numerator / denominator).item()


async def _router_path_entropy_async(experiment_name: str) -> None:
    """Async implementation of router path entropy analysis."""
    logger.info(f"Loading activations for experiment: {experiment_name}")

    # Load activations using the Activations class
    activations = await Activations.load(experiment_name=experiment_name)

    path_counter: Counter[tuple[int, ...]] = Counter()
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None
    total_tokens = 0

    # Iterate through activation batches
    for batch in activations(batch_size=4096):
        router_logits = batch[ActivationKeys.ROUTER_LOGITS]

        if top_k is None:
            top_k = batch["topk"]
            num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
            logger.info(
                f"Router configuration: {num_layers} layers, {num_experts} experts per layer, top-k={top_k}"
            )

        # Get dimensions
        batch_size = router_logits.shape[0]

        # Convert to binary activations (top-k selection)
        top_k_indices = th.topk(router_logits, k=top_k, dim=2).indices

        # For each token in the batch, create a path tuple
        # Path is the concatenation of activated expert indices across all layers
        for token_idx in range(batch_size):
            # Get activated experts for this token across all layers
            # Shape: (num_layers, top_k)
            token_experts = top_k_indices[token_idx]

            # Sort experts within each layer to create a canonical representation
            token_experts_sorted = th.sort(token_experts, dim=1).values

            # Flatten to create path tuple
            path = tuple(token_experts_sorted.flatten().tolist())

            path_counter[path] += 1
            total_tokens += 1

    if top_k is None:
        raise ValueError("No activation data found")

    logger.info("Analysis complete!")
    logger.info(f"Total tokens processed: {total_tokens:,}")
    logger.info(f"Unique paths: {len(path_counter):,}")
    logger.info(f"Path length (experts per path): {num_layers * top_k}")

    # Convert to frequency tensor
    path_frequencies = th.tensor(list(path_counter.values()), dtype=th.float32)

    # Compute metrics
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

    # Compute top-k path coverage
    sorted_frequencies = th.sort(path_frequencies, descending=True).values
    cumulative = th.cumsum(sorted_frequencies, dim=0)
    for k in [10, 100, 1000, 10000]:
        if k <= len(sorted_frequencies):
            coverage = cumulative[k - 1] / total_tokens
            logger.info(
                f"  Top {k} paths cover: {coverage.item() * 100:.2f}% of tokens"
            )

    # Set default figure size
    plt.rcParams["figure.figsize"] = (16, 12)

    # Plot 1: Path frequency distribution (sorted)
    logger.info("Generating visualizations...")
    plt.figure()
    plt.plot(sorted_frequencies.cpu().numpy())
    plt.xlabel("Path rank")
    plt.ylabel("Frequency")
    plt.title(
        f"Path Frequency Distribution\n(Entropy: {entropy:.2f} bits, Gini: {gini:.4f})"
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_path_frequency.png"))
    plt.close()

    # Plot 2: Cumulative path coverage
    plt.figure()
    cumulative_normalized = cumulative / total_tokens
    plt.plot(cumulative_normalized.cpu().numpy())
    plt.xlabel("Number of paths")
    plt.ylabel("Cumulative coverage")
    plt.title("Cumulative Path Coverage")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_path_coverage.png"))
    plt.close()

    # Plot 3: Histogram of path frequencies
    plt.figure()
    path_freq_numpy = path_frequencies.cpu().numpy()
    bins = th.logspace(0, th.log10(path_frequencies.max()), 50).cpu().numpy()
    plt.hist(path_freq_numpy, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Path frequency")
    plt.ylabel("Number of paths")
    plt.title("Distribution of Path Frequencies")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_path_histogram.png"))
    plt.close()

    logger.info(f"Figures saved to {FIGURE_DIR}/")
    logger.info("  - router_path_frequency.png")
    logger.info("  - router_path_coverage.png")
    logger.info("  - router_path_histogram.png")


@arguably.command
def router_path_entropy(experiment_name: str) -> None:
    """Analyze routing path entropy and distribution for an experiment.

    This script:
    1. Loads router activations using the Activations class
    2. Converts router logits to binary activations via top-k
    3. Hashes the complete routing path for each token across all layers
    4. Measures the entropy and non-uniformity of the path distribution

    Args:
        experiment_name: Name of the experiment to analyze.
    """
    asyncio.run(_router_path_entropy_async(experiment_name))


if __name__ == "__main__":
    arguably.run()
