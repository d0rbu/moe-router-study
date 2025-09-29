"""Analyze the entropy and distribution of routing paths in MoE models."""

from collections import Counter
from itertools import count
import os

import arguably
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR


def compute_entropy(frequencies: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        frequencies: Array of frequencies (will be normalized to probabilities).

    Returns:
        Shannon entropy in bits.
    """
    # Normalize to probabilities
    probabilities = frequencies / frequencies.sum()
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    # Compute Shannon entropy
    return -np.sum(probabilities * np.log2(probabilities))


def compute_gini_coefficient(frequencies: np.ndarray) -> float:
    """Compute Gini coefficient of a distribution.

    Args:
        frequencies: Array of frequencies.

    Returns:
        Gini coefficient between 0 (perfect equality) and 1 (perfect inequality).
    """
    sorted_freqs = np.sort(frequencies)
    n = len(sorted_freqs)
    cumsum = np.cumsum(sorted_freqs)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_freqs) - (n + 1) * cumsum[-1]) / (
        n * cumsum[-1]
    )


@arguably.command()
def router_path_entropy(experiment_name: str) -> None:
    """Analyze routing path entropy and distribution for an experiment.

    This script:
    1. Loads router activations from stored .pt files
    2. Converts router logits to binary activations via top-k
    3. Hashes the complete routing path for each token across all layers
    4. Measures the entropy and non-uniformity of the path distribution

    Args:
        experiment_name: Name of the experiment to analyze.
    """
    path_counter: Counter[tuple[int, ...]] = Counter()
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None
    total_tokens = 0

    activations_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)

    print(f"Loading router activations from {activations_dir}...")

    for file_idx in tqdm(count(), desc="Loading router activations"):
        file_path = os.path.join(activations_dir, f"{file_idx}.pt")
        if not os.path.exists(file_path):
            break

        output = th.load(file_path)
        top_k = output["topk"]
        router_logits = output[str(ActivationKeys.ROUTER_LOGITS)]

        # Get dimensions
        batch_size, num_layers_local, num_experts_local = router_logits.shape
        if num_layers is None:
            num_layers = num_layers_local
            num_experts = num_experts_local
        else:
            assert num_layers == num_layers_local
            assert num_experts == num_experts_local

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
        raise ValueError("No data files found")

    print("\nAnalysis complete!")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Unique paths: {len(path_counter):,}")
    print(f"Path length (experts per path): {num_layers * top_k}")

    # Convert to frequency array
    path_frequencies = np.array(list(path_counter.values()))

    # Compute metrics
    entropy = compute_entropy(path_frequencies)
    max_entropy = np.log2(total_tokens)  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    gini = compute_gini_coefficient(path_frequencies)

    print("\nEntropy metrics:")
    print(f"  Shannon entropy: {entropy:.2f} bits")
    print(f"  Maximum entropy: {max_entropy:.2f} bits")
    print(f"  Normalized entropy: {normalized_entropy:.4f}")
    print(f"  Gini coefficient: {gini:.4f}")

    # Compute top-k path coverage
    sorted_frequencies = sorted(path_frequencies, reverse=True)
    cumulative = np.cumsum(sorted_frequencies)
    for k in [10, 100, 1000, 10000]:
        if k <= len(sorted_frequencies):
            coverage = cumulative[k - 1] / total_tokens
            print(f"  Top {k} paths cover: {coverage * 100:.2f}% of tokens")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (16, 12)

    # Plot 1: Path frequency distribution (sorted)
    print("\nGenerating visualizations...")
    plt.figure()
    sorted_frequencies_arr = np.array(sorted_frequencies)
    plt.plot(sorted_frequencies_arr)
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
    plt.plot(cumulative_normalized)
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
    bins = np.logspace(0, np.log10(max(path_frequencies)), 50)
    plt.hist(path_frequencies, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Path frequency")
    plt.ylabel("Number of paths")
    plt.title("Distribution of Path Frequencies")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "router_path_histogram.png"))
    plt.close()

    print(f"Figures saved to {FIGURE_DIR}/")
    print("  - router_path_frequency.png")
    print("  - router_path_coverage.png")
    print("  - router_path_histogram.png")


if __name__ == "__main__":
    arguably.run()
