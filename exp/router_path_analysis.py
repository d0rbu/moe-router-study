"""Router path analysis for investigating path collapse in MoE models.

This module implements experiments to analyze router path collapse by:
1. Hashing router paths for each token and measuring entropy/non-uniformity
2. Analyzing the distribution of routing patterns across the model
"""

from collections import Counter
import hashlib
from itertools import count as itertools_count
import os

import arguably
import numpy as np
from scipy.stats import entropy
import torch as th
from tqdm import tqdm

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR


def hash_path(path_indices: th.Tensor) -> str:
    """Hash a router path (sequence of activated experts across layers).

    Args:
        path_indices: Tensor of shape (num_layers, top_k) containing expert indices

    Returns:
        Hexadecimal hash string representing the path
    """
    # Convert to numpy and then to bytes for consistent hashing
    path_bytes = path_indices.cpu().numpy().tobytes()
    return hashlib.md5(path_bytes).hexdigest()


def compute_path_entropy(path_counts: Counter) -> dict:
    """Compute entropy and other statistics for path distribution.

    Args:
        path_counts: Counter mapping path hashes to their frequencies

    Returns:
        Dictionary containing entropy and distribution statistics
    """
    total_paths = sum(path_counts.values())
    unique_paths = len(path_counts)

    # Convert to probability distribution
    probabilities = np.array(list(path_counts.values())) / total_paths

    # Compute entropy (in bits)
    path_entropy = entropy(probabilities, base=2)

    # Compute maximum possible entropy (uniform distribution)
    max_entropy = np.log2(unique_paths) if unique_paths > 1 else 0

    # Compute normalized entropy (0 = completely collapsed, 1 = uniform)
    normalized_entropy = path_entropy / max_entropy if max_entropy > 0 else 0

    # Get top paths
    most_common_paths = path_counts.most_common(10)
    top_path_frequencies = [count for _, count in most_common_paths]

    # Compute concentration metrics
    top_1_concentration = (
        most_common_paths[0][1] / total_paths if most_common_paths else 0
    )
    top_10_concentration = sum(top_path_frequencies) / total_paths

    return {
        "total_tokens": total_paths,
        "unique_paths": unique_paths,
        "entropy_bits": path_entropy,
        "max_entropy_bits": max_entropy,
        "normalized_entropy": normalized_entropy,
        "top_1_concentration": top_1_concentration,
        "top_10_concentration": top_10_concentration,
        "most_common_paths": most_common_paths,
        "gini_coefficient": compute_gini_coefficient(probabilities),
    }


def compute_gini_coefficient(probabilities: np.ndarray) -> float:
    """Compute Gini coefficient for measuring inequality in path distribution.

    Args:
        probabilities: Array of path probabilities

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    if len(probabilities) <= 1:
        return 0.0

    # Sort probabilities
    sorted_probs = np.sort(probabilities)
    n = len(sorted_probs)

    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_probs)
    return (n + 1 - 2 * np.sum(cumsum)) / (n * np.sum(sorted_probs))


@arguably.command()
def analyze_router_paths(
    experiment_name: str,
    output_file: str | None = None,
    max_files: int | None = None,
) -> None:
    """Analyze router path collapse by hashing paths and measuring entropy.

    Args:
        experiment_name: Name of the experiment containing router activations
        output_file: Optional file to save results (defaults to stdout)
        max_files: Maximum number of activation files to process (for testing)
    """
    print(f"Analyzing router paths for experiment: {experiment_name}")

    path_counter = Counter()
    total_tokens = 0
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None

    # Process activation files
    files_processed = 0
    for file_idx in tqdm(itertools_count(), desc="Processing activation files"):
        if max_files is not None and files_processed >= max_files:
            break

        file_path = os.path.join(
            OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME, f"{file_idx}.pt"
        )
        if not os.path.exists(file_path):
            break

        # Load router logits
        try:
            output = th.load(file_path, map_location="cpu")
            if "router_logits" not in output:
                print(f"Warning: No router_logits found in {file_path}")
                continue

            router_logits = output["router_logits"]
            top_k = output.get("topk", top_k)

            # Get dimensions
            batch_size, num_layers, num_experts = router_logits.shape

            # Get top-k activated experts for each token
            # Shape: (batch_size, num_layers, top_k)
            top_k_indices = th.topk(router_logits, k=top_k, dim=2).indices

            # Process each token in the batch
            for token_idx in range(batch_size):
                # Get path for this token: (num_layers, top_k)
                token_path = top_k_indices[token_idx]

                # Hash the path
                path_hash = hash_path(token_path)
                path_counter[path_hash] += 1
                total_tokens += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        files_processed += 1

    if total_tokens == 0:
        print("No router logits found in the experiment data!")
        return

    print(f"Processed {files_processed} files with {total_tokens} tokens")
    print(
        f"Model configuration: {num_layers} layers, {num_experts} experts, top-{top_k} routing"
    )

    # Compute path statistics
    stats = compute_path_entropy(path_counter)

    # Print results
    print("\n" + "=" * 60)
    print("ROUTER PATH COLLAPSE ANALYSIS")
    print("=" * 60)
    print(f"Total tokens analyzed: {stats['total_tokens']:,}")
    print(f"Unique routing paths: {stats['unique_paths']:,}")
    print(f"Path entropy: {stats['entropy_bits']:.3f} bits")
    print(f"Maximum possible entropy: {stats['max_entropy_bits']:.3f} bits")
    print(f"Normalized entropy: {stats['normalized_entropy']:.3f}")
    print(f"Gini coefficient: {stats['gini_coefficient']:.3f}")
    print(f"Top-1 path concentration: {stats['top_1_concentration']:.3f}")
    print(f"Top-10 paths concentration: {stats['top_10_concentration']:.3f}")

    print("\nMost common routing paths:")
    for i, (path_hash, count) in enumerate(stats["most_common_paths"], 1):
        percentage = count / stats["total_tokens"] * 100
        print(f"  {i:2d}. {path_hash[:12]}... : {count:,} tokens ({percentage:.2f}%)")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if stats["normalized_entropy"] < 0.3:
        collapse_level = "SEVERE"
    elif stats["normalized_entropy"] < 0.6:
        collapse_level = "MODERATE"
    elif stats["normalized_entropy"] < 0.8:
        collapse_level = "MILD"
    else:
        collapse_level = "MINIMAL"

    print(f"Path collapse level: {collapse_level}")
    print(
        f"- Normalized entropy of {stats['normalized_entropy']:.3f} indicates", end=" "
    )

    if collapse_level == "SEVERE":
        print("severe path collapse with most tokens using very few routing patterns.")
    elif collapse_level == "MODERATE":
        print("moderate path collapse with noticeable concentration in popular paths.")
    elif collapse_level == "MILD":
        print("mild path collapse with some concentration but reasonable diversity.")
    else:
        print("minimal path collapse with relatively uniform path distribution.")

    print(
        f"- Top-1 path captures {stats['top_1_concentration'] * 100:.1f}% of all tokens"
    )
    print(f"- Gini coefficient of {stats['gini_coefficient']:.3f} shows", end=" ")

    if stats["gini_coefficient"] > 0.8:
        print("very high inequality in path usage.")
    elif stats["gini_coefficient"] > 0.6:
        print("high inequality in path usage.")
    elif stats["gini_coefficient"] > 0.4:
        print("moderate inequality in path usage.")
    else:
        print("relatively equal path usage.")

    # Save results if requested
    if output_file:
        print(f"\nSaving detailed results to {output_file}")
        results = {
            "experiment_name": experiment_name,
            "model_config": {
                "num_layers": num_layers,
                "num_experts": num_experts,
                "top_k": top_k,
            },
            "statistics": stats,
            "path_counts": dict(path_counter),
        }
        th.save(results, output_file)


if __name__ == "__main__":
    arguably.run()
