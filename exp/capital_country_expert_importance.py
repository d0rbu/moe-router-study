"""
Compute expert importance scores from intervention experiment results.

This script reads the output files from capital_country_expert_diff.py and computes
expert importance scores based on when experts get activated/deactivated and the
corresponding forgetfulness changes.

For each expert (layer, expert), we:
1. Find the last alpha where the difference is zero (expert hasn't changed yet)
2. If that's not the last alpha, compute importance as:
   difference_mask_value * (pre_forgetfulness - post_forgetfulness)
   where pre_forgetfulness is at the last zero-difference alpha,
   and post_forgetfulness is at the next alpha where the expert changes.

Usage:
    uv run python -m exp.capital_country_expert_importance \\
        --input-dir "out/capital_country_expert_diff/south_korea" \\
        --output-file "out/capital_country_expert_importance/south_korea.pt"
"""

from pathlib import Path
import sys

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import torch as th


def compute_expert_importance(
    input_dir: Path,
) -> tuple[th.Tensor, dict[str, th.Tensor]]:
    """
    Compute expert importance scores from saved mask files.

    Args:
        input_dir: Directory containing the .pt files for different alphas

    Returns:
        Tuple of (importance_scores, metadata) where:
        - importance_scores: (L, E) tensor of expert importance scores
        - metadata: Dictionary with additional information
    """
    # Load all alpha files
    alpha_files = sorted(
        input_dir.glob("*.pt"), key=lambda p: float(p.stem.replace("_", "."))
    )

    if not alpha_files:
        raise ValueError(f"No .pt files found in {input_dir}")

    logger.info(f"Found {len(alpha_files)} alpha files")

    # Load all data
    alphas: list[float] = []
    pre_masks: list[th.Tensor] = []
    post_masks: list[th.Tensor] = []
    forgetfulness_scores: list[float] = []
    target_country = "Unknown"

    for alpha_file in alpha_files:
        data = th.load(alpha_file, map_location="cpu")
        alphas.append(data["alpha"])
        pre_masks.append(data["pre_intervention"])
        post_masks.append(data["post_intervention"])
        forgetfulness_scores.append(data["forgetfulness"])
        # Get target_country from first file (should be same for all)
        if target_country == "Unknown":
            target_country = data.get("target_country", "Unknown")

    logger.info(f"Loaded data for alphas: {alphas}")
    logger.info(f"Target country: {target_country}")

    # Get dimensions
    num_layers, num_experts = pre_masks[0].shape
    logger.info(f"Dimensions: {num_layers} layers, {num_experts} experts")

    # Compute difference masks for each alpha
    diff_masks: list[th.Tensor] = []
    for pre_mask, post_mask in zip(pre_masks, post_masks, strict=True):
        diff_mask = post_mask - pre_mask
        diff_masks.append(diff_mask)

    # Initialize importance scores to zero
    importance_scores = th.zeros((num_layers, num_experts), dtype=th.float32)

    # Convert forgetfulness scores to tensor for easier computation
    forgetfulness_tensor = th.tensor(forgetfulness_scores, dtype=th.float32)

    # Convert diff_masks to tensor for easier computation
    diff_masks_tensor = th.stack(diff_masks, dim=0)  # (A, L, E)
    diff_changes = diff_masks_tensor[1:] - diff_masks_tensor[:-1]  # (A-1, L, E)
    forgetfulness_changes = (
        forgetfulness_tensor[:-1] - forgetfulness_tensor[1:]
    ).expand(-1, 1, 1)  # (A-1, 1, 1)]

    importance_scores = (diff_changes * forgetfulness_changes).sum(dim=0)  # (L, E)

    metadata = {
        "target_country": target_country,
        "alphas": alphas,
        "forgetfulness_scores": forgetfulness_scores,
    }

    return importance_scores, metadata


def visualize_importance(
    importance_scores: th.Tensor,
    target_country: str,
    output_path: Path,
) -> None:
    """
    Visualize expert importance scores.

    Args:
        importance_scores: (L, E) tensor of importance scores
        target_country: Name of target country
        output_path: Path to save the visualization
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))

    # Use a diverging colormap centered at zero
    vmax = th.abs(importance_scores).max().item()
    vmin = -vmax if vmax > 0 else 0

    im = ax.imshow(
        importance_scores.cpu().float().numpy(),
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Expert Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.set_title(
        f"Expert Importance Scores\n{target_country}\n"
        "(Positive: deactivation increases forgetfulness, "
        "Negative: activation increases forgetfulness)",
        fontsize=14,
    )
    plt.colorbar(im, ax=ax, label="Importance Score")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


@arguably.command()
def capital_country_expert_importance(
    *,
    input_dir: str = "out/capital_country_expert_diff/south_korea",
    output_file: str = "out/capital_country_expert_importance/south_korea.pt",
    output_viz: str = "fig/capital_country_expert_importance/south_korea.png",
    log_level: str = "INFO",
) -> None:
    """
    Compute expert importance scores from intervention experiment results.

    Args:
        input_dir: Directory containing the .pt files for different alphas
        output_file: Path to save the importance scores tensor
        output_viz: Optional path to save visualization (default: output_file with .png)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(
        f"Running capital_country_expert_importance with log level: {log_level}"
    )

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    # Compute importance scores
    logger.info("=" * 80)
    logger.info("Computing expert importance scores")
    logger.info("=" * 80)

    importance_scores, metadata = compute_expert_importance(input_path)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    th.save(
        {
            "importance_scores": importance_scores,
            **metadata,
        },
        output_path,
    )
    logger.info(f"Saved importance scores to {output_path}")

    # Create visualization
    if output_viz is None:
        output_viz = str(output_path).replace(".pt", ".png")

    viz_path = Path(output_viz)
    visualize_importance(importance_scores, metadata["target_country"], viz_path)

    # Print summary
    logger.info("=" * 80)
    logger.info("COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target country: {metadata['target_country']}")
    logger.info(f"Importance scores shape: {importance_scores.shape}")
    logger.info(f"Min importance: {importance_scores.min().item():.4f}")
    logger.info(f"Max importance: {importance_scores.max().item():.4f}")
    logger.info(
        f"Mean absolute importance: {th.abs(importance_scores).mean().item():.4f}"
    )
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Visualization saved to: {viz_path}")


if __name__ == "__main__":
    arguably.run()
