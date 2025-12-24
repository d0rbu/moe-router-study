"""
Analyze expert importance by testing interventions on subsets of experts.

This script:
1. Loads expert importance scores
2. Filters experts above a threshold
3. Sorts experts by importance
4. Tests interventions on different subsets:
   - Single experts (most important, 2nd, 3rd, etc.)
   - Pairs (most+2nd, most+3rd, etc.) - only pairs to avoid combinatorial explosion
   - Top-k (top 1, top 2, top 3, etc.)
5. For each subset, creates an intervention path (all zeros except 1 at selected experts)
6. Runs intervention experiments and generates forgetfulness graphs

Usage:
    uv run python -m exp.capital_country_expert_subset_analysis \\
        --importance-file "out/capital_country_expert_importance/south_korea.pt" \\
        --target-country "South Korea" \\
        --threshold 0.1
"""

from itertools import batched
from pathlib import Path
import sys

import arguably
from loguru import logger
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.dtype import get_dtype
from core.model import get_model_config
from exp.capital_country import (
    COUNTRY_TO_CAPITAL,
    ExperimentResults,
    ExperimentType,
    InterventionMetric,
    InterventionResult,
    get_all_prompts,
    run_intervention,
)
from viz import FIGURE_DIR


def load_and_filter_experts(
    importance_file: Path,
    importance_threshold: float = 0.1,
) -> tuple[th.Tensor, th.Tensor, str]:
    """
    Load expert importance scores and filter/sort by importance.

    Args:
        importance_file: Path to the importance scores .pt file
        importance_threshold: Minimum importance score to keep

    Returns:
        Tuple of (expert_indices, expert_importances, target_country) where expert_indices is (N, 2) and expert_importances is (N,)
        and target_country is the country for which the importance scores were computed
    """
    data = th.load(importance_file, map_location="cpu")
    importance_scores = data["importance_scores"]  # (L, E)
    target_country = data["target_country"]

    logger.info(f"Loaded importance scores: {importance_scores.shape}")
    logger.info(f"Target country: {target_country}")

    experts_indices_unsorted = th.nonzero(
        importance_scores >= importance_threshold, as_tuple=False
    )  # (N, 2)
    # Extract importance values using advanced indexing
    expert_importances_unsorted = importance_scores[
        experts_indices_unsorted[:, 0], experts_indices_unsorted[:, 1]
    ]  # (N,)

    expert_importances, expert_importance_indices = th.sort(
        expert_importances_unsorted, descending=True
    )
    expert_indices = experts_indices_unsorted[expert_importance_indices]  # (N, 2)

    return expert_indices, expert_importances, target_country


def create_intervention_path(
    selected_expert_indices: th.Tensor,  # (N, 2) where each row is [layer_idx, expert_idx]
    num_layers: int,
    num_experts: int,
) -> th.Tensor:
    """
    Create an intervention path with 1.0 at selected experts, 0.0 elsewhere.

    Args:
        selected_expert_indices: Tensor of shape (N, 2) where each row is [layer_idx, expert_idx]
        num_layers: Number of layers
        num_experts: Number of experts per layer

    Returns:
        Intervention path tensor of shape (L, E)
    """
    path = th.zeros((num_layers, num_experts), dtype=th.float32)
    if len(selected_expert_indices) > 0:
        layer_indices = selected_expert_indices[:, 0].long()
        expert_indices = selected_expert_indices[:, 1].long()
        path[layer_indices, expert_indices] = 1.0
    return path


def generate_expert_subsets(
    expert_indices: th.Tensor,  # (N, 2) where each row is [layer_idx, expert_idx]
) -> dict[str, list[th.Tensor]]:
    """
    Generate different types of expert subsets for testing.

    Args:
        expert_indices: Tensor of shape (N, 2) with experts sorted by importance (most important first)

    Returns:
        Dictionary mapping subset type to list of expert index tensors, each of shape (M, 2)
    """
    subsets: dict[str, list[th.Tensor]] = {
        "single": [],
        "pairs": [],
        "top_k": [],
    }

    num_experts = len(expert_indices)

    # Single experts: most important, 2nd, 3rd, etc.
    for i in range(num_experts):
        subsets["single"].append(expert_indices[i : i + 1])  # Keep shape (1, 2)

    # Pairs: (most, 2nd), (most, 3rd), etc.
    if num_experts >= 2:
        for i in range(1, num_experts):
            # Stack first expert with i-th expert
            pair = th.stack([expert_indices[0], expert_indices[i]], dim=0)  # (2, 2)
            subsets["pairs"].append(pair)

    # Top-k: top 1, top 2, top 3, etc.
    for k in range(1, num_experts + 1):
        subsets["top_k"].append(expert_indices[:k])  # (k, 2)

    logger.info(f"Generated {len(subsets['single'])} single expert subsets")
    logger.info(f"Generated {len(subsets['pairs'])} pair subsets")
    logger.info(f"Generated {len(subsets['top_k'])} top-k subsets")

    return subsets


@th.no_grad()
def run_subset_intervention_experiment(
    model: StandardizedTransformer,
    target_country: str,
    intervention_path: th.Tensor,  # (L, E)
    alphas: set[float],
    top_k: int,
    batch_size: int = 64,
) -> ExperimentResults:
    """
    Run intervention experiment for a specific intervention path.

    Args:
        model: The MoE model
        target_country: Target country for intervention
        intervention_path: Intervention path (L, E) - all zeros except 1 at selected experts
        alphas: Set of alpha values to test
        top_k: Number of top experts to select
        batch_size: Number of prompts to process at once

    Returns:
        ExperimentResults for the target country
    """
    prompts = get_all_prompts(model.tokenizer)

    all_results: set[InterventionResult] = set()

    for alpha in tqdm(
        alphas, desc="Testing alpha values", total=len(alphas), leave=False, position=1
    ):
        # Process prompts in batches
        prompt_batches = list(batched(prompts, batch_size))
        for batch_prompts in tqdm(
            prompt_batches,
            desc=f"Testing prompt batches for alpha {alpha}",
            total=len(prompt_batches),
            leave=False,
            position=0,
        ):
            batch_prompts_list = list(batch_prompts)

            # Create intervention paths dict (using PRE_ANSWER experiment type)
            intervention_paths = {ExperimentType.PRE_ANSWER: intervention_path}

            pre_probs, post_probs_dict = run_intervention(
                batch_prompts_list, model, intervention_paths, alpha, top_k
            )

            # Process results for each prompt in the batch
            for prompt_idx, prompt in enumerate(batch_prompts_list):
                pre_prob = pre_probs[prompt_idx]
                for post_probs in post_probs_dict.values():
                    post_prob = post_probs[prompt_idx]
                    # Calculate normalized forgetfulness: (pre - post) / pre
                    normalized_forgetfulness = (
                        (pre_prob - post_prob) / pre_prob if pre_prob > 0 else 0.0
                    )
                    all_results.add(
                        InterventionResult(
                            country=prompt.country,
                            intervention_country=target_country,
                            pre_intervention_prob=pre_prob,
                            post_intervention_prob=post_prob,
                            forgetfulness=InterventionMetric(
                                alpha=alpha, value=normalized_forgetfulness
                            ),
                        )
                    )

    # Structure results similar to run_intervention_experiment
    self_intervention_results = {
        result for result in all_results if result.country == target_country
    }
    other_intervention_results = {
        result for result in all_results if result.country != target_country
    }

    other_results_averaged = set()
    target_results_averaged = set()
    specificity_scores = set()

    for alpha in alphas:
        other_results_for_alpha = {
            result
            for result in other_intervention_results
            if result.forgetfulness.alpha == alpha
        }
        if not other_results_for_alpha:
            continue

        avg_pre_intervention_prob = sum(
            [result.pre_intervention_prob for result in other_results_for_alpha]
        ) / len(other_results_for_alpha)
        avg_post_intervention_prob = sum(
            [result.post_intervention_prob for result in other_results_for_alpha]
        ) / len(other_results_for_alpha)
        avg_forgetfulness = (
            (avg_pre_intervention_prob - avg_post_intervention_prob)
            / avg_pre_intervention_prob
            if avg_pre_intervention_prob > 0
            else 0.0
        )

        other_results_averaged.add(
            InterventionResult(
                country="avg",
                intervention_country=target_country,
                pre_intervention_prob=avg_pre_intervention_prob,
                post_intervention_prob=avg_post_intervention_prob,
                forgetfulness=InterventionMetric(alpha=alpha, value=avg_forgetfulness),
            )
        )

        target_results_for_alpha = {
            result
            for result in self_intervention_results
            if result.forgetfulness.alpha == alpha
        }
        if not target_results_for_alpha:
            continue

        avg_target_pre_intervention_prob = sum(
            [result.pre_intervention_prob for result in target_results_for_alpha]
        ) / len(target_results_for_alpha)
        avg_target_post_intervention_prob = sum(
            [result.post_intervention_prob for result in target_results_for_alpha]
        ) / len(target_results_for_alpha)
        avg_target_forgetfulness = (
            (avg_target_pre_intervention_prob - avg_target_post_intervention_prob)
            / avg_target_pre_intervention_prob
            if avg_target_pre_intervention_prob > 0
            else 0.0
        )

        specificity_scores.add(
            InterventionMetric(
                alpha=alpha, value=avg_target_forgetfulness - avg_forgetfulness
            )
        )
        target_results_averaged.add(
            InterventionResult(
                country=target_country,
                intervention_country=target_country,
                pre_intervention_prob=avg_target_pre_intervention_prob,
                post_intervention_prob=avg_target_post_intervention_prob,
                forgetfulness=InterventionMetric(
                    alpha=alpha, value=avg_target_forgetfulness
                ),
            )
        )

    return ExperimentResults(
        target_country=target_country,
        target_results=tuple(target_results_averaged),
        other_results=tuple(other_intervention_results),
        other_results_averaged=tuple(other_results_averaged),
        specificity_scores=tuple(specificity_scores),
    )


def plot_subset_results(
    results: ExperimentResults,
    subset_name: str,
    subset_expert_indices: th.Tensor,  # (N, 2) where each row is [layer_idx, expert_idx]
    output_path: Path,
) -> None:
    """
    Create visualization for a subset's intervention results.

    Args:
        results: ExperimentResults for the subset
        subset_name: Name/description of the subset
        subset_expert_indices: Tensor of shape (N, 2) with expert indices
        output_path: Path to save the visualization
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract alphas and sort results by alpha
    alphas = sorted(score.alpha for score in results.specificity_scores)

    # Build forgetfulness values sorted by alpha
    target_forgetfulnesses: list[float] = []
    other_forgetfulnesses: list[float] = []
    specificity_scores: list[float] = []

    for alpha in alphas:
        # Get target forgetfulness for this alpha
        targets_for_alpha = {
            result
            for result in results.target_results
            if result.forgetfulness.alpha == alpha
        }
        if not targets_for_alpha:
            continue
        target_forgetfulness = next(iter(targets_for_alpha)).forgetfulness.value
        target_forgetfulnesses.append(target_forgetfulness)

        # Get averaged other forgetfulness for this alpha
        other_results_averaged_for_alpha = [
            result
            for result in results.other_results_averaged
            if result.forgetfulness.alpha == alpha
        ]
        if not other_results_averaged_for_alpha:
            continue
        other_forgetfulness = next(
            iter(other_results_averaged_for_alpha)
        ).forgetfulness.value
        other_forgetfulnesses.append(other_forgetfulness)

        # Get specificity score for this alpha
        specificity_scores_for_alpha = {
            result for result in results.specificity_scores if result.alpha == alpha
        }
        if not specificity_scores_for_alpha:
            continue
        specificity_score = next(iter(specificity_scores_for_alpha)).value
        specificity_scores.append(specificity_score)

    if not alphas:
        logger.warning(f"No results to plot for subset {subset_name}")
        plt.close()
        return

    target_capital = COUNTRY_TO_CAPITAL[results.target_country]

    # Create expert description
    expert_desc = ", ".join(
        [f"L{int(idx[0])}E{int(idx[1])}" for idx in subset_expert_indices]
    )

    # Plot 1: Forgetfulness by alpha
    ax1 = axes[0]
    ax1.plot(
        alphas,
        target_forgetfulnesses,
        "b-o",
        label=f"{results.target_country} (target)",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        alphas,
        other_forgetfulnesses,
        "r-s",
        label="Other countries (avg)",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Alpha (intervention strength)", fontsize=12)
    ax1.set_ylabel("Forgetfulness (normalized: (pre-post)/pre)", fontsize=12)
    ax1.set_title(
        f"Forgetfulness vs Intervention Strength\n{subset_name}\n{expert_desc}",
        fontsize=12,
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 2: Specificity score
    ax2 = axes[1]
    ax2.plot(alphas, specificity_scores, "g-^", linewidth=2, markersize=8)
    ax2.set_xlabel("Alpha (intervention strength)", fontsize=12)
    ax2.set_ylabel("Specificity (target - other forgetfulness)", fontsize=12)
    ax2.set_title(
        f"Specificity of {results.target_country}-{target_capital} Knowledge\n{subset_name}\n{expert_desc}",
        fontsize=12,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.fill_between(alphas, specificity_scores, alpha=0.3, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


@arguably.command()
def capital_country_expert_subset_analysis(
    *,
    model_name: str = "olmoe-i",
    model_dtype: str = "bf16",
    importance_file: str = "out/capital_country_expert_importance/south_korea.pt",
    target_country: str = "South Korea",
    importance_threshold: float = 0.1,
    alpha_min: float = 0.0,
    alpha_max: float = 2.0,
    alpha_steps: int = 9,
    intervention_batch_size: int = 8,
    seed: int = 0,
    hf_token: str = "",
    output_dir: str = "out/capital_country_expert_subset_analysis",
    log_level: str = "INFO",
) -> None:
    """
    Analyze expert importance by testing interventions on subsets of experts.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, gpt, etc.)
        model_dtype: Data type for model weights
        importance_file: Path to expert importance scores .pt file
        target_country: Target country for analysis
        importance_threshold: Minimum importance score to keep experts
        alpha_min: Minimum alpha value for intervention sweep
        alpha_max: Maximum alpha value for intervention sweep
        alpha_steps: Number of alpha values to test
        intervention_batch_size: Batch size for intervention experiments
        seed: Random seed for reproducibility
        hf_token: Hugging Face API token
        output_dir: Directory to save results
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(
        f"Running capital_country_expert_subset_analysis with log level: {log_level}"
    )

    assert target_country in COUNTRY_TO_CAPITAL, (
        f"Target country '{target_country}' not found in COUNTRY_TO_CAPITAL. "
        f"Available countries: {sorted(COUNTRY_TO_CAPITAL.keys())}"
    )

    # Set random seeds
    th.manual_seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and filter experts
    logger.info("=" * 80)
    logger.info("STEP 1: Loading and filtering experts")
    logger.info("=" * 80)

    importance_path = Path(importance_file)
    assert importance_path.exists(), (
        f"Importance file does not exist: {importance_path}"
    )
    expert_indices, expert_importances, importance_target_country = (
        load_and_filter_experts(importance_path, importance_threshold)
    )

    assert importance_target_country == target_country, (
        f"Importance file target country ({importance_target_country}) "
        f"does not match specified target country ({target_country})"
    )
    assert len(expert_indices) > 0, (
        f"No experts found above importance threshold {importance_threshold}"
    )

    logger.info(
        f"Found {len(expert_indices)} experts above threshold "
        f"with importance scores: {expert_importances.tolist()[:5]}..."
    )

    # Generate expert subsets
    logger.info("=" * 80)
    logger.info("STEP 2: Generating expert subsets")
    logger.info("=" * 80)

    subsets = generate_expert_subsets(expert_indices)

    # Load model
    logger.info("=" * 80)
    logger.info("STEP 3: Loading model")
    logger.info("=" * 80)

    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict()
    model_dtype_torch = get_dtype(model_dtype)

    logger.info(f"Loading model: {model_config.hf_name}")
    logger.info(f"Checkpoint: {model_ckpt}")

    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        torch_dtype=model_dtype_torch,
        token=hf_token,
    )

    logger.info("Model loaded successfully")
    logger.info(f"Number of layers with routers: {len(model.layers_with_routers)}")

    # Get model architecture info
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok
    num_layers = len(model.layers_with_routers)

    logger.info(f"Number of experts: {num_experts}")
    logger.info(f"Top-k: {top_k}")
    logger.info(f"Number of layers: {num_layers}")

    # Alpha values
    alphas = set(th.linspace(alpha_min, alpha_max, alpha_steps).tolist())
    logger.info(f"Testing alphas: {sorted(alphas)}")

    # Run experiments for each subset type
    logger.info("=" * 80)
    logger.info("STEP 4: Running intervention experiments")
    logger.info("=" * 80)

    country_slug = target_country.lower().replace(" ", "_")
    results_dir = output_path / country_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(FIGURE_DIR) / "capital_country_expert_subset_analysis" / country_slug
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Process each subset type
    for subset_type, subset_list in subsets.items():
        logger.info(f"Processing {subset_type} subsets: {len(subset_list)} total")

        for subset_idx, subset_expert_indices in enumerate(
            tqdm(subset_list, desc=f"Processing {subset_type} subsets")
        ):
            # Create intervention path
            intervention_path = create_intervention_path(
                subset_expert_indices, num_layers, num_experts
            )

            # Run intervention experiment
            results = run_subset_intervention_experiment(
                model,
                target_country,
                intervention_path,
                alphas,
                top_k,
                batch_size=intervention_batch_size,
            )

            # Get importance scores for this subset
            # Find which indices in expert_indices match the subset experts
            subset_importance_values = []
            for idx_row in subset_expert_indices:
                layer_idx, expert_idx = int(idx_row[0]), int(idx_row[1])
                # Find matching expert in the sorted list
                matching = (expert_indices[:, 0] == layer_idx) & (
                    expert_indices[:, 1] == expert_idx
                )
                if matching.any():
                    importance_idx = matching.nonzero(as_tuple=False)[0, 0]
                    subset_importance_values.append(
                        expert_importances[importance_idx].item()
                    )
                else:
                    subset_importance_values.append(0.0)

            # Save results
            subset_name = f"{subset_type}_{subset_idx}"
            results_file = results_dir / f"{subset_name}.pt"
            th.save(
                {
                    "results": results,
                    "subset_type": subset_type,
                    "subset_idx": subset_idx,
                    "expert_indices": subset_expert_indices,
                    "expert_importances": th.tensor(subset_importance_values),
                    "alphas": sorted(alphas),
                },
                results_file,
            )

            # Create visualization
            viz_file = fig_dir / f"{subset_name}.png"
            plot_subset_results(results, subset_name, subset_expert_indices, viz_file)

    # Print summary
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target country: {target_country}")
    logger.info(f"Total experts analyzed: {len(expert_indices)}")
    logger.info(f"Single expert subsets: {len(subsets['single'])}")
    logger.info(f"Pair subsets: {len(subsets['pairs'])}")
    logger.info(f"Top-k subsets: {len(subsets['top_k'])}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Visualizations saved to: {fig_dir}")


if __name__ == "__main__":
    arguably.run()
