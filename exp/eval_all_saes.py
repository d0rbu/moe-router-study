"""
Comprehensive evaluation script for all SAE outputs.

This script:
1. Discovers all SAE experiment directories
2. Evaluates each SAE using both sae_eval_saebench and eval_intruder
3. Aggregates results and generates rankings
4. Creates visualizations for the results

Requirements:
    pip install pandas seaborn

Or add to pyproject.toml:
    dependencies = [..., "pandas>=2.0.0", "seaborn>=0.12.0"]
"""

import argparse
import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import pandas as pd
    import seaborn as sns

    HAS_VIZ_DEPS = True
except ImportError:
    HAS_VIZ_DEPS = False
    logger.warning(
        "pandas and/or seaborn not installed. Install with: pip install pandas seaborn"
    )

from exp import OUTPUT_DIR


@dataclass
class SAEExperiment:
    """Represents a single SAE experiment directory."""

    experiment_name: str
    sae_dirs: list[Path]
    config_paths: list[Path]
    ae_paths: list[Path]


@dataclass
class EvaluationResults:
    """Aggregated evaluation results for an SAE."""

    experiment_name: str
    sae_id: str
    saebench_results: dict[str, Any] = field(default_factory=dict)
    intruder_results: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def discover_sae_experiments(output_dir: Path) -> list[SAEExperiment]:
    """
    Discover all SAE experiment directories.

    An experiment directory contains subdirectories with config.json and ae.pt files.
    """
    experiments = []

    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Check if this directory contains subdirectories with config.json
        sae_dirs = []
        config_paths = []
        ae_paths = []

        for sub_dir in exp_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            config_path = sub_dir / "config.json"
            # Look for ae.pt or ae_*.pt files
            ae_pt_files = list(sub_dir.glob("ae*.pt"))

            if config_path.exists() and ae_pt_files:
                sae_dirs.append(sub_dir)
                config_paths.append(config_path)
                ae_paths.append(ae_pt_files[0])  # Use first ae.pt file found

        if sae_dirs:
            experiments.append(
                SAEExperiment(
                    experiment_name=exp_dir.name,
                    sae_dirs=sae_dirs,
                    config_paths=config_paths,
                    ae_paths=ae_paths,
                )
            )

    return experiments


def run_saebench_eval(
    experiment_name: str,
    model_name: str,
    eval_types: list[str],
    batchsize: int,
    dtype: str,
    seed: int,
) -> bool:
    """Run SAEBench evaluation on an experiment."""
    logger.info(f"Running SAEBench evaluation on {experiment_name}")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "exp.sae_eval_saebench",
        "--experiment-dir",
        experiment_name,
        "--model-name",
        model_name,
        "--batchsize",
        str(batchsize),
        "--dtype",
        dtype,
        "--seed",
        str(seed),
    ]

    if eval_types:
        for eval_type in eval_types:
            cmd.extend(["--eval-types", eval_type])

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"SAEBench evaluation completed for {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"SAEBench evaluation failed for {experiment_name}: {e.stderr}")
        return False


def run_intruder_eval(
    experiment_name: str,
    model_name: str,
    model_dtype: str,
    n_tokens: int,
    batchsize: int,
    n_latents: int,
    seed: int,
) -> bool:
    """Run intruder evaluation on an experiment."""
    logger.info(f"Running intruder evaluation on {experiment_name}")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "exp.eval_intruder",
        "--experiment-dir",
        experiment_name,
        "--model-name",
        model_name,
        "--model-dtype",
        model_dtype,
        "--n-tokens",
        str(n_tokens),
        "--batchsize",
        str(batchsize),
        "--n-latents",
        str(n_latents),
        "--seed",
        str(seed),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"Intruder evaluation completed for {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Intruder evaluation failed for {experiment_name}: {e.stderr}")
        return False


def load_saebench_results(experiment_dir: Path) -> dict[str, Any]:
    """Load SAEBench evaluation results."""
    results = {}

    # SAEBench saves results in the experiment directory
    # Look for result files
    result_files = list(experiment_dir.glob("**/results*.json"))

    for result_file in result_files:
        try:
            with open(result_file) as f:
                result_data = json.load(f)
                results[result_file.stem] = result_data
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    return results


def load_intruder_results(experiment_dir: Path) -> dict[str, Any]:
    """Load intruder evaluation results."""
    results = {}

    # Intruder results are saved in delphi/scores/
    scores_dir = experiment_dir / "delphi" / "scores"

    if not scores_dir.exists():
        return results

    # Load all score files
    for score_file in scores_dir.glob("*.txt"):
        try:
            with open(score_file, "rb") as f:
                import orjson

                score_data = orjson.loads(f.read())
                results[score_file.stem] = score_data
        except Exception as e:
            logger.warning(f"Failed to load {score_file}: {e}")

    return results


def aggregate_results(
    experiments: list[SAEExperiment],
) -> list[EvaluationResults]:
    """Aggregate all evaluation results."""
    all_results = []

    for exp in tqdm(experiments, desc="Aggregating results"):
        exp_dir = Path(OUTPUT_DIR) / exp.experiment_name

        for sae_dir, config_path in zip(exp.sae_dirs, exp.config_paths, strict=False):
            # Load config
            with open(config_path) as f:
                config = json.load(f)

            # Load evaluation results
            saebench_results = load_saebench_results(exp_dir)
            intruder_results = load_intruder_results(exp_dir)

            result = EvaluationResults(
                experiment_name=exp.experiment_name,
                sae_id=sae_dir.name,
                saebench_results=saebench_results,
                intruder_results=intruder_results,
                config=config,
            )

            all_results.append(result)

    return all_results


def extract_metrics(results: list[EvaluationResults]):
    """Extract key metrics from results into a DataFrame."""
    if not HAS_VIZ_DEPS:
        logger.error(
            "pandas is required for metric extraction. Install with: pip install pandas"
        )
        return None

    rows = []

    for result in results:
        row = {
            "experiment": result.experiment_name,
            "sae_id": result.sae_id,
        }

        # Extract config parameters
        if "trainer" in result.config:
            trainer_config = result.config["trainer"]
            row.update(
                {
                    "expansion_factor": trainer_config.get("dict_size", 0)
                    / trainer_config.get("activation_dim", 1),
                    "k": trainer_config.get("k", 0),
                    "layer": trainer_config.get("layer", 0),
                    "lr": trainer_config.get("lr", 0),
                    "seed": trainer_config.get("seed", 0),
                }
            )

        # Extract SAEBench metrics
        for eval_name, eval_data in result.saebench_results.items():
            if isinstance(eval_data, dict):
                for metric_name, metric_value in eval_data.items():
                    if isinstance(metric_value, int | float):
                        row[f"saebench_{eval_name}_{metric_name}"] = metric_value

        # Extract intruder metrics (average score)
        if result.intruder_results:
            scores = [
                score
                for score in result.intruder_results.values()
                if isinstance(score, int | float)
            ]
            if scores:
                row["intruder_avg_score"] = np.mean(scores)
                row["intruder_std_score"] = np.std(scores)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_rankings(df):
    """Compute rankings for each metric."""
    if not HAS_VIZ_DEPS or df is None:
        logger.error("pandas is required for computing rankings")
        return {}

    rankings = {}

    # Get all metric columns
    metric_cols = [
        col
        for col in df.columns
        if col.startswith("saebench_") or col.startswith("intruder_")
    ]

    for metric in metric_cols:
        if metric in df.columns:
            # Rank by metric (higher is better for most metrics)
            ranked = df.sort_values(metric, ascending=False).copy()
            ranked[f"{metric}_rank"] = range(1, len(ranked) + 1)
            rankings[metric] = ranked[
                ["experiment", "sae_id", metric, f"{metric}_rank"]
            ]

    # Compute overall ranking (average of all ranks)
    if metric_cols:
        rank_cols = [col for col in df.columns if col.endswith("_rank")]
        if rank_cols:
            df["overall_rank"] = df[rank_cols].mean(axis=1)
            overall_ranking = df.sort_values("overall_rank").copy()
            rankings["overall"] = overall_ranking[
                ["experiment", "sae_id", "overall_rank", *metric_cols]
            ]

    return rankings


def create_visualizations(
    df,
    rankings: dict,
    output_dir: Path,
) -> None:
    """Create visualizations for the evaluation results."""
    if not HAS_VIZ_DEPS or df is None:
        logger.warning(
            "pandas and seaborn are required for visualizations. Skipping..."
        )
        return

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1. Overall ranking bar chart
    if "overall" in rankings:
        plt.figure(figsize=(14, 8))
        overall = rankings["overall"].head(20)  # Top 20
        plt.barh(
            range(len(overall)),
            overall["overall_rank"],
            color=sns.color_palette("viridis", len(overall)),
        )
        plt.yticks(
            range(len(overall)),
            [f"{row['experiment']}/{row['sae_id']}" for _, row in overall.iterrows()],
        )
        plt.xlabel("Overall Rank (lower is better)")
        plt.title("Top 20 SAEs by Overall Ranking")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(viz_dir / "overall_ranking.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 2. Metric comparison heatmap
    metric_cols = [
        col
        for col in df.columns
        if col.startswith("saebench_") or col.startswith("intruder_")
    ]
    if metric_cols:
        # Normalize metrics to 0-1 scale for comparison
        normalized_df = df[["experiment", "sae_id", *metric_cols]].copy()
        for col in metric_cols:
            if col in normalized_df.columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (
                        max_val - min_val
                    )

        # Create heatmap
        plt.figure(figsize=(16, 10))
        heatmap_data = normalized_df.set_index(
            normalized_df["experiment"] + "/" + normalized_df["sae_id"]
        )[metric_cols]

        # Only show top 30 SAEs for readability
        if len(heatmap_data) > 30:
            heatmap_data = heatmap_data.head(30)

        sns.heatmap(
            heatmap_data.T,
            cmap="RdYlGn",
            center=0.5,
            cbar_kws={"label": "Normalized Score"},
            linewidths=0.5,
        )
        plt.title("Metric Comparison Heatmap (Normalized)")
        plt.xlabel("SAE")
        plt.ylabel("Metric")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(viz_dir / "metric_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Individual metric rankings
    for metric in metric_cols:
        if metric in df.columns and df[metric].notna().sum() > 0:
            plt.figure(figsize=(12, 8))
            top_saes = df.nlargest(15, metric)

            plt.barh(
                range(len(top_saes)),
                top_saes[metric],
                color=sns.color_palette("coolwarm", len(top_saes)),
            )
            plt.yticks(
                range(len(top_saes)),
                [
                    f"{row['experiment']}/{row['sae_id']}"
                    for _, row in top_saes.iterrows()
                ],
            )
            plt.xlabel(metric)
            plt.title(f"Top 15 SAEs by {metric}")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            safe_filename = metric.replace("/", "_").replace(" ", "_")
            plt.savefig(
                viz_dir / f"{safe_filename}_ranking.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    # 4. Hyperparameter correlation plots
    hparam_cols = ["expansion_factor", "k", "layer", "lr"]
    available_hparams = [col for col in hparam_cols if col in df.columns]

    if available_hparams and metric_cols:
        for hparam in available_hparams:
            for metric in metric_cols[:3]:  # Only plot top 3 metrics
                if metric in df.columns:
                    plt.figure(figsize=(10, 6))
                    valid_data = df[[hparam, metric]].dropna()

                    if len(valid_data) > 0:
                        plt.scatter(
                            valid_data[hparam],
                            valid_data[metric],
                            alpha=0.6,
                            s=100,
                        )
                        plt.xlabel(hparam)
                        plt.ylabel(metric)
                        plt.title(f"{metric} vs {hparam}")
                        plt.tight_layout()

                        safe_metric = metric.replace("/", "_").replace(" ", "_")
                        plt.savefig(
                            viz_dir / f"{hparam}_vs_{safe_metric}.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()

    logger.info(f"Visualizations saved to {viz_dir}")


def save_results(
    df,
    rankings: dict,
    output_dir: Path,
) -> None:
    """Save aggregated results and rankings to files."""
    if not HAS_VIZ_DEPS or df is None:
        logger.warning("pandas is required for saving results. Skipping...")
        return

    results_dir = output_dir / "evaluation_results"
    results_dir.mkdir(exist_ok=True, parents=True)

    # Save full results DataFrame
    df.to_csv(results_dir / "all_results.csv", index=False)
    df.to_json(results_dir / "all_results.json", orient="records", indent=2)

    # Save rankings
    for metric, ranking_df in rankings.items():
        safe_metric = metric.replace("/", "_").replace(" ", "_")
        ranking_df.to_csv(results_dir / f"ranking_{safe_metric}.csv", index=False)

    logger.info(f"Results saved to {results_dir}")


async def main(
    model_name: str,
    run_saebench: bool,
    run_intruder: bool,
    saebench_eval_types: list[str] | None,
    saebench_batchsize: int,
    intruder_n_tokens: int,
    intruder_batchsize: int,
    intruder_n_latents: int,
    dtype: str,
    seed: int,
    skip_evaluation: bool,
) -> None:
    """Main evaluation pipeline."""
    output_path = Path(OUTPUT_DIR)

    # Discover experiments
    logger.info("Discovering SAE experiments...")
    experiments = discover_sae_experiments(output_path)
    logger.info(f"Found {len(experiments)} experiments")

    for exp in experiments:
        logger.info(
            f"  - {exp.experiment_name}: {len(exp.sae_dirs)} SAE configurations"
        )

    if not skip_evaluation:
        # Run evaluations
        for exp in tqdm(experiments, desc="Evaluating experiments"):
            if run_saebench:
                run_saebench_eval(
                    exp.experiment_name,
                    model_name,
                    saebench_eval_types or [],
                    saebench_batchsize,
                    dtype,
                    seed,
                )

            if run_intruder:
                run_intruder_eval(
                    exp.experiment_name,
                    model_name,
                    dtype,
                    intruder_n_tokens,
                    intruder_batchsize,
                    intruder_n_latents,
                    seed,
                )

    # Aggregate results
    logger.info("Aggregating results...")
    results = aggregate_results(experiments)
    logger.info(f"Aggregated {len(results)} SAE results")

    # Extract metrics into DataFrame
    df = extract_metrics(results)
    logger.info(f"Extracted metrics for {len(df)} SAEs")

    # Compute rankings
    logger.info("Computing rankings...")
    rankings = compute_rankings(df)
    logger.info(f"Computed {len(rankings)} ranking tables")

    # Save results
    save_results(df, rankings, output_path)

    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(df, rankings, output_path)

    logger.info("✅ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all SAE experiments and generate rankings"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="olmoe-i",
        help="Model name for evaluation",
    )
    parser.add_argument(
        "--run-saebench",
        action="store_true",
        help="Run SAEBench evaluation",
    )
    parser.add_argument(
        "--run-intruder",
        action="store_true",
        help="Run intruder evaluation",
    )
    parser.add_argument(
        "--saebench-eval-types",
        type=str,
        nargs="+",
        default=None,
        help="SAEBench evaluation types to run",
    )
    parser.add_argument(
        "--saebench-batchsize",
        type=int,
        default=512,
        help="Batch size for SAEBench evaluation",
    )
    parser.add_argument(
        "--intruder-n-tokens",
        type=int,
        default=10_000_000,
        help="Number of tokens for intruder evaluation",
    )
    parser.add_argument(
        "--intruder-batchsize",
        type=int,
        default=8,
        help="Batch size for intruder evaluation",
    )
    parser.add_argument(
        "--intruder-n-latents",
        type=int,
        default=1000,
        help="Number of latents for intruder evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Data type for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip running evaluations and only aggregate existing results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Run main
    asyncio.run(
        main(
            model_name=args.model_name,
            run_saebench=args.run_saebench,
            run_intruder=args.run_intruder,
            saebench_eval_types=args.saebench_eval_types,
            saebench_batchsize=args.saebench_batchsize,
            intruder_n_tokens=args.intruder_n_tokens,
            intruder_batchsize=args.intruder_batchsize,
            intruder_n_latents=args.intruder_n_latents,
            dtype=args.dtype,
            seed=args.seed,
            skip_evaluation=args.skip_evaluation,
        )
    )
