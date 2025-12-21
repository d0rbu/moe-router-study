"""
Comprehensive evaluation script for all SAE outputs.

This script:
1. Discovers all SAE experiment directories
2. Evaluates each SAE using both sae_eval_saebench and eval_intruder
3. Aggregates results and generates rankings
4. Creates visualizations for the results
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from exp import OUTPUT_DIR
from viz import FIGURE_DIR

EVAL_TIMEOUT = 172800  # 48 hours


@dataclass
class SAEInfo:
    """Represents a single SAE configuration."""

    sae_dir: Path
    config_path: Path
    ae_path: Path


@dataclass
class SAEExperiment:
    """Represents a single SAE experiment directory."""

    experiment_name: str
    saes: list[SAEInfo]


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
    logger.info(f"Scanning output directory: {output_dir}")

    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return experiments

    all_dirs = list(output_dir.iterdir())
    logger.info(f"Found {len(all_dirs)} items in output directory")

    for exp_dir in all_dirs:
        if not exp_dir.is_dir():
            logger.debug(f"Skipping non-directory: {exp_dir.name}")
            continue

        logger.info(f"Examining experiment directory: {exp_dir.name}")

        # Check if this directory contains subdirectories with config.json
        saes = []
        sub_dirs = list(exp_dir.iterdir())
        logger.info(f"  Found {len(sub_dirs)} items in {exp_dir.name}")

        for sub_dir in sub_dirs:
            if not sub_dir.is_dir():
                logger.debug(f"  Skipping non-directory: {sub_dir.name}")
                continue

            config_path = sub_dir / "config.json"
            # Look for ae.pt or ae_*.pt files
            ae_pt_files = list(sub_dir.glob("ae*.pt"))

            logger.debug(
                f"  Checking {sub_dir.name}: config={config_path.exists()}, ae_files={len(ae_pt_files)}"
            )

            if config_path.exists() and ae_pt_files:
                logger.debug(f"    ✅ Valid SAE found: {sub_dir.name}")
                saes.append(
                    SAEInfo(
                        sae_dir=sub_dir,
                        config_path=config_path,
                        ae_path=ae_pt_files[0],  # Use first ae.pt file found
                    )
                )
            else:
                logger.debug(
                    f"    ❌ Invalid SAE: {sub_dir.name} (missing config or ae.pt)"
                )

        if saes:
            logger.info(f"  ✅ Experiment {exp_dir.name}: {len(saes)} SAEs")
            experiments.append(
                SAEExperiment(
                    experiment_name=exp_dir.name,
                    saes=saes,
                )
            )
        else:
            logger.warning(f"  ❌ No valid SAEs found in {exp_dir.name}")

    logger.info(f"Discovery complete: {len(experiments)} experiments found")
    return experiments


def run_saebench_eval(
    experiment_name: str,
    model_name: str,
    eval_types: list[str],
    batchsize: int,
    num_autointerp_latents: int,
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
        "--num-autointerp-latents",
        str(num_autointerp_latents),
        "--dtype",
        dtype,
        "--seed",
        str(seed),
    ]

    if eval_types:
        for eval_type in eval_types:
            cmd.extend(["--eval-types", eval_type])

    logger.debug(f"🔧 Running command: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=EVAL_TIMEOUT,
        )
        logger.debug(f"✅ SAEBench evaluation completed for {experiment_name}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"❌ SAEBench evaluation timed out for {experiment_name}")
        return False
    except subprocess.CalledProcessError as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(f"❌ SAEBench evaluation failed for {experiment_name}")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Return code: {exception.returncode}")
        return False
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(
            f"❌ Unexpected error in SAEBench evaluation for {experiment_name}:\n{traceback_str}{exception_str}"
        )
        return False


def run_intruder_eval(
    experiment_name: str,
    model_name: str,
    model_dtype: str,
    n_tokens: int,
    batchsize: int,
    n_latents: int,
    vllm_num_gpus: int,
    cache_device_idx: int,
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
        "eval-intruder",
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
        "--vllm-num-gpus",
        str(vllm_num_gpus),
        "--seed",
        str(seed),
    ]

    logger.debug(f"🔧 Running command: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=EVAL_TIMEOUT,
        )
        logger.debug(f"✅ Intruder evaluation completed for {experiment_name}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Intruder evaluation timed out for {experiment_name}")
        return False
    except subprocess.CalledProcessError as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(f"❌ Intruder evaluation failed for {experiment_name}")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Return code: {exception.returncode}")
        return False
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(
            f"❌ Unexpected error in intruder evaluation for {experiment_name}:\n{traceback_str}{exception_str}"
        )
        return False


def load_saebench_results(experiment_dir: Path, sae_id: str) -> dict[str, Any]:
    """Load SAEBench evaluation results."""
    results = {}
    logger.debug(f"Loading SAEBench results from {experiment_dir} for SAE {sae_id}")

    # SAEBench saves results in the experiment directory
    # Look for result files
    result_files = list(experiment_dir.glob(f"**/*_{sae_id}_*results*.json"))
    logger.debug(f"Found {len(result_files)} potential SAEBench result files")
    if result_files:
        logger.trace(f"SAEBench result files: {[f.name for f in result_files]}")

    for result_file in result_files:
        try:
            with open(result_file) as f:
                result_data = json.load(f)
                results[result_file.stem] = result_data
                logger.debug(f"  ✅ Loaded SAEBench results from {result_file.name}")
        except Exception as exception:
            traceback_lines = traceback.format_tb(exception.__traceback__)
            traceback_str = "".join(traceback_lines)
            exception_str = str(exception)
            logger.warning(
                f"Failed to load {result_file}:\n{traceback_str}{exception_str}"
            )

    if not results:
        logger.debug(f"  ❌ No SAEBench results found in {experiment_dir}")

    return results


def load_intruder_results(
    experiment_dir: Path, sae_id: str | None = None
) -> dict[str, Any]:
    """Load intruder evaluation results."""
    results = {}
    logger.debug(
        f"Loading intruder results from {experiment_dir}"
        + (f" for SAE {sae_id}" if sae_id else "")
    )

    # Intruder results are saved in delphi/scores/
    scores_dir = experiment_dir / "delphi" / "scores"
    logger.debug(f"Looking for intruder results in {scores_dir}")

    if not scores_dir.exists():
        logger.error(f"  ❌ Intruder scores directory does not exist: {scores_dir}")
        return results

    # Load all score files
    score_files = list(scores_dir.glob("*.txt"))
    logger.debug(f"Found {len(score_files)} potential intruder score files")
    if score_files:
        logger.trace(f"Intruder score files: {[f.name for f in score_files]}")

    for score_file in score_files:
        try:
            with open(score_file, "rb") as f:
                score_data = orjson.loads(f.read())
                results[score_file.stem] = score_data
                logger.debug(f"  ✅ Loaded intruder results from {score_file.name}")
        except Exception as exception:
            traceback_lines = traceback.format_tb(exception.__traceback__)
            traceback_str = "".join(traceback_lines)
            exception_str = str(exception)
            logger.warning(
                f"Failed to load {score_file}:\n{traceback_str}{exception_str}"
            )

    if not results:
        logger.error(f"  ❌ No intruder results found in {scores_dir}")

    return results


DEFAULT_EVAL_RESULTS_DIR = "eval_results"


def aggregate_results(
    experiments: list[SAEExperiment],
) -> list[EvaluationResults]:
    """Aggregate all evaluation results."""
    all_results = []
    logger.info(f"Aggregating results from {len(experiments)} experiments")

    for exp in tqdm(experiments, desc="Aggregating results"):
        exp_dir = Path(OUTPUT_DIR) / exp.experiment_name
        logger.debug(
            f"Processing experiment: {exp.experiment_name} ({len(exp.saes)} SAEs)"
        )

        for sae_info in exp.saes:
            sae_id = sae_info.sae_dir.name
            logger.debug(f"  Processing SAE {sae_id}")

            # Load config
            with open(sae_info.config_path) as f:
                config = json.load(f)

            # Load evaluation results - try both experiment level and SAE level
            # First try at experiment level (for backward compatibility)
            saebench_results = load_saebench_results(exp_dir, sae_id)
            intruder_results = load_intruder_results(exp_dir, sae_id)

            # If no results at experiment level, try at SAE level
            if not saebench_results and not intruder_results:
                logger.debug(
                    f"    No results at experiment level, trying SAE directory: {sae_info.sae_dir}"
                )
                saebench_results = load_saebench_results(sae_info.sae_dir, sae_id)
                intruder_results = load_intruder_results(sae_info.sae_dir, sae_id)

            # If no results at SAE level, try at eval_results directory
            if not saebench_results and not intruder_results:
                logger.debug(
                    f"    No results at SAE level, trying default directory: {DEFAULT_EVAL_RESULTS_DIR}"
                )
                saebench_results = load_saebench_results(
                    Path(DEFAULT_EVAL_RESULTS_DIR), sae_id
                )
                intruder_results = load_intruder_results(
                    Path(DEFAULT_EVAL_RESULTS_DIR), sae_id
                )

            # Check if we still have no evaluation results after trying both locations
            if not saebench_results and not intruder_results:
                logger.critical(
                    f"    ❌ No evaluation results found for SAE {sae_id} in either experiment or SAE directory"
                )
                # Continue anyway to preserve metadata, but log the issue

            result = EvaluationResults(
                experiment_name=exp.experiment_name,
                sae_id=sae_id,
                saebench_results=saebench_results,
                intruder_results=intruder_results,
                config=config,
            )

            all_results.append(result)

            # Log what we found
            saebench_count = len(saebench_results)
            intruder_count = len(intruder_results)
            logger.debug(
                f"    Results: {saebench_count} SAEBench, {intruder_count} intruder"
            )

    logger.info(f"Aggregation complete: {len(all_results)} SAE results collected")
    return all_results


def extract_metrics(results: list[EvaluationResults]) -> pd.DataFrame:
    """Extract key metrics from results into a DataFrame."""
    rows = []
    logger.info(f"Extracting metrics from {len(results)} evaluation results")

    for i, result in enumerate(results):
        logger.debug(
            f"Processing result {i + 1}/{len(results)}: {result.experiment_name}/{result.sae_id}"
        )

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
            logger.debug(
                f"  Config extracted: expansion_factor={row.get('expansion_factor')}, k={row.get('k')}"
            )

        # Extract SAEBench metrics
        saebench_metrics_count = 0
        for eval_name, eval_data in result.saebench_results.items():
            if isinstance(eval_data, dict):
                for metric_name, metric_value in eval_data.items():
                    if isinstance(metric_value, int | float):
                        row[f"saebench_{eval_name}_{metric_name}"] = metric_value
                        saebench_metrics_count += 1

        logger.debug(f"  SAEBench metrics extracted: {saebench_metrics_count}")

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
                logger.debug(
                    f"  Intruder metrics extracted: avg={row['intruder_avg_score']:.4f}, std={row['intruder_std_score']:.4f}"
                )
            else:
                logger.debug("  No valid intruder scores found")
        else:
            logger.debug("  No intruder results found")

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        f"Metrics extraction complete: {len(df)} rows, {len(df.columns)} columns"
    )
    logger.info(f"Columns: {list(df.columns)}")
    return df


def compute_rankings(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute rankings for each metric."""
    rankings = {}
    logger.info(
        f"Computing rankings for DataFrame with {len(df)} rows and {len(df.columns)} columns"
    )
    logger.info(f"DataFrame columns: {list(df.columns)}")

    # Get all metric columns
    metric_cols = [
        col
        for col in df.columns
        if col.startswith("saebench_") or col.startswith("intruder_")
    ]
    logger.info(f"Found {len(metric_cols)} metric columns: {metric_cols}")

    if not metric_cols:
        logger.warning("No metric columns found! Cannot compute rankings.")
        return rankings

    for metric in metric_cols:
        if metric in df.columns:
            # Check for non-null values
            non_null_count = df[metric].notna().sum()
            logger.info(f"Metric {metric}: {non_null_count}/{len(df)} non-null values")

            if non_null_count > 0:
                # Rank by metric (higher is better for most metrics)
                ranked = df.sort_values(metric, ascending=False).copy()
                ranked[f"{metric}_rank"] = range(1, len(ranked) + 1)
                rankings[metric] = ranked[
                    ["experiment", "sae_id", metric, f"{metric}_rank"]
                ]
                logger.info(f"  ✅ Created ranking for {metric}")
            else:
                logger.warning(f"  ❌ Skipping {metric} - no valid values")

    # Compute overall ranking (average of all ranks)
    if metric_cols:
        rank_cols = [col for col in df.columns if col.endswith("_rank")]
        logger.info(f"Found {len(rank_cols)} rank columns: {rank_cols}")

        if rank_cols:
            df["overall_rank"] = df[rank_cols].mean(axis=1)
            overall_ranking = df.sort_values("overall_rank").copy()
            rankings["overall"] = overall_ranking[
                ["experiment", "sae_id", "overall_rank", *metric_cols]
            ]
            logger.info("  ✅ Created overall ranking")
        else:
            logger.warning("  ❌ No rank columns found for overall ranking")
    else:
        logger.warning("No metric columns found for ranking computation")

    logger.info(
        f"Rankings computation complete: {len(rankings)} ranking tables created"
    )
    return rankings


def create_visualizations(
    df: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
) -> None:
    """Create visualizations for the evaluation results."""
    viz_dir = Path(FIGURE_DIR) / "eval_all_saes"
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
    df: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save aggregated results and rankings to files."""
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


def main(
    model_name: str,
    run_saebench: bool,
    run_intruder: bool,
    saebench_eval_types: list[str] | None,
    saebench_batchsize: int,
    num_autointerp_latents: int,
    intruder_n_tokens: int,
    intruder_batchsize: int,
    intruder_n_latents: int,
    intruder_vllm_num_gpus: int,
    intruder_cache_device_idx: int,
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
        logger.info(f"  - {exp.experiment_name}: {len(exp.saes)} SAE configurations")

    logger.debug(
        f"Evaluation parameters: skip_evaluation={skip_evaluation}, run_saebench={run_saebench}, run_intruder={run_intruder}"
    )

    if not skip_evaluation:
        logger.debug("🚀 Starting evaluation phase...")
        # Run evaluations
        for i, exp in enumerate(tqdm(experiments, desc="Evaluating experiments")):
            logger.debug(
                f"📊 Processing experiment {i + 1}/{len(experiments)}: {exp.experiment_name}"
            )

            if run_saebench:
                saebench_success = run_saebench_eval(
                    exp.experiment_name,
                    model_name,
                    saebench_eval_types or [],
                    saebench_batchsize,
                    num_autointerp_latents,
                    dtype,
                    seed,
                )
                if not saebench_success:
                    raise RuntimeError(
                        f"SAEBench evaluation failed for {exp.experiment_name}"
                    )

            if run_intruder:
                intruder_success = run_intruder_eval(
                    exp.experiment_name,
                    model_name,
                    dtype,
                    intruder_n_tokens,
                    intruder_batchsize,
                    intruder_n_latents,
                    intruder_vllm_num_gpus,
                    intruder_cache_device_idx,
                    seed,
                )
                if not intruder_success:
                    raise RuntimeError(
                        f"Intruder evaluation failed for {exp.experiment_name}"
                    )

        logger.debug("🏁 Evaluation phase complete!")
    else:
        logger.warning("⚠️ Evaluation skipped due to --skip-evaluation flag")

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
    create_visualizations(df, rankings)

    logger.info("✅ Evaluation complete!")


@arguably.command()
def eval_all_saes(
    *,
    model_name: str = "olmoe-i",
    run_saebench: bool = True,
    run_intruder: bool = True,
    saebench_eval_types: list[str] | None = None,
    saebench_batchsize: int = 64,
    num_autointerp_latents: int = 1000,
    intruder_n_tokens: int = 10_000_000,
    intruder_batchsize: int = 32,
    intruder_n_latents: int = 1000,
    intruder_vllm_num_gpus: int = 1,
    intruder_cache_device_idx: int = 1,
    dtype: str = "fp32",
    seed: int = 0,
    skip_evaluation: bool = False,
    log_level: str = "INFO",
) -> None:
    """Evaluate all SAE experiments and generate rankings.

    Args:
        model_name: Model name to evaluate on
        run_saebench: Whether to run SAEBench evaluation
        run_intruder: Whether to run intruder evaluation
        saebench_eval_types: List of SAEBench evaluation types to run
        saebench_batchsize: Batch size for SAEBench evaluation
        intruder_n_tokens: Number of tokens for intruder evaluation
        intruder_batchsize: Batch size for intruder evaluation
        intruder_n_latents: Number of latents for intruder evaluation
        intruder_vllm_num_gpus: Number of GPUs for VLLM (default: 1, uses device 0)
        intruder_cache_device_idx: Device index for caching model (default: 1, reserves 0 for VLLM)
        dtype: Data type for evaluation
        seed: Random seed
        skip_evaluation: Whether to skip evaluation and only aggregate results
        log_level: Logging level
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Run main
    main(
        model_name=model_name,
        run_saebench=run_saebench,
        run_intruder=run_intruder,
        saebench_eval_types=saebench_eval_types,
        saebench_batchsize=saebench_batchsize,
        num_autointerp_latents=num_autointerp_latents,
        intruder_n_tokens=intruder_n_tokens,
        intruder_batchsize=intruder_batchsize,
        intruder_n_latents=intruder_n_latents,
        intruder_vllm_num_gpus=intruder_vllm_num_gpus,
        intruder_cache_device_idx=intruder_cache_device_idx,
        dtype=dtype,
        seed=seed,
        skip_evaluation=skip_evaluation,
    )


if __name__ == "__main__":
    arguably.run()
