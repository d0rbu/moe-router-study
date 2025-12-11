"""Measure the degree of privileged-ness of various transformer bases using kurtosis.

This experiment computes kurtosis statistics for different bases:
1. Raw residual stream activations at each layer
2. MLP up-projection and gate-projection neurons
3. MLP down-projection neurons
4. Expert routers (per-layer and aggregated across all layers)
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import os
import sys
from typing import Any, cast

import arguably
from loguru import logger
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import numpy as np
import torch as th
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from core.model import get_model_config
from exp import OUTPUT_DIR
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys

KURTOSIS_DIR = os.path.join(OUTPUT_DIR, "kurtosis_basis")


@dataclass
class GlobalStats:
    """Statistics for computing kurtosis."""

    mean: th.Tensor
    std: th.Tensor


@dataclass
class Accumulator:
    """Accumulator for computing global statistics."""

    sum: th.Tensor | None = None
    sum_sq: th.Tensor | None = None
    count: int = 0


@dataclass
class FinalStats:
    """Final kurtosis statistics."""

    mean: float
    median: float
    std: float
    q25: float
    q75: float
    min: float
    max: float


def update_accumulator(
    accumulator: Accumulator, tensor: th.Tensor, batch_size: int
) -> None:
    """Update accumulator with new batch data."""
    if accumulator.sum is None:
        accumulator.sum = tensor.sum(dim=0)
        accumulator.sum_sq = (tensor * tensor).sum(dim=0)
    else:
        accumulator.sum += tensor.sum(dim=0)
        accumulator.sum_sq += (tensor * tensor).sum(dim=0)
    accumulator.count += batch_size


def compute_kurtosis(
    x: th.Tensor,
    dim: int = 0,
    mean: th.Tensor | None = None,
    std: th.Tensor | None = None,
) -> th.Tensor:
    """Compute kurtosis (Fisher's definition) along a dimension.

    Kurtosis = E[(X - μ)^4] / sigma^4 - 3

    Args:
        x: Input tensor
        dim: Dimension along which to compute kurtosis
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)

    Returns:
        Kurtosis values
    """
    # Compute or use provided mean and std
    if mean is None:
        mean = x.mean(dim=dim, keepdim=True)
    else:
        mean = mean.unsqueeze(dim) if mean.dim() == x.dim() - 1 else mean

    if std is None:
        std = x.std(dim=dim, keepdim=True, unbiased=False)
    else:
        std = std.unsqueeze(dim) if std.dim() == x.dim() - 1 else std

    # Compute normalized deviations
    z = (x - mean) / (std + 1e-8)

    # Compute kurtosis (excess kurtosis)
    kurtosis = (z**4).mean(dim=dim) - 3.0  # type: ignore[misc]

    return kurtosis


@arguably.command()
def kurtosis_basis(
    *,
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 100_000,
    reshuffled_tokens_per_file: int = 100_000,
    context_length: int = 2048,
    checkpoint_idx: int | None = None,
    device: str = "cpu",
    max_samples: int = 10_000_000,
    batch_size: int = 10_000,
    seed: int = 0,
    debug: bool = False,
    log_level: str = "INFO",
) -> None:
    """Compute kurtosis statistics for various transformer bases.

    Args:
        model_name: Name of the model to analyze
        dataset_name: Name of the dataset used for activations
        tokens_per_file: Tokens per file in original activations
        reshuffled_tokens_per_file: Tokens per file in reshuffled activations
        context_length: Context length for activations
        checkpoint_idx: Model checkpoint index (None for latest)
        device: Device to load model on
        max_samples: Maximum number of activation samples to use
        batch_size: Batch size for loading activations
        seed: Random seed for reshuffling
        debug: Debug mode (uses fewer files)
        log_level: Log level
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info(f"Running with log level: {log_level}")

    os.makedirs(KURTOSIS_DIR, exist_ok=True)

    # Get model config
    model_config = get_model_config(model_name)

    if checkpoint_idx is None:
        revision = None
    else:
        revision = str(model_config.checkpoints[checkpoint_idx])

    logger.info(f"Loading model {model_name} (revision={revision})")

    # Load model to get weight matrices
    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        device_map={"": device},
        revision=revision,
        dispatch=True,
    )

    router_layers: list[int] = model.layers_with_routers
    num_layers = len(cast("nn.ModuleList", model.layers))

    logger.info(f"Model has {num_layers} layers, {len(router_layers)} with routers")

    # Load activations
    logger.info("Loading activations...")
    activations, activation_dims = asyncio.run(
        load_activations_and_init_dist(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            submodule_names=[
                ActivationKeys.LAYER_OUTPUT,
                ActivationKeys.MLP_OUTPUT,
            ],
            context_length=context_length,
            seed=seed,
            debug=debug,
        )
    )

    logger.info(f"Activation dimensions: {activation_dims}")

    # Storage for kurtosis values
    results: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_idx": checkpoint_idx,
        "revision": revision,
        "num_layers": num_layers,
        "router_layers": router_layers,
        "max_samples": max_samples,
    }

    # Two-pass approach: First pass to compute means and stds, second pass to compute kurtosis
    logger.info("First pass: Computing means and standard deviations...")

    # Storage for means and stds for each basis type
    global_stats: dict[str, GlobalStats] = {}

    # Accumulators for computing global statistics
    accumulators: dict[str, Accumulator] = defaultdict(Accumulator)

    # First pass: accumulate statistics
    activation_iterator = activations(
        batch_size=batch_size, start_idx=0, max_samples=max_samples
    )

    num_batches_processed = 0

    with th.no_grad():
        for batch in tqdm(
            activation_iterator, desc="First pass - computing statistics",
        ):
            # Get activations
            # layer_outputs: (batch, num_layers, hidden_dim)
            layer_outputs = batch[ActivationKeys.LAYER_OUTPUT].to(device=device)
            # mlp_outputs: (batch, num_layers, hidden_dim)
            mlp_outputs = batch[ActivationKeys.MLP_OUTPUT].to(device=device)

            assert layer_outputs.shape[1] == mlp_outputs.shape[1] == num_layers, (
                "Number of layers mismatch: "
                f"Layer outputs shape: {layer_outputs.shape}, MLP outputs shape: {mlp_outputs.shape}, "
                f"Number of layers: {num_layers}, "
                f"Router layers: {router_layers}"
            )

            batch_size_actual = layer_outputs.shape[0]

            # Create list of (activation_tensor, basis_key) pairs to process
            activations_to_process = []

            # Process all layers in one loop
            for layer_idx in range(num_layers):
                # 1. Raw residual stream activations per layer
                # layer_acts: (batch, hidden_dim)
                layer_acts = layer_outputs[:, layer_idx, :]
                activations_to_process.append(
                    (layer_acts, f"layer_{layer_idx}_residual")
                )

                # Get pre-MLP residuals: layer_output - mlp_output (needed for both cases)
                pre_mlp_residuals = (
                    layer_outputs[:, layer_idx, :] - mlp_outputs[:, layer_idx, :]
                )

                # 2-3. MLP projections (dense layers) vs 4. Expert routers (MoE layers)
                if layer_idx not in router_layers:
                    # Dense MLP layer
                    up_w = (
                        cast("Tensor", model.mlps[layer_idx].up_proj.weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )
                    gate_w = (
                        cast("Tensor", model.mlps[layer_idx].gate_proj.weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )
                    down_w = (
                        cast("Tensor", model.mlps[layer_idx].down_proj.weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )

                    # Add projections to processing list
                    activations_to_process.append(
                        (pre_mlp_residuals @ up_w.T, f"layer_{layer_idx}_up_proj")
                    )
                    activations_to_process.append(
                        (pre_mlp_residuals @ gate_w.T, f"layer_{layer_idx}_gate_proj")
                    )
                    activations_to_process.append(
                        (layer_acts @ down_w, f"layer_{layer_idx}_down_proj")
                    )
                else:
                    # MoE layer - expert routers
                    router_weight = (
                        cast("Tensor", model.routers[layer_idx].weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )

                    # Compute router logits
                    router_logits = pre_mlp_residuals @ router_weight.T
                    activations_to_process.append(
                        (router_logits, f"layer_{layer_idx}_router")
                    )

            # Update all accumulators
            for tensor, basis_key in activations_to_process:
                update_accumulator(accumulators[basis_key], tensor, batch_size_actual)

            num_batches_processed += 1

    # Compute global means and stds
    logger.info("Computing global means and standard deviations...")
    for basis_key, acc in accumulators.items():
        count = acc.count
        mean = acc.sum / count
        var = (acc.sum_sq / count) - (mean**2)
        std = th.sqrt(th.clamp(var, min=1e-8))

        global_stats[basis_key] = GlobalStats(mean=mean, std=std)

    logger.info(f"Completed first pass with {num_batches_processed} batches")

    # Second pass: compute kurtosis using global statistics
    logger.info("Second pass: Computing kurtosis...")

    # Track kurtosis values by basis
    layerwise_kurtosis: dict[str, list[th.Tensor]] = defaultdict(list)

    # Second pass
    activation_iterator = activations(
        batch_size=batch_size, start_idx=0, max_samples=max_samples
    )

    second_pass_num_batches_processed = 0

    with th.no_grad():
        for batch in tqdm(activation_iterator, desc="Second pass - computing kurtosis", total=num_batches_processed):
            # Get activations
            layer_outputs = batch[ActivationKeys.LAYER_OUTPUT].to(device=device)
            mlp_outputs = batch[ActivationKeys.MLP_OUTPUT].to(device=device)

            # Create list of (activation_tensor, basis_key) pairs to process (same as first pass)
            activations_to_process = []

            # Process all layers in one loop
            for layer_idx in range(num_layers):
                # 1. Raw residual stream activations per layer
                # layer_acts: (batch, hidden_dim)
                layer_acts = layer_outputs[:, layer_idx, :]
                activations_to_process.append(
                    (layer_acts, f"layer_{layer_idx}_residual")
                )

                # Get pre-MLP residuals: layer_output - mlp_output (needed for both cases)
                pre_mlp_residuals = (
                    layer_outputs[:, layer_idx, :] - mlp_outputs[:, layer_idx, :]
                )

                # 2-3. MLP projections (dense layers) vs 4. Expert routers (MoE layers)
                if layer_idx not in router_layers:
                    # Dense MLP layer
                    up_w = (
                        cast("Tensor", model.mlps[layer_idx].up_proj.weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )
                    gate_w = (
                        cast("Tensor", model.mlps[layer_idx].gate_proj.weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )
                    down_w = (
                        cast("Tensor", model.mlps[layer_idx].down_proj.weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )

                    # Add projections to processing list
                    activations_to_process.append(
                        (pre_mlp_residuals @ up_w.T, f"layer_{layer_idx}_up_proj")
                    )
                    activations_to_process.append(
                        (pre_mlp_residuals @ gate_w.T, f"layer_{layer_idx}_gate_proj")
                    )
                    activations_to_process.append(
                        (layer_acts @ down_w, f"layer_{layer_idx}_down_proj")
                    )
                else:
                    # MoE layer - expert routers
                    router_weight = (
                        cast("Tensor", model.routers[layer_idx].weight)
                        .detach()
                        .to(dtype=pre_mlp_residuals.dtype)
                    )

                    # Compute router logits
                    router_logits = pre_mlp_residuals @ router_weight.T
                    activations_to_process.append(
                        (router_logits, f"layer_{layer_idx}_router")
                    )

            # Compute kurtosis for all activations
            for tensor, basis_key in activations_to_process:
                stats = global_stats[basis_key]
                kurtosis = compute_kurtosis(
                    tensor, dim=0, mean=stats.mean, std=stats.std
                )
                layerwise_kurtosis[basis_key].append(kurtosis)

            second_pass_num_batches_processed += 1

    logger.info(f"Completed second pass with {second_pass_num_batches_processed} batches")

    # Aggregate kurtosis values and compute statistics
    logger.info("Computing final statistics...")

    statistics: dict[str, FinalStats] = {}

    for basis_name, kurtosis_list in layerwise_kurtosis.items():
        # Concatenate all kurtosis values for this basis
        # Convert to float32 since quantile() requires float or double dtype
        all_kurtosis = th.cat(kurtosis_list, dim=0).float()

        # Compute statistics
        statistics[basis_name] = FinalStats(
            mean=float(all_kurtosis.mean().item()),
            median=float(all_kurtosis.median().item()),
            std=float(all_kurtosis.std().item()),
            q25=float(all_kurtosis.quantile(0.25).item()),
            q75=float(all_kurtosis.quantile(0.75).item()),
            min=float(all_kurtosis.min().item()),
            max=float(all_kurtosis.max().item()),
        )

    # Also compute aggregated statistics across all router layers
    logger.info("Computing aggregated router statistics...")
    all_router_kurtosis = []
    for layer_idx in router_layers:
        basis_name = f"layer_{layer_idx}_router"
        if basis_name in layerwise_kurtosis:
            all_router_kurtosis.extend(layerwise_kurtosis[basis_name])

    if all_router_kurtosis:
        # Convert to float32 since quantile() requires float or double dtype
        all_router_kurtosis_cat = th.cat(all_router_kurtosis, dim=0).float()
        statistics["all_layers_router"] = FinalStats(
            mean=float(all_router_kurtosis_cat.mean().item()),
            median=float(all_router_kurtosis_cat.median().item()),
            std=float(all_router_kurtosis_cat.std().item()),
            q25=float(all_router_kurtosis_cat.quantile(0.25).item()),
            q75=float(all_router_kurtosis_cat.quantile(0.75).item()),
            min=float(all_router_kurtosis_cat.min().item()),
            max=float(all_router_kurtosis_cat.max().item()),
        )

    results["statistics"] = statistics

    # Save results
    output_filename = f"kurtosis_{model_name}"
    if checkpoint_idx is not None:
        output_filename += f"_checkpoint{checkpoint_idx}"
    output_filename += ".pt"

    output_path = os.path.join(KURTOSIS_DIR, output_filename)
    th.save(results, output_path)
    logger.info(f"Saved results to {output_path}")

    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(results, output_filename.replace(".pt", ""))

    logger.info("Done!")


def create_visualizations(results: dict[str, Any], output_prefix: str) -> None:
    """Create bar charts with whiskers for kurtosis statistics.

    Args:
        results: Results dictionary with statistics
        output_prefix: Prefix for output filenames
    """
    statistics = results["statistics"]
    num_layers = results["num_layers"]
    router_layers = results["router_layers"]

    # 1. Plot: Residual stream kurtosis by layer, box-and-whisker style
    fig, ax = plt.subplots(figsize=(12, 6))

    residual_stats = [
        statistics[f"layer_{layer_idx}_residual"] for layer_idx in range(num_layers)
    ]

    medians = [s.median for s in residual_stats]
    q25s = [s.q25 for s in residual_stats]
    q75s = [s.q75 for s in residual_stats]

    x = np.arange(num_layers)
    ax.bar(x, medians, color="steelblue", alpha=0.7, label="Median kurtosis")

    # Add error bars for 25th-75th percentile
    yerr_lower = np.array(medians) - np.array(q25s)
    yerr_upper = np.array(q75s) - np.array(medians)
    ax.errorbar(
        x, medians, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="black", capsize=3
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Residual Stream Kurtosis by Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_residual_median.png"), dpi=150)
    plt.close()

    # 1.5. Plot: Residual stream kurtosis by layer, means + std
    fig, ax = plt.subplots(figsize=(12, 6))

    residual_stats = [
        statistics[f"layer_{layer_idx}_residual"] for layer_idx in range(num_layers)
    ]

    means = [s.mean for s in residual_stats]
    stds = [s.std for s in residual_stats]

    x = np.arange(num_layers)
    ax.bar(x, means, color="steelblue", alpha=0.7, label="Mean kurtosis")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Residual Stream Kurtosis by Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_residual_mean_std.png"), dpi=150)
    plt.close()

    # 2. Plot: MLP projections kurtosis by layer (only dense layers)
    dense_layers = [i for i in range(num_layers) if i not in router_layers]

    if dense_layers:
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        for ax, proj_type in zip(
            axes, ["up_proj", "gate_proj", "down_proj"], strict=False
        ):
            proj_stats = [
                statistics[f"layer_{layer_idx}_{proj_type}"]
                for layer_idx in dense_layers
                if f"layer_{layer_idx}_{proj_type}" in statistics
            ]

            means = [s.mean for s in proj_stats]
            q25s = [s.q25 for s in proj_stats]
            q75s = [s.q75 for s in proj_stats]

            x = np.arange(len(dense_layers))
            ax.bar(x, means, color="coral", alpha=0.7, label="Mean kurtosis")

            yerr_lower = np.array(means) - np.array(q25s)
            yerr_upper = np.array(q75s) - np.array(means)
            ax.errorbar(
                x,
                means,
                yerr=[yerr_lower, yerr_upper],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

            ax.set_xlabel("Dense Layer Index")
            ax.set_ylabel("Kurtosis")
            ax.set_title(
                f"MLP {proj_type.replace('_', ' ').title()} Kurtosis by Dense Layer"
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in dense_layers])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_mlp_projs.png"), dpi=150
        )
        plt.close()

    # 3. Plot: Router kurtosis (per-layer and aggregated)
    if router_layers:
        fig, ax = plt.subplots(figsize=(12, 6))

        router_stats_per_layer = [
            statistics[f"layer_{layer_idx}_router"] for layer_idx in router_layers
        ]

        means = [s.mean for s in router_stats_per_layer]
        q25s = [s.q25 for s in router_stats_per_layer]
        q75s = [s.q75 for s in router_stats_per_layer]

        x = np.arange(len(router_layers))
        ax.bar(
            x, means, color="mediumseagreen", alpha=0.7, label="Per-layer mean kurtosis"
        )

        yerr_lower = np.array(means) - np.array(q25s)
        yerr_upper = np.array(q75s) - np.array(means)
        ax.errorbar(
            x,
            means,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

        # Add aggregated router statistics
        if "all_layers_router" in statistics:
            agg_stats = statistics["all_layers_router"]
            agg_x = len(router_layers) + 0.5
            ax.bar(
                agg_x,
                agg_stats.mean,
                color="darkgreen",
                alpha=0.7,
                label="All layers aggregated",
            )
            ax.errorbar(
                agg_x,
                agg_stats.mean,
                yerr=[
                    [agg_stats.mean - agg_stats.q25],
                    [agg_stats.q75 - agg_stats.mean],
                ],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        ax.set_xlabel("Router Layer")
        ax.set_ylabel("Kurtosis")
        ax.set_title("Expert Router Kurtosis by Layer")
        ax.set_xticks(list(x) + [len(router_layers) + 0.5])
        ax.set_xticklabels([str(l) for l in router_layers] + ["All"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_router.png"), dpi=150)
        plt.close()

    # 4. Plot: Comparison across all basis types
    fig, ax = plt.subplots(figsize=(16, 8))

    # Collect all basis types with their statistics
    basis_types = []
    basis_means = []
    basis_q25s = []
    basis_q75s = []

    # Add residual streams
    for layer_idx in range(num_layers):
        basis_name = f"layer_{layer_idx}_residual"
        if basis_name in statistics:
            basis_types.append(f"L{layer_idx}_res")
            stats = statistics[basis_name]
            basis_means.append(stats.mean)
            basis_q25s.append(stats.q25)
            basis_q75s.append(stats.q75)

    # Add MLP projections (dense layers only)
    for layer_idx in dense_layers:
        for proj_type in ["up_proj", "gate_proj", "down_proj"]:
            basis_name = f"layer_{layer_idx}_{proj_type}"
            if basis_name in statistics:
                proj_abbrev = proj_type.replace("_proj", "").replace("_", "")[:2]
                basis_types.append(f"L{layer_idx}_{proj_abbrev}")
                stats = statistics[basis_name]
                basis_means.append(stats.mean)
                basis_q25s.append(stats.q25)
                basis_q75s.append(stats.q75)

    # Add routers
    for layer_idx in router_layers:
        basis_name = f"layer_{layer_idx}_router"
        if basis_name in statistics:
            basis_types.append(f"L{layer_idx}_rtr")
            stats = statistics[basis_name]
            basis_means.append(stats.mean)
            basis_q25s.append(stats.q25)
            basis_q75s.append(stats.q75)

    # Add aggregated router
    if "all_layers_router" in statistics:
        basis_types.append("All_rtr")
        stats = statistics["all_layers_router"]
        basis_means.append(stats.mean)
        basis_q25s.append(stats.q25)
        basis_q75s.append(stats.q75)

    if basis_types:  # Only create plot if we have data
        x = np.arange(len(basis_types))
        ax.bar(x, basis_means, color="slateblue", alpha=0.7)

        yerr_lower = np.array(basis_means) - np.array(basis_q25s)
        yerr_upper = np.array(basis_q75s) - np.array(basis_means)
        ax.errorbar(
            x,
            basis_means,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            capsize=2,
        )

        ax.set_xlabel("Basis Type")
        ax.set_ylabel("Kurtosis")
        ax.set_title("Kurtosis Comparison Across All Basis Types")
        ax.set_xticks(x)
        ax.set_xticklabels(basis_types, rotation=90, fontsize=6)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_comparison.png"), dpi=150
        )
        plt.close()

    logger.info(f"Saved visualizations to {KURTOSIS_DIR}")


if __name__ == "__main__":
    arguably.run()
