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


def compute_kurtosis_statistics(
    model_name: str,
    dataset_name: str,
    tokens_per_file: int,
    reshuffled_tokens_per_file: int,
    context_length: int,
    checkpoint_idx: int | None,
    device: str,
    max_samples: int,
    batch_size: int,
    seed: int,
    debug: bool,
) -> dict[str, Any]:
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

    Returns:
        Dictionary containing computed kurtosis statistics
    """
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

    # Get hidden dimension for random projections
    hidden_dim = activation_dims[ActivationKeys.LAYER_OUTPUT]

    # Generate random projection matrices (same for all layers)
    # 1. Random orthonormal matrix (QR decomposition of random matrix)
    random_normal = th.randn(hidden_dim, hidden_dim, device=device, dtype=th.float32)
    q_orthonormal, _ = th.linalg.qr(random_normal)
    random_orthonormal_matrix = q_orthonormal.T  # Transpose for right multiplication

    # 2. Random orthogonal matrix (QR decomposition of a different random matrix)
    random_normal_2 = th.randn(hidden_dim, hidden_dim, device=device, dtype=th.float32)
    q_orthogonal, _ = th.linalg.qr(random_normal_2)
    random_orthogonal_matrix = q_orthogonal.T  # Transpose for right multiplication

    # 3. Completely random matrix from normal distribution
    random_normal_matrix = th.randn(
        hidden_dim, hidden_dim, device=device, dtype=th.float32
    )
    random_normal_matrix = random_normal_matrix.T  # Transpose for right multiplication

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
            activation_iterator,
            desc="First pass - computing statistics",
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

                # Add random projections for this layer
                layer_acts_dtype = layer_acts.to(dtype=random_orthonormal_matrix.dtype)
                activations_to_process.append(
                    (
                        layer_acts_dtype @ random_orthonormal_matrix,
                        f"layer_{layer_idx}_random_orthonormal",
                    )
                )
                activations_to_process.append(
                    (
                        layer_acts_dtype @ random_orthogonal_matrix,
                        f"layer_{layer_idx}_random_orthogonal",
                    )
                )
                activations_to_process.append(
                    (
                        layer_acts_dtype @ random_normal_matrix,
                        f"layer_{layer_idx}_random_normal",
                    )
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
        for batch in tqdm(
            activation_iterator,
            desc="Second pass - computing kurtosis",
            total=num_batches_processed,
        ):
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

                # Add random projections for this layer
                layer_acts_dtype = layer_acts.to(dtype=random_orthonormal_matrix.dtype)
                activations_to_process.append(
                    (
                        layer_acts_dtype @ random_orthonormal_matrix,
                        f"layer_{layer_idx}_random_orthonormal",
                    )
                )
                activations_to_process.append(
                    (
                        layer_acts_dtype @ random_orthogonal_matrix,
                        f"layer_{layer_idx}_random_orthogonal",
                    )
                )
                activations_to_process.append(
                    (
                        layer_acts_dtype @ random_normal_matrix,
                        f"layer_{layer_idx}_random_normal",
                    )
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

    logger.info(
        f"Completed second pass with {second_pass_num_batches_processed} batches"
    )

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

    return results


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

    # Determine output filename
    output_filename = f"kurtosis_{model_name}"
    if checkpoint_idx is not None:
        output_filename += f"_checkpoint{checkpoint_idx}"
    output_filename += ".pt"

    output_path = os.path.join(KURTOSIS_DIR, output_filename)

    # Check if cached results exist
    if os.path.exists(output_path):
        logger.info(f"Loading cached results from {output_path}")
        results = th.load(output_path, weights_only=False)
    else:
        logger.info("No cached results found. Computing kurtosis statistics...")
        results = compute_kurtosis_statistics(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            context_length=context_length,
            checkpoint_idx=checkpoint_idx,
            device=device,
            max_samples=max_samples,
            batch_size=batch_size,
            seed=seed,
            debug=debug,
        )

        # Save results
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
    plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_residual.png"), dpi=150)
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
    plt.savefig(
        os.path.join(KURTOSIS_DIR, f"{output_prefix}_residual_mean_std.png"), dpi=150
    )
    plt.close()

    # 1.75. Plot: Random projection kurtosis comparison (orthonormal, orthogonal, normal)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Get stats for each random projection type
    projection_types = [
        ("random_orthonormal", "Random Orthonormal (QR)", "purple"),
        ("random_orthogonal", "Random Orthogonal (Normalized)", "magenta"),
        ("random_normal", "Random Normal", "pink"),
    ]

    # First subplot: Medians with Q25/Q75
    ax = axes[0]
    for proj_type, label, color in projection_types:
        proj_stats = [
            statistics[f"layer_{layer_idx}_{proj_type}"]
            for layer_idx in range(num_layers)
            if f"layer_{layer_idx}_{proj_type}" in statistics
        ]
        if proj_stats:
            medians = [s.median for s in proj_stats]
            q25s = [s.q25 for s in proj_stats]
            q75s = [s.q75 for s in proj_stats]
            x = np.arange(len(proj_stats))
            ax.plot(
                x, medians, marker="o", label=label, color=color, alpha=0.7, linewidth=2
            )
            ax.fill_between(x, q25s, q75s, alpha=0.2, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Random Projection Kurtosis Comparison - Median with Q25/Q75")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Second subplot: Means with std
    ax = axes[1]
    for proj_type, label, color in projection_types:
        proj_stats = [
            statistics[f"layer_{layer_idx}_{proj_type}"]
            for layer_idx in range(num_layers)
            if f"layer_{layer_idx}_{proj_type}" in statistics
        ]
        if proj_stats:
            means = [s.mean for s in proj_stats]
            stds = [s.std for s in proj_stats]
            x = np.arange(len(proj_stats))
            ax.plot(
                x, means, marker="o", label=label, color=color, alpha=0.7, linewidth=2
            )
            ax.fill_between(
                x,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Random Projection Kurtosis Comparison - Mean with Std")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(KURTOSIS_DIR, f"{output_prefix}_random_projections.png"), dpi=150
    )
    plt.close()

    # 2. Plot: MLP projections kurtosis by layer (only dense layers), box-and-whisker style and means + std
    dense_layers = [i for i in range(num_layers) if i not in router_layers]

    if dense_layers:
        # Box-and-whisker style
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        for ax, proj_type in zip(
            axes, ["up_proj", "gate_proj", "down_proj"], strict=False
        ):
            proj_stats = [
                statistics[f"layer_{layer_idx}_{proj_type}"]
                for layer_idx in dense_layers
                if f"layer_{layer_idx}_{proj_type}" in statistics
            ]

            medians = [s.median for s in proj_stats]
            q25s = [s.q25 for s in proj_stats]
            q75s = [s.q75 for s in proj_stats]

            x = np.arange(len(dense_layers))
            ax.bar(x, medians, color="coral", alpha=0.7, label="Median kurtosis")

            yerr_lower = np.array(medians) - np.array(q25s)
            yerr_upper = np.array(q75s) - np.array(medians)
            ax.errorbar(
                x,
                medians,
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
            ax.set_xticklabels([str(layer_idx) for layer_idx in dense_layers])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_mlp_projs.png"), dpi=150
        )
        plt.close()

        # Means + std
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        for ax, proj_type in zip(
            axes, ["up_proj", "gate_proj", "down_proj"], strict=False
        ):
            proj_stats = [
                statistics[f"layer_{layer_idx}_{proj_type}"]
                for layer_idx in dense_layers
                if f"layer_{layer_idx}_{proj_type}" in statistics
            ]

            means = [s.mean for s in proj_stats]
            stds = [s.std for s in proj_stats]

            x = np.arange(len(dense_layers))
            ax.bar(x, means, color="coral", alpha=0.7, label="Mean kurtosis")
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=3)

            ax.set_xlabel("Dense Layer Index")
            ax.set_ylabel("Kurtosis")
            ax.set_title(
                f"MLP {proj_type.replace('_', ' ').title()} Kurtosis by Dense Layer"
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(layer_idx) for layer_idx in dense_layers])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_mlp_projs_mean_std.png"),
            dpi=150,
        )
        plt.close()

    # 3. Plot: Router kurtosis (per-layer and aggregated), box-and-whisker style and means + std
    if router_layers:
        fig, ax = plt.subplots(figsize=(12, 6))

        router_stats_per_layer = [
            statistics[f"layer_{layer_idx}_router"] for layer_idx in router_layers
        ]

        medians = [s.median for s in router_stats_per_layer]
        q25s = [s.q25 for s in router_stats_per_layer]
        q75s = [s.q75 for s in router_stats_per_layer]

        x = np.arange(len(router_layers))
        ax.bar(
            x,
            medians,
            color="mediumseagreen",
            alpha=0.7,
            label="Per-layer median kurtosis",
        )

        yerr_lower = np.array(medians) - np.array(q25s)
        yerr_upper = np.array(q75s) - np.array(medians)
        ax.errorbar(
            x,
            medians,
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
                agg_stats.median,
                color="darkgreen",
                alpha=0.7,
                label="All layers aggregated",
            )
            ax.errorbar(
                agg_x,
                agg_stats.median,
                yerr=[
                    [agg_stats.median - agg_stats.q25],
                    [agg_stats.q75 - agg_stats.median],
                ],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        ax.set_xlabel("Router Layer")
        ax.set_ylabel("Kurtosis")
        ax.set_title("Expert Router Kurtosis by Layer")
        ax.set_xticks([*list(x), len(router_layers) + 0.5])
        ax.set_xticklabels([str(layer_idx) for layer_idx in router_layers] + ["All"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_router.png"), dpi=150)
        plt.close()

        # Means + std
        fig, ax = plt.subplots(figsize=(12, 6))

        router_stats_per_layer = [
            statistics[f"layer_{layer_idx}_router"] for layer_idx in router_layers
        ]
        means = [s.mean for s in router_stats_per_layer]
        stds = [s.std for s in router_stats_per_layer]
        x = np.arange(len(router_layers))
        ax.bar(x, means, color="mediumseagreen", alpha=0.7, label="Mean kurtosis")
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=3)
        ax.set_xlabel("Router Layer")
        ax.set_ylabel("Kurtosis")
        ax.set_title("Expert Router Kurtosis by Layer")
        ax.set_xticks([*list(x), len(router_layers) + 0.5])
        ax.set_xticklabels([str(layer_idx) for layer_idx in router_layers] + ["All"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_router_mean_std.png"), dpi=150
        )
        plt.close()

    # 4. Plot: Comparison across all basis types (grouped bar chart)
    # Version 1: Q25, median, Q75 (box-and-whisker style)
    # Version 2: Mean with std error bars

    # Define bar categories
    bar_categories = ["Residual", "Up", "Gate", "Down", "Router"]
    num_categories = len(bar_categories)
    bar_width = 0.15

    # Colors for each category (distinct colors for Up, Gate, Down)
    colors = {
        "Residual": "steelblue",
        "Up": "coral",
        "Gate": "orange",
        "Down": "tomato",
        "Router": "mediumseagreen",
    }

    # Collect data organized by layer groups
    # Each group stores both median/q25/q75 and mean/std
    layer_groups = []
    group_labels = []

    # Process each layer
    for layer_idx in range(num_layers):
        group_data_median = {}
        group_data_mean = {}
        group_errors_q25_q75 = {}
        group_errors_std = {}

        # Residual stream (always present)
        basis_name = f"layer_{layer_idx}_residual"
        if basis_name in statistics:
            stats = statistics[basis_name]
            group_data_median["Residual"] = stats.median
            group_data_mean["Residual"] = stats.mean
            group_errors_q25_q75["Residual"] = (
                stats.median - stats.q25,
                stats.q75 - stats.median,
            )
            group_errors_std["Residual"] = stats.std

        # MLP projections (only for dense layers)
        if layer_idx in dense_layers:
            for proj_type in ["up_proj", "gate_proj", "down_proj"]:
                basis_name = f"layer_{layer_idx}_{proj_type}"
                if basis_name in statistics:
                    stats = statistics[basis_name]
                    category = proj_type.replace("_proj", "").title()
                    group_data_median[category] = stats.median
                    group_data_mean[category] = stats.mean
                    group_errors_q25_q75[category] = (
                        stats.median - stats.q25,
                        stats.q75 - stats.median,
                    )
                    group_errors_std[category] = stats.std

        # Router (only for router layers)
        if layer_idx in router_layers:
            basis_name = f"layer_{layer_idx}_router"
            if basis_name in statistics:
                stats = statistics[basis_name]
                group_data_median["Router"] = stats.median
                group_data_mean["Router"] = stats.mean
                group_errors_q25_q75["Router"] = (
                    stats.median - stats.q25,
                    stats.q75 - stats.median,
                )
                group_errors_std["Router"] = stats.std

        if group_data_median:  # Only add if we have data for this layer
            layer_groups.append(
                {
                    "median": group_data_median,
                    "mean": group_data_mean,
                    "q25_q75": group_errors_q25_q75,
                    "std": group_errors_std,
                }
            )
            group_labels.append(f"L{layer_idx}")

    # Add aggregated group (if available)
    aggregated_data_median = {}
    aggregated_data_mean = {}
    aggregated_errors_q25_q75 = {}
    aggregated_errors_std = {}
    has_aggregated = False

    # Aggregate residual streams across all layers
    residual_medians = []
    residual_means = []
    residual_q25s = []
    residual_q75s = []
    residual_stds = []
    for layer_idx in range(num_layers):
        basis_name = f"layer_{layer_idx}_residual"
        if basis_name in statistics:
            stats = statistics[basis_name]
            residual_medians.append(stats.median)
            residual_means.append(stats.mean)
            residual_q25s.append(stats.q25)
            residual_q75s.append(stats.q75)
            residual_stds.append(stats.std)
    if residual_medians:
        aggregated_data_median["Residual"] = np.mean(residual_medians)
        aggregated_data_mean["Residual"] = np.mean(residual_means)
        # Use mean of error ranges
        mean_err_lower = np.mean(
            [m - q25 for m, q25 in zip(residual_medians, residual_q25s, strict=False)]
        )
        mean_err_upper = np.mean(
            [q75 - m for m, q75 in zip(residual_medians, residual_q75s, strict=False)]
        )
        aggregated_errors_q25_q75["Residual"] = (mean_err_lower, mean_err_upper)
        aggregated_errors_std["Residual"] = np.mean(residual_stds)
        has_aggregated = True

    # Aggregate MLP projections across all dense layers
    for proj_type in ["up_proj", "gate_proj", "down_proj"]:
        proj_medians = []
        proj_means = []
        proj_q25s = []
        proj_q75s = []
        proj_stds = []
        for layer_idx in dense_layers:
            basis_name = f"layer_{layer_idx}_{proj_type}"
            if basis_name in statistics:
                stats = statistics[basis_name]
                proj_medians.append(stats.median)
                proj_means.append(stats.mean)
                proj_q25s.append(stats.q25)
                proj_q75s.append(stats.q75)
                proj_stds.append(stats.std)
        if proj_medians:
            category = proj_type.replace("_proj", "").title()
            aggregated_data_median[category] = np.mean(proj_medians)
            aggregated_data_mean[category] = np.mean(proj_means)
            mean_err_lower = np.mean(
                [m - q25 for m, q25 in zip(proj_medians, proj_q25s, strict=False)]
            )
            mean_err_upper = np.mean(
                [q75 - m for m, q75 in zip(proj_medians, proj_q75s, strict=False)]
            )
            aggregated_errors_q25_q75[category] = (mean_err_lower, mean_err_upper)
            aggregated_errors_std[category] = np.mean(proj_stds)
            has_aggregated = True

    # Aggregate routers
    if "all_layers_router" in statistics:
        stats = statistics["all_layers_router"]
        aggregated_data_median["Router"] = stats.median
        aggregated_data_mean["Router"] = stats.mean
        aggregated_errors_q25_q75["Router"] = (
            stats.median - stats.q25,
            stats.q75 - stats.median,
        )
        aggregated_errors_std["Router"] = stats.std
        has_aggregated = True

    if has_aggregated:
        layer_groups.append(
            {
                "median": aggregated_data_median,
                "mean": aggregated_data_mean,
                "q25_q75": aggregated_errors_q25_q75,
                "std": aggregated_errors_std,
            }
        )
        group_labels.append("All")

    if layer_groups:  # Only create plot if we have data
        num_groups = len(layer_groups)
        x_base = np.arange(num_groups)

        # Version 1: Q25, median, Q75 (box-and-whisker style)
        fig, ax = plt.subplots(figsize=(20, 8))

        # Plot bars for each category
        for cat_idx, category in enumerate(bar_categories):
            values = []
            errors_lower = []
            errors_upper = []
            x_positions = []

            for group_idx, group in enumerate(layer_groups):
                if category in group["median"]:
                    values.append(group["median"][category])
                    err_lower, err_upper = group["q25_q75"].get(category, (0, 0))
                    errors_lower.append(err_lower)
                    errors_upper.append(err_upper)
                    x_positions.append(group_idx)

            if values:  # Only plot if we have values for this category
                x_pos = (
                    x_base[x_positions]
                    + (cat_idx - num_categories / 2 + 0.5) * bar_width
                )
                ax.bar(
                    x_pos,
                    values,
                    bar_width,
                    label=category,
                    color=colors[category],
                    alpha=0.7,
                )
                # Add error bars
                ax.errorbar(
                    x_pos,
                    values,
                    yerr=[errors_lower, errors_upper],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                )

        ax.set_xlabel("Layer Group")
        ax.set_ylabel("Kurtosis")
        ax.set_title(
            "Kurtosis Comparison Across All Basis Types (Grouped by Layer) - Median with Q25/Q75"
        )
        ax.set_xticks(x_base)
        ax.set_xticklabels(group_labels)
        ax.legend(title="Basis Type", loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_comparison.png"), dpi=150
        )
        plt.close()

        # Version 2: Mean with std error bars
        fig, ax = plt.subplots(figsize=(20, 8))

        # Plot bars for each category
        for cat_idx, category in enumerate(bar_categories):
            values = []
            errors_std = []
            x_positions = []

            for group_idx, group in enumerate(layer_groups):
                if category in group["mean"]:
                    values.append(group["mean"][category])
                    errors_std.append(group["std"].get(category, 0))
                    x_positions.append(group_idx)

            if values:  # Only plot if we have values for this category
                x_pos = (
                    x_base[x_positions]
                    + (cat_idx - num_categories / 2 + 0.5) * bar_width
                )
                ax.bar(
                    x_pos,
                    values,
                    bar_width,
                    label=category,
                    color=colors[category],
                    alpha=0.7,
                )
                # Add error bars
                ax.errorbar(
                    x_pos,
                    values,
                    yerr=errors_std,
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                )

        ax.set_xlabel("Layer Group")
        ax.set_ylabel("Kurtosis")
        ax.set_title(
            "Kurtosis Comparison Across All Basis Types (Grouped by Layer) - Mean with Std"
        )
        ax.set_xticks(x_base)
        ax.set_xticklabels(group_labels)
        ax.legend(title="Basis Type", loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(KURTOSIS_DIR, f"{output_prefix}_comparison_mean_std.png"),
            dpi=150,
        )
        plt.close()

    logger.info(f"Saved visualizations to {KURTOSIS_DIR}")


if __name__ == "__main__":
    arguably.run()
