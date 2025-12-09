"""Measure the degree of privileged-ness of various transformer bases using kurtosis.

This experiment computes kurtosis statistics for different bases:
1. Raw residual stream activations at each layer
2. MLP up-projection and gate-projection neurons
3. MLP down-projection neurons
4. Expert routers (per-layer and aggregated across all layers)
"""

import asyncio
from collections import defaultdict
import os
from typing import Any, cast

try:
    from typing import assert_type  # Python 3.11+
except ImportError:
    from typing_extensions import assert_type

import arguably
from loguru import logger
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import numpy as np
import torch as th
from torch import Tensor
from tqdm import tqdm

from core.model import get_model_config
from exp import OUTPUT_DIR
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys

KURTOSIS_DIR = os.path.join(OUTPUT_DIR, "kurtosis_basis")


def compute_kurtosis(x: th.Tensor, dim: int = 0) -> th.Tensor:
    """Compute kurtosis (Fisher's definition) along a dimension.

    Kurtosis = E[(X - μ)^4] / sigma^4 - 3

    Args:
        x: Input tensor
        dim: Dimension along which to compute kurtosis

    Returns:
        Kurtosis values
    """
    # Compute mean and std
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False)

    # Compute normalized deviations
    z = (x - mean) / (std + 1e-8)

    # Compute kurtosis (excess kurtosis)
    kurtosis = (z**4).mean(dim=dim) - 3.0  # type: ignore[misc]

    return kurtosis


@arguably.command()
def kurtosis_basis(
    model_name: str = "olmoe",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    checkpoint_idx: int | None = None,
    device: str = "cpu",
    max_samples: int = 100_000,
    batch_size: int = 4096,
    seed: int = 0,
    debug: bool = False,
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
    """
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
        device_map=device,
        revision=revision,
    )

    router_layers: list[int] = model.layers_with_routers
    # Get num_layers from model config or estimate from router layers
    try:
        num_layers = len(model.layers)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        # Fallback: estimate from router layers if model.layers is not sized
        num_layers = max(router_layers) + 1 if router_layers else 0

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
                ActivationKeys.ATTN_OUTPUT,
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

    # Track statistics by layer for some bases
    layerwise_kurtosis: dict[str, dict[int, list[th.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Process activations in batches
    logger.info(f"Processing activations (max_samples={max_samples})...")

    activation_iterator = activations(
        batch_size=batch_size, start_idx=0, max_samples=max_samples
    )

    num_batches_processed = 0

    with th.no_grad():
        for batch in tqdm(activation_iterator, desc="Processing batches"):
            # Get layer outputs (post-layer-norm residual stream at each layer)
            # Shape: (batch, num_layers, hidden_dim)
            layer_outputs = batch[ActivationKeys.LAYER_OUTPUT]

            # Get attention outputs (post-attention residual stream)
            # Shape: (batch, num_layers, hidden_dim)
            attn_outputs = batch[ActivationKeys.ATTN_OUTPUT]

            # Get MLP outputs (post-MLP residual stream)
            # Shape: (batch, num_layers, hidden_dim)
            mlp_outputs = batch[ActivationKeys.MLP_OUTPUT]

            # 1. Raw residual stream activations per layer
            for layer_idx in range(num_layers):
                # Compute kurtosis for each element in the residual stream
                layer_acts = layer_outputs[:, layer_idx, :]  # (batch, hidden_dim)
                layer_kurtosis = compute_kurtosis(layer_acts, dim=0)  # (hidden_dim,)

                layerwise_kurtosis[f"layer_{layer_idx}_residual"][layer_idx].append(
                    layer_kurtosis.cpu()
                )

            # 2. MLP up-projection and gate-projection neurons
            for layer_idx in tqdm(
                range(num_layers), desc="MLP projections", leave=False
            ):
                # Get post-attention, pre-MLP residuals
                pre_mlp_residuals = attn_outputs[:, layer_idx, :]  # (batch, hidden_dim)

                # Check if this layer has MLP experts (MoE layer)
                if layer_idx in router_layers:
                    # MoE layer - get weights from experts
                    experts = model.mlps[layer_idx].experts

                    # Get number of experts
                    if isinstance(experts, list):
                        num_experts = len(experts)
                    else:
                        # Try to count experts
                        num_experts = 0
                        while True:
                            try:
                                if isinstance(experts, list):
                                    _ = experts[num_experts]
                                else:
                                    try:
                                        _ = getattr(experts, str(num_experts))
                                    except AttributeError:
                                        _ = experts[str(num_experts)]
                                num_experts += 1
                            except (IndexError, KeyError):
                                break

                    # Accumulate projections across all experts
                    all_up_projs = []
                    all_gate_projs = []

                    for expert_idx in range(num_experts):
                        # Get expert
                        if isinstance(experts, list):
                            expert = experts[expert_idx]
                        else:
                            try:
                                expert = getattr(experts, str(expert_idx))
                            except AttributeError:
                                expert = experts[str(expert_idx)]

                        # Get weights
                        up_w = cast(
                            "Tensor", expert.up_proj.weight
                        ).detach()  # (mlp_dim, hidden_dim)
                        gate_w = cast(
                            "Tensor", expert.gate_proj.weight
                        ).detach()  # (mlp_dim, hidden_dim)

                        # Project: (batch, mlp_dim) = (batch, hidden_dim) @ (hidden_dim, mlp_dim)
                        up_proj = pre_mlp_residuals @ up_w.T.to(
                            pre_mlp_residuals.device
                        )
                        gate_proj = pre_mlp_residuals @ gate_w.T.to(
                            pre_mlp_residuals.device
                        )

                        all_up_projs.append(up_proj)
                        all_gate_projs.append(gate_proj)

                    # Concatenate all expert projections
                    # Shape: (batch, num_experts * mlp_dim)
                    all_up_projs_cat = th.cat(all_up_projs, dim=1)
                    all_gate_projs_cat = th.cat(all_gate_projs, dim=1)

                    # Compute kurtosis
                    up_kurtosis = compute_kurtosis(all_up_projs_cat, dim=0)
                    gate_kurtosis = compute_kurtosis(all_gate_projs_cat, dim=0)

                    layerwise_kurtosis[f"layer_{layer_idx}_up_proj"][layer_idx].append(
                        up_kurtosis.cpu()
                    )
                    layerwise_kurtosis[f"layer_{layer_idx}_gate_proj"][
                        layer_idx
                    ].append(gate_kurtosis.cpu())
                else:
                    # Dense MLP layer
                    up_w = cast("Tensor", model.mlps[layer_idx].up_proj.weight).detach()
                    gate_w = cast(
                        "Tensor", model.mlps[layer_idx].gate_proj.weight
                    ).detach()

                    up_proj = pre_mlp_residuals @ up_w.T.to(pre_mlp_residuals.device)
                    gate_proj = pre_mlp_residuals @ gate_w.T.to(
                        pre_mlp_residuals.device
                    )

                    up_kurtosis = compute_kurtosis(up_proj, dim=0)
                    gate_kurtosis = compute_kurtosis(gate_proj, dim=0)

                    layerwise_kurtosis[f"layer_{layer_idx}_up_proj"][layer_idx].append(
                        up_kurtosis.cpu()
                    )
                    layerwise_kurtosis[f"layer_{layer_idx}_gate_proj"][
                        layer_idx
                    ].append(gate_kurtosis.cpu())

            # 3. MLP down-projection neurons
            for layer_idx in tqdm(range(num_layers), desc="MLP down-proj", leave=False):
                # Get post-MLP residuals
                post_mlp_residuals = mlp_outputs[:, layer_idx, :]  # (batch, hidden_dim)

                # Check if this layer has MLP experts (MoE layer)
                if layer_idx in router_layers:
                    # MoE layer
                    experts = model.mlps[layer_idx].experts

                    # Get number of experts
                    if isinstance(experts, list):
                        num_experts = len(experts)
                    else:
                        num_experts = 0
                        while True:
                            try:
                                if isinstance(experts, list):
                                    _ = experts[num_experts]
                                else:
                                    try:
                                        _ = getattr(experts, str(num_experts))
                                    except AttributeError:
                                        _ = experts[str(num_experts)]
                                num_experts += 1
                            except (IndexError, KeyError):
                                break

                    # Accumulate projections across all experts
                    all_down_projs = []

                    for expert_idx in range(num_experts):
                        # Get expert
                        if isinstance(experts, list):
                            expert = experts[expert_idx]
                        else:
                            try:
                                expert = getattr(experts, str(expert_idx))
                            except AttributeError:
                                expert = experts[str(expert_idx)]

                        # Get weights (transpose for "reading" from down_proj)
                        down_w = cast(
                            "Tensor", expert.down_proj.weight
                        ).detach()  # (hidden_dim, mlp_dim)

                        # Project: (batch, mlp_dim) = (batch, hidden_dim) @ (hidden_dim, mlp_dim)
                        # We want to see what neurons the residual reads FROM
                        down_proj = post_mlp_residuals @ down_w.to(
                            post_mlp_residuals.device
                        )

                        all_down_projs.append(down_proj)

                    # Concatenate all expert projections
                    all_down_projs_cat = th.cat(all_down_projs, dim=1)

                    # Compute kurtosis
                    down_kurtosis = compute_kurtosis(all_down_projs_cat, dim=0)

                    layerwise_kurtosis[f"layer_{layer_idx}_down_proj"][
                        layer_idx
                    ].append(down_kurtosis.cpu())
                else:
                    # Dense MLP layer
                    down_w = cast(
                        "Tensor", model.mlps[layer_idx].down_proj.weight
                    ).detach()

                    down_proj = post_mlp_residuals @ down_w.to(
                        post_mlp_residuals.device
                    )

                    down_kurtosis = compute_kurtosis(down_proj, dim=0)

                    layerwise_kurtosis[f"layer_{layer_idx}_down_proj"][
                        layer_idx
                    ].append(down_kurtosis.cpu())

            # 4. Expert routers (per-layer)
            for layer_idx in tqdm(router_layers, desc="Expert routers", leave=False):
                # Get post-attention residuals for routing
                pre_mlp_residuals = attn_outputs[:, layer_idx, :]  # (batch, hidden_dim)

                # Get router weights
                router_weight = cast(
                    "Tensor", model.routers[layer_idx].weight
                ).detach()  # (num_experts, hidden_dim)

                # Compute router logits: (batch, num_experts) = (batch, hidden_dim) @ (hidden_dim, num_experts)
                router_logits = pre_mlp_residuals @ router_weight.T.to(
                    pre_mlp_residuals.device
                )

                # Compute kurtosis
                router_kurtosis = compute_kurtosis(router_logits, dim=0)

                layerwise_kurtosis[f"layer_{layer_idx}_router"][layer_idx].append(
                    router_kurtosis.cpu()
                )

            num_batches_processed += 1

    logger.info(f"Processed {num_batches_processed} batches")

    # Aggregate kurtosis values and compute statistics
    logger.info("Computing statistics...")

    # For each basis type, concatenate all kurtosis values and compute stats
    statistics: dict[str, dict[str, float]] = {}

    for basis_name, layer_dict in layerwise_kurtosis.items():
        for kurtosis_list in layer_dict.values():
            # Concatenate all kurtosis values for this basis
            all_kurtosis = th.cat(kurtosis_list, dim=0)

            # Compute statistics
            stats = {
                "mean": float(all_kurtosis.mean().item()),
                "median": float(all_kurtosis.median().item()),
                "std": float(all_kurtosis.std().item()),
                "q25": float(all_kurtosis.quantile(0.25).item()),
                "q75": float(all_kurtosis.quantile(0.75).item()),
                "min": float(all_kurtosis.min().item()),
                "max": float(all_kurtosis.max().item()),
            }

            statistics[basis_name] = stats

    # Also compute aggregated statistics across all router layers
    logger.info("Computing aggregated router statistics...")
    all_router_kurtosis = []
    for layer_idx in router_layers:
        basis_name = f"layer_{layer_idx}_router"
        if basis_name in layerwise_kurtosis:
            kurtosis_list = layerwise_kurtosis[basis_name][layer_idx]
            all_router_kurtosis.extend(kurtosis_list)

    if all_router_kurtosis:
        all_router_kurtosis_cat = th.cat(all_router_kurtosis, dim=0)
        statistics["all_layers_router"] = {
            "mean": float(all_router_kurtosis_cat.mean().item()),
            "median": float(all_router_kurtosis_cat.median().item()),
            "std": float(all_router_kurtosis_cat.std().item()),
            "q25": float(all_router_kurtosis_cat.quantile(0.25).item()),
            "q75": float(all_router_kurtosis_cat.quantile(0.75).item()),
            "min": float(all_router_kurtosis_cat.min().item()),
            "max": float(all_router_kurtosis_cat.max().item()),
        }

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

    # 1. Plot: Residual stream kurtosis by layer
    fig, ax = plt.subplots(figsize=(12, 6))

    residual_stats = [
        statistics.get(f"layer_{layer_idx}_residual", {})
        for layer_idx in range(num_layers)
    ]

    means = [s.get("mean", 0) for s in residual_stats]
    q25s = [s.get("q25", 0) for s in residual_stats]
    q75s = [s.get("q75", 0) for s in residual_stats]

    x = np.arange(num_layers)
    ax.bar(x, means, color="steelblue", alpha=0.7, label="Mean kurtosis")

    # Add error bars for 25th-75th percentile
    yerr_lower = np.array(means) - np.array(q25s)
    yerr_upper = np.array(q75s) - np.array(means)
    ax.errorbar(
        x, means, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="black", capsize=3
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Residual Stream Kurtosis by Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_residual.png"), dpi=150)
    plt.close()

    # 2. Plot: MLP projections kurtosis by layer
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    for ax, proj_type in zip(axes, ["up_proj", "gate_proj", "down_proj"], strict=False):
        proj_stats = [
            statistics.get(f"layer_{layer_idx}_{proj_type}", {})
            for layer_idx in range(num_layers)
        ]

        means = [s.get("mean", 0) for s in proj_stats]
        q25s = [s.get("q25", 0) for s in proj_stats]
        q75s = [s.get("q75", 0) for s in proj_stats]

        x = np.arange(num_layers)
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

        ax.set_xlabel("Layer")
        ax.set_ylabel("Kurtosis")
        ax.set_title(f"MLP {proj_type.replace('_', ' ').title()} Kurtosis by Layer")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_mlp_projs.png"), dpi=150)
    plt.close()

    # 3. Plot: Router kurtosis (per-layer and aggregated)
    fig, ax = plt.subplots(figsize=(12, 6))

    router_stats_per_layer = [
        statistics.get(f"layer_{layer_idx}_router", {}) for layer_idx in router_layers
    ]

    means = [s.get("mean", 0) for s in router_stats_per_layer]
    q25s = [s.get("q25", 0) for s in router_stats_per_layer]
    q75s = [s.get("q75", 0) for s in router_stats_per_layer]

    x = np.arange(len(router_layers))
    ax.bar(x, means, color="mediumseagreen", alpha=0.7, label="Per-layer mean kurtosis")

    yerr_lower = np.array(means) - np.array(q25s)
    yerr_upper = np.array(q75s) - np.array(means)
    ax.errorbar(
        x, means, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="black", capsize=3
    )

    # Add aggregated router statistics
    if "all_layers_router" in statistics:
        agg_stats = statistics["all_layers_router"]
        agg_x = len(router_layers) + 0.5
        ax.bar(
            agg_x,
            agg_stats["mean"],
            color="darkgreen",
            alpha=0.7,
            label="All layers aggregated",
        )
        ax.errorbar(
            agg_x,
            agg_stats["mean"],
            yerr=[
                [agg_stats["mean"] - agg_stats["q25"]],
                [agg_stats["q75"] - agg_stats["mean"]],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    ax.set_xlabel("Router Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Expert Router Kurtosis by Layer")
    ax.set_xticks([*list(x), len(router_layers) + 0.5])
    ax.set_xticklabels([str(layer) for layer in router_layers] + ["All"])
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
            basis_means.append(stats["mean"])
            basis_q25s.append(stats["q25"])
            basis_q75s.append(stats["q75"])

    # Add MLP projections
    for layer_idx in range(num_layers):
        for proj_type in ["up_proj", "gate_proj", "down_proj"]:
            basis_name = f"layer_{layer_idx}_{proj_type}"
            if basis_name in statistics:
                proj_abbrev = proj_type.replace("_proj", "").replace("_", "")[:2]
                basis_types.append(f"L{layer_idx}_{proj_abbrev}")
                stats = statistics[basis_name]
                basis_means.append(stats["mean"])
                basis_q25s.append(stats["q25"])
                basis_q75s.append(stats["q75"])

    # Add routers
    for layer_idx in router_layers:
        basis_name = f"layer_{layer_idx}_router"
        if basis_name in statistics:
            basis_types.append(f"L{layer_idx}_rtr")
            stats = statistics[basis_name]
            basis_means.append(stats["mean"])
            basis_q25s.append(stats["q25"])
            basis_q75s.append(stats["q75"])

    # Add aggregated router
    if "all_layers_router" in statistics:
        basis_types.append("All_rtr")
        stats = statistics["all_layers_router"]
        basis_means.append(stats["mean"])
        basis_q25s.append(stats["q25"])
        basis_q75s.append(stats["q75"])

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
    plt.savefig(os.path.join(KURTOSIS_DIR, f"{output_prefix}_comparison.png"), dpi=150)
    plt.close()

    logger.info(f"Saved visualizations to {KURTOSIS_DIR}")


if __name__ == "__main__":
    arguably.run()
