from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import gc
from itertools import batched, islice
import os
import queue
import sys
from typing import Any, TypeVar

import arguably
from loguru import logger
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import yaml

from core.device import (
    DeviceType,
    assert_device_type,
    get_backend,
    get_device,
    get_distributed_backend,
)
from core.moe import convert_router_logits_to_paths
from exp import OUTPUT_DIR
from exp.activations import Activations, load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from exp.kmeans_validation import (
    VALIDATION_SIZE_K_PROPORTION,
    WARNING_WINDOW_SIZE,
    check_monotonic_increasing_window,
    validate_centroid_distribution,
)
from exp.training import get_experiment_name

T = TypeVar("T")

GPU_QUEUE_MAXSIZE = 4


def check_worker_health(workers: dict[str, mp.Process], *, context: str = "") -> None:
    """Check if any workers have failed and raise appropriate exceptions."""
    for worker_name, worker in workers.items():
        if not worker.is_alive() and worker.exitcode != 0:
            context_str = f" [{context}]" if context else ""
            logger.critical(
                f"{worker_name} worker failed{context_str} with exit code {worker.exitcode}"
            )
            raise RuntimeError(
                f"{worker_name} worker failed with exit code {worker.exitcode}"
            )
        elif not worker.is_alive() and worker.exitcode == 0:
            context_str = f" [{context}]" if context else ""
            logger.critical(f"{worker_name} worker completed unexpectedly{context_str}")
            raise RuntimeError(f"{worker_name} worker completed unexpectedly")


@dataclass
class RunningKMeansData:
    # list of centroids of shape (K, D)
    centroid_sets: list[th.Tensor]
    # list of weights of shape (K) for online running updates
    weight_sets: list[th.Tensor]
    # losses of shape (num_K) for online running updates
    losses: th.Tensor

    def clear(self, clear_losses: bool = False) -> None:
        for centroids in self.centroid_sets:
            centroids.zero_()
        for weights in self.weight_sets:
            weights.zero_()

        if clear_losses:
            self.losses.zero_()

    def clone(self) -> RunningKMeansData:
        return RunningKMeansData(
            centroid_sets=[centroids.clone() for centroids in self.centroid_sets],
            weight_sets=[weights.clone() for weights in self.weight_sets],
            losses=self.losses.clone(),
        )

    def to(self, device: th.device) -> RunningKMeansData:
        return RunningKMeansData(
            centroid_sets=[centroids.to(device) for centroids in self.centroid_sets],
            weight_sets=[weights.to(device) for weights in self.weight_sets],
            losses=self.losses.to(device),
        )

    def copy_(self, other: RunningKMeansData) -> None:
        self.centroid_sets = [
            centroids.copy_(other_centroids)
            for centroids, other_centroids in zip(
                self.centroid_sets, other.centroid_sets, strict=True
            )
        ]
        self.weight_sets = [
            weights.copy_(other_weights)
            for weights, other_weights in zip(
                self.weight_sets, other.weight_sets, strict=True
            )
        ]
        self.losses.copy_(other.losses)

    def __add__(self, other: RunningKMeansData) -> RunningKMeansData:
        new_data = RunningKMeansData(
            centroid_sets=[
                th.empty_like(centroids) for centroids in self.centroid_sets
            ],
            weight_sets=[th.empty_like(weights) for weights in self.weight_sets],
            losses=th.empty_like(self.losses),
        )

        for losses_idx, (
            base_centroids,
            base_weights,
            other_centroids,
            other_weights,
            new_centroids,
            new_weights,
        ) in enumerate(
            zip(
                self.centroid_sets,
                self.weight_sets,
                other.centroid_sets,
                other.weight_sets,
                new_data.centroid_sets,
                new_data.weight_sets,
                strict=True,
            )
        ):
            logger.trace(
                f"Base weights {type(base_weights)} {base_weights.shape} {base_weights.dtype} {base_weights.device}"
            )
            logger.trace(
                f"Other weights {type(other_weights)} {other_weights.shape} {other_weights.dtype} {other_weights.device}"
            )
            logger.trace(
                f"New weights {type(new_weights)} {new_weights.shape} {new_weights.dtype} {new_weights.device}"
            )

            new_weights.copy_(base_weights + other_weights)

            # Avoid division by zero when weights are zero
            mask = new_weights > 0
            base_weight_proportion = th.ones_like(base_weights, dtype=th.float32)
            base_weight_proportion[mask] = base_weights[mask] / new_weights[mask]
            other_weight_proportion = 1 - base_weight_proportion

            # Debug: Check for problematic weight proportions
            if (
                th.isnan(base_weight_proportion).any()
                or th.isnan(other_weight_proportion).any()
            ):
                logger.warning(
                    f"NaN in weight proportions! base_weights: {base_weights}, other_weights: {other_weights}, new_weights: {new_weights}"
                )

            # Check if all weights are zero (which should NEVER happen)
            if new_weights.sum() == 0:
                logger.critical(
                    f"🚨 CRITICAL: new_weights.sum() == 0! This should NEVER happen.\n"
                    f"  base_weights sum: {base_weights.sum()}\n"
                    f"  other_weights sum: {other_weights.sum()}\n"
                    f"  new_weights sum: {new_weights.sum()}\n"
                    f"  Expected sum: minibatch_size or effective_batch_size\n"
                    f"  base_centroids device: {base_centroids.device}\n"
                    f"  other_centroids device: {other_centroids.device}\n"
                    f"  Has NaN in base_centroids: {th.isnan(base_centroids).any()}\n"
                    f"  Has NaN in other_centroids: {th.isnan(other_centroids).any()}\n"
                    f"  Has Inf in base_centroids: {th.isinf(base_centroids).any()}\n"
                    f"  Has Inf in other_centroids: {th.isinf(other_centroids).any()}"
                )
                raise RuntimeError(
                    "All weights are zero in __add__ - this indicates a serious bug in the k-means algorithm"
                )

            logger.trace(
                f"Base centroids {type(base_centroids)} {base_centroids.shape} {base_centroids.dtype} {base_centroids.device}"
            )
            logger.trace(
                f"Other centroids {type(other_centroids)} {other_centroids.shape} {other_centroids.dtype} {other_centroids.device}"
            )
            logger.trace(
                f"Base weight proportion {type(base_weight_proportion)} {base_weight_proportion.shape} {base_weight_proportion.dtype} {base_weight_proportion.device}"
            )
            logger.trace(
                f"Other weight proportion {type(other_weight_proportion)} {other_weight_proportion.shape} {other_weight_proportion.dtype} {other_weight_proportion.device}"
            )

            # Compute weighted average of centroids
            base_contribution = base_weight_proportion.unsqueeze(-1) * base_centroids
            other_contribution = other_weight_proportion.unsqueeze(-1) * other_centroids
            new_centroid_values = base_contribution + other_contribution

            new_centroids.copy_(new_centroid_values)

            base_weights_sum = base_weights.sum()
            new_weights_sum = new_weights.sum()
            if new_weights_sum == 0:
                new_data.losses[losses_idx] = self.losses[losses_idx]
                continue

            base_loss_proportion = base_weights_sum / new_weights_sum
            other_loss_proportion = 1 - base_loss_proportion
            new_data.losses[losses_idx] = (
                base_loss_proportion * self.losses[losses_idx]
                + other_loss_proportion * other.losses[losses_idx]
            )

        return new_data


@dataclass
class GPUData:
    synced_data: RunningKMeansData
    dirty_data: RunningKMeansData
    queue: mp.Queue | None = None

    def copy_(self, other: GPUData) -> None:
        self.synced_data.copy_(other.synced_data)
        self.dirty_data.copy_(other.dirty_data)

    def to(self, device: th.device) -> GPUData:
        return GPUData(
            synced_data=self.synced_data.to(device),
            dirty_data=self.dirty_data.to(device),
            queue=self.queue,
        )

    def reset_dirty_data(self) -> None:
        self.dirty_data.copy_(self.synced_data)
        for centroid_weights in self.dirty_data.weight_sets:
            centroid_weights.zero_()


def compute_all_centroids_from_assignments(
    data: th.Tensor,
    assignments: th.Tensor,
    num_centroids: int,
    assignment_minibatch_size: int = 0,
) -> tuple[th.Tensor, th.Tensor]:
    """
    Vectorized computation of all centroids using scatter_add_.

    Args:
        data: (B, D) tensor of data points
        assignments: (B,) tensor of centroid assignments
        num_centroids: Total number of centroids (K)
        assignment_minibatch_size: Size of data minibatches to process (0 = no batching, default: 0)

    Returns:
        new_centroids: (K, D) tensor of new centroid positions
        weights: (K,) tensor of number of points assigned to each centroid
    """
    assert assignment_minibatch_size >= 0, (
        f"assignment_minibatch_size must be non-negative, got {assignment_minibatch_size}"
    )

    batch_size, embed_dim = data.shape

    # Set assignment_minibatch_size to batch_size if 0 or larger than batch_size
    if assignment_minibatch_size == 0 or assignment_minibatch_size > batch_size:
        assignment_minibatch_size = batch_size

    # Initialize tensors for sums and counts
    centroid_sums = th.zeros(
        num_centroids, embed_dim, dtype=data.dtype, device=data.device
    )
    weights = th.zeros(num_centroids, dtype=th.int64, device=data.device)

    # Process data in minibatches using torch.split
    data_batches = th.split(data, assignment_minibatch_size, dim=0)
    assignments_batches = th.split(assignments, assignment_minibatch_size, dim=0)

    for data_batch, assignments_batch in zip(
        data_batches, assignments_batches, strict=False
    ):
        # Scatter add data points to their assigned centroids
        assignments_expanded = assignments_batch.unsqueeze(1).expand(-1, embed_dim)
        centroid_sums.scatter_add_(0, assignments_expanded, data_batch)

        # Count number of points per centroid
        weights.scatter_add_(0, assignments_batch, th.ones_like(assignments_batch))

    # Compute means with safe division (avoid div by zero for empty clusters)
    weights_expanded = weights.unsqueeze(1)
    weights_float = weights_expanded.to(dtype=data.dtype)
    new_centroids = th.where(
        weights_expanded > 0,
        centroid_sums / weights_float,
        th.zeros_like(centroid_sums),
    )

    # Assertions to validate correctness
    assert new_centroids.shape == (num_centroids, embed_dim), (
        f"Expected shape ({num_centroids}, {embed_dim}), got {new_centroids.shape}"
    )
    assert weights.shape == (num_centroids,), (
        f"Expected shape ({num_centroids},), got {weights.shape}"
    )
    assert (weights >= 0).all(), "Weights should be non-negative"
    assert weights.sum() == batch_size, (
        f"Total weights {weights.sum()} should not exceed batch_size {batch_size}"
    )

    # Check for NaN in results
    if th.isnan(new_centroids).any():
        nan_mask = th.isnan(new_centroids).any(dim=1)
        nan_indices = th.where(nan_mask)[0]
        logger.error(
            f"🚨 NaN detected in centroid computation for centroids: {nan_indices.tolist()[:10]}"
        )
        logger.error(
            f"data shape: {data.shape}, assignments shape: {assignments.shape}"
        )
        logger.error(f"weights: {weights[nan_indices[:5]]}")

    # Log empty clusters at TRACE level
    num_empty = (weights == 0).sum().item()
    if num_empty > 0:
        logger.trace(f"{num_empty} centroids have no assigned points")

    return new_centroids, weights


def validate_gpu_centroid_synchronization(
    all_gpu_data: list[GPUData],
    k_values: tuple[int, ...],
    context: str = "",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Validate that centroids are synchronized across all GPUs and all ranks.

    Args:
        all_gpu_data: List of GPU data containing centroids
        k_values: List of k values for different centroid sets
        context: Context string for logging (e.g., "after initialization")
        rtol: Relative tolerance for allclose comparison
        atol: Absolute tolerance for allclose comparison

    Returns:
        True if all centroids are synchronized, False otherwise
    """
    if len(all_gpu_data) <= 1 and dist.get_world_size() <= 1:
        logger.trace(
            f"Only {len(all_gpu_data)} GPU(s) and {dist.get_world_size()} rank(s), skipping synchronization validation"
        )
        return True

    context_str = f" {context}" if context else ""
    logger.debug(
        f"🔍 SYNC VALIDATION{context_str}: Checking centroid synchronization across {len(all_gpu_data)} GPUs and {dist.get_world_size()} ranks"
    )

    all_synchronized = True

    # First check within-rank GPU synchronization (existing logic)
    if len(all_gpu_data) > 1:
        for k_idx, k in enumerate(k_values):
            # Get reference centroids from GPU 0
            ref_centroids = all_gpu_data[0].synced_data.centroid_sets[k_idx].cpu()

            for gpu_idx in range(1, len(all_gpu_data)):
                gpu_centroids = (
                    all_gpu_data[gpu_idx].synced_data.centroid_sets[k_idx].cpu()
                )

                # Check if centroids are synchronized
                if not th.allclose(ref_centroids, gpu_centroids, rtol=rtol, atol=atol):
                    all_synchronized = False

                    # Calculate differences for detailed logging
                    diff = th.abs(ref_centroids - gpu_centroids)
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()

                    # Check norms
                    ref_norms = th.norm(ref_centroids, dim=1)
                    gpu_norms = th.norm(gpu_centroids, dim=1)
                    norm_diff = th.abs(ref_norms - gpu_norms)
                    max_norm_diff = norm_diff.max().item()

                    logger.error(
                        f"🚨 SYNC MISMATCH{context_str} k_idx={k_idx} (k={k}): "
                        f"GPU 0 vs GPU {gpu_idx} - "
                        f"Max diff: {max_diff:.8f}, Mean diff: {mean_diff:.8f}, "
                        f"Max norm diff: {max_norm_diff:.8f}"
                    )

                    # Log centroid statistics
                    logger.error(
                        f"GPU 0 centroids - Norms: min={ref_norms.min():.6f}, max={ref_norms.max():.6f}, mean={ref_norms.mean():.6f}"
                    )
                    logger.error(
                        f"GPU {gpu_idx} centroids - Norms: min={gpu_norms.min():.6f}, max={gpu_norms.max():.6f}, mean={gpu_norms.mean():.6f}"
                    )

                    # Count zero-norm centroids
                    ref_zeros = (ref_norms == 0).sum().item()
                    gpu_zeros = (gpu_norms == 0).sum().item()
                    logger.error(
                        f"Zero-norm centroids - GPU 0: {ref_zeros}/{len(ref_norms)}, GPU {gpu_idx}: {gpu_zeros}/{len(gpu_norms)}"
                    )
                else:
                    logger.trace(
                        f"✅ SYNC OK{context_str} k_idx={k_idx} (k={k}): GPU 0 and GPU {gpu_idx} centroids match"
                    )

    # Now check across-rank synchronization using allgather
    if dist.get_world_size() > 1:
        world_size = dist.get_world_size()

        for k_idx, k in enumerate(k_values):
            # Get centroids from GPU 0 on this rank and move to CPU
            local_centroids = all_gpu_data[0].synced_data.centroid_sets[k_idx].cpu()

            # Prepare list to gather centroids from all ranks
            gathered_centroids = [
                th.zeros_like(local_centroids) for _ in range(world_size)
            ]

            # Allgather centroids from all ranks
            dist.all_gather(gathered_centroids, local_centroids)

            # Compare centroids across ranks (use rank 0 as reference)
            ref_centroids = gathered_centroids[0]

            for rank_idx in range(1, world_size):
                rank_centroids = gathered_centroids[rank_idx]

                if not th.allclose(ref_centroids, rank_centroids, rtol=rtol, atol=atol):
                    all_synchronized = False

                    # Calculate differences for detailed logging
                    diff = th.abs(ref_centroids - rank_centroids)
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()

                    # Check norms
                    ref_norms = th.norm(ref_centroids, dim=1)
                    rank_norms = th.norm(rank_centroids, dim=1)
                    norm_diff = th.abs(ref_norms - rank_norms)
                    max_norm_diff = norm_diff.max().item()

                    logger.error(
                        f"🚨 RANK SYNC MISMATCH{context_str} k_idx={k_idx} (k={k}): "
                        f"Rank 0 vs Rank {rank_idx} - "
                        f"Max diff: {max_diff:.8f}, Mean diff: {mean_diff:.8f}, "
                        f"Max norm diff: {max_norm_diff:.8f}"
                    )

                    # Log centroid statistics
                    logger.error(
                        f"Rank 0 centroids - Norms: min={ref_norms.min():.6f}, max={ref_norms.max():.6f}, mean={ref_norms.mean():.6f}"
                    )
                    logger.error(
                        f"Rank {rank_idx} centroids - Norms: min={rank_norms.min():.6f}, max={rank_norms.max():.6f}, mean={rank_norms.mean():.6f}"
                    )

                    # Count zero-norm centroids
                    ref_zeros = (ref_norms == 0).sum().item()
                    rank_zeros = (rank_norms == 0).sum().item()
                    logger.error(
                        f"Zero-norm centroids - Rank 0: {ref_zeros}/{len(ref_norms)}, Rank {rank_idx}: {rank_zeros}/{len(rank_norms)}"
                    )
                else:
                    logger.trace(
                        f"✅ RANK SYNC OK{context_str} k_idx={k_idx} (k={k}): Rank 0 and Rank {rank_idx} centroids match"
                    )

    if all_synchronized:
        logger.debug(
            f"✅ SYNC VALIDATION{context_str}: All centroids synchronized across GPUs and ranks"
        )
    else:
        logger.error(
            f"🚨 SYNC VALIDATION{context_str}: Centroid synchronization FAILED!"
        )

    return all_synchronized


def kmeans_step(
    data: th.Tensor,  # (B, L * E)
    centroids: th.Tensor,  # (K, L * E)
    centroid_minibatch_size: int = 65536,
    assignment_minibatch_size: int = 4096,
    gpu_idx: int | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    logger.trace(
        f"Running kmeans step with {data.shape[0]} data points and {centroids.shape[0]} centroids"
    )
    logger.trace(f"Data: {data.dtype} {data.device} {data.shape}")
    logger.trace(f"Centroids: {centroids.dtype} {centroids.device} {centroids.shape}")

    # Validate centroids before distance computation
    if th.isnan(centroids).any():
        num_nan = th.isnan(centroids).sum().item()
        logger.error(
            f"[GPU {gpu_idx}] 🚨 NaN in centroids! {num_nan}/{centroids.numel()} values are NaN\n"
            f"  centroids shape: {centroids.shape}\n"
            f"  data shape: {data.shape}\n"
            f"  centroid_minibatch_size: {centroid_minibatch_size}"
        )
        nan_mask = th.isnan(centroids).any(dim=1)
        nan_indices = th.where(nan_mask)[0]
        logger.error(f"Centroids with NaN: {nan_indices.tolist()[:10]}")
        raise RuntimeError("NaN detected in centroids")

    if th.isinf(centroids).any():
        num_inf = th.isinf(centroids).sum().item()
        logger.error(
            f"[GPU {gpu_idx}] 🚨 Inf in centroids! {num_inf}/{centroids.numel()} values are Inf\n"
            f"  centroids shape: {centroids.shape}\n"
            f"  data shape: {data.shape}\n"
            f"  centroid_minibatch_size: {centroid_minibatch_size}"
        )
        inf_mask = th.isinf(centroids).any(dim=1)
        inf_indices = th.where(inf_mask)[0]
        logger.error(f"Centroids with Inf: {inf_indices.tolist()[:10]}")
        raise RuntimeError("Inf detected in centroids")

    # (B, K) - Compute distances with centroid batching to avoid CUDA limits
    data_float = data.to(th.float32)
    centroids_float = centroids.to(th.float32)

    if centroids_float.shape[0] <= centroid_minibatch_size:
        # Small enough to compute in one go
        distances = th.cdist(data_float, centroids_float, p=1)
    else:
        # Chunk centroids to avoid CUDA configuration limits
        n_centroids = centroids_float.shape[0]
        n_chunks = (
            n_centroids + centroid_minibatch_size - 1
        ) // centroid_minibatch_size

        # Split centroids into chunks
        centroid_chunks = th.tensor_split(centroids_float, n_chunks, dim=0)

        # Compute distances for each chunk
        chunk_distances = [
            th.cdist(data_float, chunk, p=1) for chunk in centroid_chunks
        ]

        # Concatenate along centroid dimension
        distances = th.cat(chunk_distances, dim=1)

    logger.trace(f"Computed distances with shape {distances.shape}")

    # Validate distances before argmin
    if th.isnan(distances).any():
        num_nan = th.isnan(distances).sum().item()
        logger.error(
            f"[GPU {gpu_idx}] 🚨 NaN in distances! {num_nan}/{distances.numel()} values"
        )
        raise RuntimeError("NaN detected in distance computation")

    if th.isinf(distances).any():
        num_inf = th.isinf(distances).sum().item()
        logger.error(
            f"[GPU {gpu_idx}] 🚨 Inf in distances! {num_inf}/{distances.numel()} values"
        )
        raise RuntimeError("Inf detected in distance computation")
    # (B)
    assignments = th.argmin(distances, dim=1)
    logger.trace(f"Computed assignments with shape {assignments.shape}")

    # for calculating loss, we get the distances from each data point to the closest centroid
    centroid_distances = th.gather(distances, 1, assignments.unsqueeze(1))
    logger.trace(f"Computed centroid distances with shape {centroid_distances.shape}")

    new_centroids, new_weights = compute_all_centroids_from_assignments(
        data=data,
        assignments=assignments,
        num_centroids=centroids.shape[0],
        assignment_minibatch_size=assignment_minibatch_size,
    )
    logger.trace(f"Computed centroids and weights with shape {new_centroids.shape}")

    new_loss = centroid_distances.mean()
    logger.trace(f"Computed new loss with shape {new_loss.shape}")

    # Log update statistics
    # Validate new centroids
    if th.isnan(new_centroids).any():
        num_nan = th.isnan(new_centroids).sum().item()
        logger.error(
            f"[GPU {gpu_idx}] 🚨 NaN in new_centroids! {num_nan}/{new_centroids.numel()} values"
        )
        nan_mask = th.isnan(new_centroids).any(dim=1)
        nan_indices = th.where(nan_mask)[0]
        for idx in nan_indices[:5]:
            logger.error(f"  Centroid {idx}: weight={new_weights[idx]}")
        raise RuntimeError("NaN in new centroids after update")

    update_norms = th.norm(new_centroids, dim=1)
    zero_update_norms = (update_norms == 0).sum().item()
    total_weight = new_weights.sum().item()
    logger.debug(
        f"[GPU {gpu_idx}] 📊 UPDATE: Computed updates - zero_norms={zero_update_norms}/{len(update_norms)}, norm_stats: min={update_norms.min():.6f}, max={update_norms.max():.6f}, mean={update_norms.mean():.6f}, total_weight={total_weight}"
    )

    logger.trace(
        f"New centroids: {new_centroids.dtype} {new_centroids.device} {new_centroids.shape}"
    )
    logger.trace(
        f"New weights: {new_weights.dtype} {new_weights.device} {new_weights.shape}"
    )
    logger.trace(f"New loss: {new_loss.dtype} {new_loss.device} {new_loss.shape}")

    return new_centroids, new_weights, new_loss


def sync(
    gpu_idx: int,
    all_gpu_data: list[GPUData],
    gpu_data: GPUData,
    losses_over_time: list[th.Tensor],
    barrier,
    group: dist.ProcessGroup | None = None,
    device_type: DeviceType = "cuda",
) -> None:
    backend = get_backend(device_type)
    device = get_device(device_type, gpu_idx)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Use gpu_specific_group for synchronization

    # Clear cache at start to prevent memory fragmentation
    backend.empty_cache()
    logger.debug(f"🔄 SYNC: Cleared GPU cache for GPU {gpu_idx}")

    logger.debug(f"🔄 SYNC: Starting sync for GPU {gpu_idx}, rank {rank}")

    # Log dirty data state before sync
    for k_idx, (centroids, weights) in enumerate(
        zip(
            gpu_data.dirty_data.centroid_sets,
            gpu_data.dirty_data.weight_sets,
            strict=True,
        )
    ):
        centroid_norms = th.norm(centroids, dim=1)
        zero_norms = (centroid_norms == 0).sum().item()
        weight_sum = weights.sum().item()
        logger.debug(
            f"🔄 SYNC GPU {gpu_idx} k_idx={k_idx} BEFORE: Dirty centroids zero_norms={zero_norms}/{len(centroid_norms)}, norm_stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}, weight_sum={weight_sum:.2f}"
        )

    # gather across nodes
    # (N, num_K)
    all_losses = (
        th.empty_like(gpu_data.dirty_data.losses).unsqueeze(0).repeat(world_size, 1)
    )
    dist.all_gather_into_tensor(all_losses, gpu_data.dirty_data.losses, group=group)

    logger.trace(
        f"All losses: {all_losses.shape} {all_losses.dtype} {all_losses.device} {all_losses}"
    )
    logger.trace(
        f"All losses stats: "
        f"Min: {all_losses.min()}, "
        f"Max: {all_losses.max()}, "
        f"Mean: {all_losses.mean()}, "
        f"Std: {all_losses.std()}"
    )

    for losses_idx, (
        centroids,
        weights,
    ) in enumerate(
        zip(
            gpu_data.dirty_data.centroid_sets,
            gpu_data.dirty_data.weight_sets,
            strict=True,
        )
    ):
        # (N, K, D)
        all_centroids = th.empty_like(centroids).unsqueeze(0).repeat(world_size, 1, 1)
        # (N, K)
        all_weights = th.empty_like(weights).unsqueeze(0).repeat(world_size, 1)

        dist.all_gather_into_tensor(all_centroids, centroids, group=group)
        dist.all_gather_into_tensor(all_weights, weights, group=group)

        # (K)
        weights_total = all_weights.sum(dim=0)

        # make sure that any centroids with zero weights are all equal
        empty_centroids_mask = weights_total == 0
        num_empty_centroids = empty_centroids_mask.sum().item()
        if num_empty_centroids > 0:
            logger.debug(f"Found {num_empty_centroids} centroids with zero weights")

            empty_centroids = all_centroids[:, empty_centroids_mask]
            empty_centroids_rolled = empty_centroids.roll(1, dims=0)
            assert th.allclose(empty_centroids, empty_centroids_rolled), (
                f"Empty centroids are not equal: {empty_centroids} != {empty_centroids_rolled}"
            )

            # make sure the centroids themselves are not zero
            empty_centroids_norms = th.norm(empty_centroids, dim=-1)
            num_zero_norms = (empty_centroids_norms < 1e-6).sum().item()

            assert num_zero_norms == 0, (
                f"Found {num_zero_norms} empty centroids with zero norm out of {num_empty_centroids} empty centroids"
            )

        # (N, K) - handle division by zero for empty centroids
        weights_proportion = th.where(
            weights_total.unsqueeze(0) > 0,
            all_weights / weights_total.unsqueeze(0),
            th.full_like(all_weights, fill_value=1 / world_size),
        )

        if th.isnan(weights_proportion).any():
            logger.error(
                f"NaN values found in weights_proportion! weights_total: {weights_total}"
            )
            logger.error(f"all_weights: {all_weights}")

        # (K, D)
        new_centroids = (all_centroids * weights_proportion.unsqueeze(-1)).sum(dim=0)

        nan_centroids = th.isnan(new_centroids).any(dim=1).sum().item()
        if nan_centroids > 0:
            logger.error(
                f"Found {nan_centroids} centroids with NaN values after aggregation!"
            )
            logger.error(f"Centroid norms: {th.norm(new_centroids, dim=1)}")
            logger.error(f"weights_total: {weights_total}")
            logger.error(f"weights_proportion: {weights_proportion}")

        centroid_norms = th.norm(new_centroids, dim=1)
        zero_norm_centroids = (centroid_norms == 0).sum().item()
        if zero_norm_centroids > 0:
            logger.debug(f"Found {zero_norm_centroids} centroids with zero norm")
            logger.debug(
                f"Centroid norm stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}"
            )

        if not th.isfinite(new_centroids).all():
            logger.error("Non-finite values detected in centroids!")
            inf_centroids = th.isinf(new_centroids).any(dim=1).sum().item()
            logger.error(f"Centroids with inf values: {inf_centroids}")
            logger.error(f"Centroids with nan values: {nan_centroids}")

        # (N)
        dist_losses = all_losses[:, losses_idx]
        # ()
        new_loss = dist_losses.mean()

        gpu_data.dirty_data.centroid_sets[losses_idx] = new_centroids
        gpu_data.dirty_data.weight_sets[losses_idx] = weights_total
        gpu_data.dirty_data.losses[losses_idx] = new_loss

        # Log new centroids after aggregation
        new_centroid_norms = th.norm(new_centroids, dim=1)
        new_zero_norms = (new_centroid_norms == 0).sum().item()
        logger.debug(
            f"🔄 SYNC GPU {gpu_idx} k_idx={losses_idx} AFTER AGGREGATION: New centroids zero_norms={new_zero_norms}/{len(new_centroid_norms)}, norm_stats: min={new_centroid_norms.min():.6f}, max={new_centroid_norms.max():.6f}, mean={new_centroid_norms.mean():.6f}, weights_total_sum={weights_total.sum().item():.2f}"
        )

        del (
            all_centroids,
            all_weights,
            weights_total,
            weights_proportion,
            new_centroids,
            new_loss,
        )

        # Clear cache after each centroid set to prevent memory fragmentation
        backend.empty_cache()

    # now do an all-gather along gpus (among entries in all_gpu_data)
    # note thtat shared memory is only on cpu, so all_gpu_data is actually on cpu
    empty_data = RunningKMeansData(
        centroid_sets=[
            th.zeros_like(centroids) for centroids in gpu_data.synced_data.centroid_sets
        ],
        weight_sets=[
            th.zeros_like(weights) for weights in gpu_data.synced_data.weight_sets
        ],
        losses=th.zeros_like(gpu_data.synced_data.losses),
    )

    shared_gpu_data = all_gpu_data[gpu_idx]
    shared_gpu_data.copy_(gpu_data)

    # Debug: Log synced data before update
    for k_idx, centroids in enumerate(gpu_data.synced_data.centroid_sets):
        centroid_norms = th.norm(centroids, dim=1)
        zero_norms = (centroid_norms == 0).sum().item()
        logger.debug(
            f"🔄 SYNC GPU {gpu_idx} k_idx={k_idx} BEFORE GPU AGGREGATION: Synced centroids zero_norms={zero_norms}/{len(centroid_norms)}, norm_stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}"
        )

    barrier.wait()

    gpu_data.synced_data += sum(
        (current_gpu_data.dirty_data.to(device) for current_gpu_data in all_gpu_data),
        start=empty_data,
    )

    backend.empty_cache()

    # Wait for all GPUs to finish reading dirty data before resetting
    barrier.wait()

    # reset dirty data now that it has been synced
    logger.debug(f"🔄 SYNC GPU {gpu_idx}: Resetting dirty data...")
    gpu_data.reset_dirty_data()

    # Log synced data state after sync
    for k_idx, centroids in enumerate(gpu_data.synced_data.centroid_sets):
        centroid_norms = th.norm(centroids, dim=1)
        zero_norms = (centroid_norms == 0).sum().item()
        # Use trace level if there are no zero norms (normal case), debug if there are issues
        log_level = logger.debug if zero_norms > 0 else logger.trace
        log_level(
            f"🔄 SYNC GPU {gpu_idx} k_idx={k_idx} FINAL: Synced centroids zero_norms={zero_norms}/{len(centroid_norms)}, norm_stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}"
        )

    if rank == 0 and gpu_idx == 0:
        logger.trace(f"GPU {gpu_idx}: Appending losses to history")
        losses_over_time.append(gpu_data.synced_data.losses.detach().cpu().clone())

        # Check for monotonically increasing loss windows
        if (
            len(losses_over_time) >= WARNING_WINDOW_SIZE
        ):  # Need at least window_size iterations
            logger.trace(
                f"GPU {gpu_idx}: Checking for monotonic increasing loss window"
            )
            losses_tensor = th.stack(losses_over_time, dim=1)
            has_problem, start_idx = check_monotonic_increasing_window(
                losses_tensor, window_size=WARNING_WINDOW_SIZE
            )
            if has_problem:
                logger.warning(
                    f"Detected monotonically increasing loss window starting at "
                    f"iteration {start_idx}. This may indicate a training problem."
                )

    logger.trace(f"✅ SYNC GPU {gpu_idx}: Sync operation fully completed")


def save_checkpoint(
    save_dir: str,
    iteration: int,
    gpu_data: GPUData,
    losses_over_time: list[th.Tensor],
    top_k: int,
    tokens_seen: int = 0,
    hyperparams: dict[str, Any] | None = None,
) -> None:
    """
    Save a checkpoint with comprehensive metadata.
    Args:
        save_dir: Directory to save the checkpoint
        iteration: Current iteration number
        gpu_data: GPU data containing synced centroids
        losses_over_time: List of loss tensors over time
        top_k: Top-k value for the model
        tokens_seen: Total number of tokens processed so far
        hyperparams: Optional dictionary of hyperparameters
    """
    # Collect centroid statistics
    centroid_stats = []
    for centroids in gpu_data.synced_data.centroid_sets:
        centroid_norms = th.norm(centroids, dim=1)
        stats = {
            "k_value": centroids.shape[0],
            "num_centroids": centroids.shape[0],
            "embedding_dim": centroids.shape[1],
            "zero_norm_count": (centroid_norms == 0).sum().item(),
            "min_norm": centroid_norms.min().item(),
            "max_norm": centroid_norms.max().item(),
            "mean_norm": centroid_norms.mean().item(),
            "std_norm": centroid_norms.std().item(),
        }
        centroid_stats.append(stats)
    # Prepare loss statistics
    if losses_over_time:
        losses_tensor = th.stack(losses_over_time, dim=1)
        loss_stats = {
            "num_iterations": losses_tensor.shape[1],
            "current_losses": losses_tensor[:, -1].tolist(),
            "mean_losses": losses_tensor.mean(dim=1).tolist(),
            "min_losses": losses_tensor.min(dim=1).values.tolist(),
            "max_losses": losses_tensor.max(dim=1).values.tolist(),
        }
    else:
        loss_stats = {"num_iterations": 0}
    # Create checkpoint data
    checkpoint_data = {
        # Model state
        "centroids": [c.cpu().clone() for c in gpu_data.synced_data.centroid_sets],
        "weights": [w.cpu().clone() for w in gpu_data.synced_data.weight_sets],
        # Training state
        "iteration": iteration,
        "tokens_seen": tokens_seen,
        "losses": th.stack(losses_over_time, dim=1).cpu()
        if losses_over_time
        else th.empty(0),
        # Model config
        "top_k": top_k,
        # Metadata
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "tokens_seen": tokens_seen,
            "centroid_stats": centroid_stats,
            "loss_stats": loss_stats,
            "hyperparams": hyperparams or {},
        },
    }
    # Save numbered checkpoint
    checkpoint_path = os.path.join(
        save_dir, CHECKPOINT_FILENAME.format(iteration=iteration)
    )
    th.save(checkpoint_data, checkpoint_path)
    logger.info(
        f"Saved checkpoint at iteration {iteration} ({tokens_seen:,} tokens) to {checkpoint_path}"
    )
    # Also save as latest checkpoint for easy resumption
    latest_checkpoint_path = os.path.join(save_dir, LATEST_CHECKPOINT_FILENAME)
    th.save(checkpoint_data, latest_checkpoint_path)
    logger.debug(f"Updated latest checkpoint: {latest_checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    all_gpu_data: list[GPUData],
    device_type: DeviceType = "cuda",  # noqa: ARG001
) -> tuple[int, int, list[th.Tensor], int]:
    """
    Load a checkpoint and restore training state.
    Args:
        checkpoint_path: Path to the checkpoint file
        all_gpu_data: List of GPU data to populate with loaded centroids
        device_type: Device type ("cuda" or "xpu")
    Returns:
        Tuple of (start_iteration, tokens_seen, losses_over_time, top_k)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = th.load(checkpoint_path, map_location="cpu")
    # Restore centroids and weights to all GPUs
    for gpu_data in all_gpu_data:
        for k_idx, (loaded_centroids, loaded_weights) in enumerate(
            zip(checkpoint_data["centroids"], checkpoint_data["weights"], strict=True)
        ):
            gpu_data.synced_data.centroid_sets[k_idx].copy_(loaded_centroids)
            gpu_data.synced_data.weight_sets[k_idx].copy_(loaded_weights)
            gpu_data.dirty_data.centroid_sets[k_idx].copy_(loaded_centroids)
            gpu_data.dirty_data.weight_sets[k_idx].zero_()
    # Restore training state
    start_iteration = checkpoint_data["iteration"] + 1
    tokens_seen = checkpoint_data.get("tokens_seen", 0)
    top_k = checkpoint_data["top_k"]
    # Restore loss history
    losses_tensor = checkpoint_data["losses"]
    losses_over_time = (
        [losses_tensor[:, i] for i in range(losses_tensor.shape[1])]
        if losses_tensor.numel() > 0
        else []
    )
    # Log checkpoint metadata if available
    if "metadata" in checkpoint_data:
        metadata = checkpoint_data["metadata"]
        logger.info("Checkpoint metadata:")
        logger.info(f"  Saved at: {metadata.get('timestamp', 'unknown')}")
        logger.info(f"  Iteration: {metadata.get('iteration', 'unknown')}")
        logger.info(f"  Tokens seen: {metadata.get('tokens_seen', 'unknown'):,}")
        if "centroid_stats" in metadata:
            logger.info("  Centroid statistics:")
            for stats in metadata["centroid_stats"]:
                logger.info(
                    f"    k={stats['k_value']}: "
                    f"zero_norms={stats['zero_norm_count']}/{stats['num_centroids']}, "
                    f"norm_stats: min={stats['min_norm']:.6f}, "
                    f"max={stats['max_norm']:.6f}, mean={stats['mean_norm']:.6f}"
                )
    logger.info(
        f"Resumed from iteration {checkpoint_data['iteration']}, "
        f"starting at iteration {start_iteration} with {tokens_seen:,} tokens seen"
    )
    return start_iteration, tokens_seen, losses_over_time, top_k


def gpu_worker(
    gpu_idx: int,
    all_gpu_data: list[GPUData],
    top_k: int,
    losses_over_time: list[th.Tensor],
    barrier,
    rank: int,
    world_size: int,
    save_dir: str | None = None,
    validate_every: int = 64,
    centroid_minibatch_size: int = 65536,
    assignment_minibatch_size: int = 4096,
    device_type: DeviceType = "cuda",
) -> None:
    """
    GPU worker for distributed k-means clustering.

    Args:
        gpu_idx: Index of the device this worker is responsible for
        all_gpu_data: List of GPU data objects for all devices
        top_k: Number of top experts to consider
        losses_over_time: Shared list to store losses over time
        barrier: Synchronization barrier for coordinating workers
        rank: Rank of this process in the distributed group (must be passed as argument)
        world_size: Number of distributed nodes (must be passed as argument for serialization)
        save_dir: Directory to save checkments (if any)
        validate_every: Validate centroid synchronization every N sync operations (default: 1)
        centroid_minibatch_size: Size of centroid chunks to avoid device limits (default: 65536)
        device_type: Device type ("cuda" or "xpu", defaults to "cuda")
    """
    device = get_device(device_type, gpu_idx)
    backend = get_backend(device_type)
    backend_name = get_distributed_backend(device_type)

    logger.info(f"Starting GPU worker {gpu_idx}")

    # Set unique port for this GPU worker to avoid conflicts
    # MUST be done BEFORE init_process_group() to prevent port conflicts
    base_port = int(os.environ.get("MASTER_PORT", "29500"))
    worker_port = base_port + gpu_idx + 1
    os.environ["MASTER_PORT"] = str(worker_port)

    # Initialize distributed process group in this worker process
    # Each spawned process needs its own init_process_group call
    assert not dist.is_initialized(), (
        "Distributed should not be initialized in worker process"
    )
    dist.init_process_group(
        backend=backend_name,
        rank=rank,
        world_size=world_size,
    )
    logger.debug(
        f"GPU worker {gpu_idx} initialized process group (rank={rank}, world_size={world_size})"
    )

    shared_gpu_data = all_gpu_data[gpu_idx]
    local_gpu_data = shared_gpu_data.to(device)

    # Create GPU-specific group for this worker (always, regardless of world_size)
    gpu_specific_group = dist.new_group(
        ranks=list(range(world_size)), backend=backend_name
    )
    logger.debug(
        f"GPU worker {gpu_idx} created {backend_name} process group on port {worker_port}"
    )

    sync_iteration = 0

    while True:
        logger.trace(f"GPU {gpu_idx} waiting for queue item...")
        try:
            queue_item = shared_gpu_data.queue.get(timeout=60.0)
            logger.trace(f"GPU {gpu_idx} picked up item from queue")
        except queue.Empty:
            logger.warning(
                f"GPU {gpu_idx} timed out waiting for queue item - may indicate main loop hang"
            )
            continue

        if queue_item is None:
            logger.trace(f"GPU {gpu_idx} received stop signal")
            break

        router_logits, should_sync, save_idx = queue_item

        logger.trace(
            f"GPU {gpu_idx} received queue item (should_sync={should_sync}, save_idx={save_idx})"
        )

        # assert that if save_idx is not None, then should_sync is also true
        if save_idx is not None:
            assert should_sync, "save_idx can only be set when should_sync is true"

        logger.trace(f"GPU {gpu_idx} converting router logits to paths")

        # (B, L, E)
        device = get_device(device_type, gpu_idx)
        router_logits = router_logits.to(device)

        # convert from logits to paths
        paths_sparse = th.topk(router_logits, k=top_k, dim=-1).indices
        router_paths = th.zeros_like(router_logits)
        router_paths.scatter_(-1, paths_sparse, 1)
        del router_logits, paths_sparse

        logger.trace(f"GPU {gpu_idx} flattened router paths")

        # (B, L, E) -> (B, L * E)
        flat_data = router_paths.view(router_paths.shape[0], -1)

        del router_paths

        logger.trace(f"GPU {gpu_idx} emptied cache")

        backend.empty_cache()

        logger.trace(
            f"GPU {gpu_idx} running kmeans step with {len(shared_gpu_data.synced_data.centroid_sets)} centroids"
        )

        updates = [
            kmeans_step(
                data=flat_data,
                centroids=centroids,
                centroid_minibatch_size=centroid_minibatch_size,
                assignment_minibatch_size=assignment_minibatch_size,
                gpu_idx=gpu_idx,
            )
            for centroids in local_gpu_data.synced_data.centroid_sets
        ]
        new_centroid_sets, new_weight_sets, new_losses = zip(*updates, strict=True)

        logger.trace(f"GPU {gpu_idx} updated dirty data")
        logger.trace(
            f"New centroid sets: {len(new_centroid_sets)} {type(new_centroid_sets)} {new_centroid_sets[0].shape} {new_centroid_sets[0].dtype} {new_centroid_sets[0].device}"
        )
        logger.trace(
            f"New weight sets: {len(new_weight_sets)} {type(new_weight_sets)} {new_weight_sets[0].shape} {new_weight_sets[0].dtype} {new_weight_sets[0].device}"
        )
        logger.trace(
            f"New losses: {len(new_losses)} {type(new_losses)} {new_losses[0].shape} {new_losses[0].dtype} {new_losses[0].device}"
        )

        local_gpu_data.dirty_data += RunningKMeansData(
            centroid_sets=list(new_centroid_sets),
            weight_sets=list(new_weight_sets),
            losses=th.stack(new_losses),
        )

        logger.trace(f"GPU {gpu_idx} updated synced data")

        if not should_sync:
            logger.trace(f"GPU {gpu_idx} skipping sync, continuing to next item")
            continue

        sync(
            gpu_idx,
            all_gpu_data,
            local_gpu_data,
            losses_over_time,
            barrier,
            gpu_specific_group,
            device_type,
        )

        logger.trace(
            f"✅ GPU {gpu_idx} completed sync operation, proceeding to validation"
        )

        # Increment sync iteration counter
        sync_iteration += 1
        logger.trace(f"GPU {gpu_idx} sync_iteration now at {sync_iteration}")

        # Validate GPU synchronization after sync (only on GPU 0 to avoid redundant checks)
        # Only validate every validate_every iterations
        if (
            gpu_idx == 0
            and (len(all_gpu_data) > 1 or dist.get_world_size() > 1)
            and sync_iteration % validate_every == 0
        ):
            logger.trace(
                f"GPU {gpu_idx}: Starting validation at sync_iteration {sync_iteration}"
            )
            # We need k_values, but it's not available in this scope
            # For now, we'll infer it from the number of centroid sets
            k_values = tuple(
                centroid_set.shape[0]
                for centroid_set in all_gpu_data[0].synced_data.centroid_sets
            )
            sync_ok = validate_gpu_centroid_synchronization(
                all_gpu_data,
                k_values,
                context=f"after sync on GPU {gpu_idx} (iteration {sync_iteration})",
            )
            if not sync_ok:
                raise RuntimeError(
                    f"GPU centroid synchronization failed after sync on GPU {gpu_idx} at iteration {sync_iteration}. "
                    f"Check logs for detailed mismatch information."
                )
            logger.trace(f"GPU {gpu_idx}: Validation passed")

        # save checkpoint if save_idx is not None and we're on rank 0 gpu 0
        if (
            save_idx is not None
            and dist.get_rank() == 0
            and gpu_idx == 0
            and save_dir is not None
        ):
            logger.trace(f"GPU {gpu_idx} saving checkpoint")
            save_checkpoint(
                save_dir=save_dir,
                iteration=save_idx,
                gpu_data=all_gpu_data[gpu_idx],
                losses_over_time=losses_over_time,
                top_k=top_k,
            )

        logger.trace(f"🔄 GPU {gpu_idx} finished iteration, ready for next queue item")


def get_top_circuits(
    centroids: th.Tensor, num_layers: int, top_k: int
) -> tuple[th.Tensor, th.Tensor]:
    num_centroids = centroids.shape[0]
    circuit_centroids = centroids.view(num_centroids, num_layers, -1)

    circuits = th.topk(circuit_centroids, k=top_k, dim=2)
    circuit_mask = th.zeros_like(circuit_centroids)
    circuit_mask.scatter_(2, circuits.indices, 1)

    return circuits.indices, circuit_mask


KMEANS_TYPE = "kmeans"
METADATA_FILENAME = "metadata.yaml"
KMEANS_FILENAME = "kmeans.pt"
CHECKPOINT_FILENAME = "checkpoint_iter_{iteration}.pt"
LATEST_CHECKPOINT_FILENAME = "checkpoint_latest.pt"


def should_save_checkpoint(step: int, save_every: int) -> bool:
    """
    Determine if we should save a checkpoint at the given step.

    Args:
        step: Current step/batch number (0-indexed)
        save_every: Save frequency (every N steps after warmup)

    Returns:
        True if we should save at this step

    Logic:
        - For step 0: Always save (initial checkpoint)
        - For 0 < step < save_every: Save if step is a power of 2 (exponential warmup)
        - For step >= save_every: Save if step % save_every == 0 (linear schedule)
    """
    if step == 0:
        return True
    elif 0 < step < save_every:
        # Check if step is a power of 2
        return (step & (step - 1)) == 0
    else:
        # Linear schedule: save every save_every steps
        return step % save_every == 0


def kmeans_manhattan(
    activations: Activations,
    activation_dim: int,
    k_values: tuple[int, ...],
    effective_batch_size: int | None = None,
    max_iters: int = 16,
    minibatch_size: int | None = None,
    centroid_minibatch_size: int = 32768,
    assignment_minibatch_size: int = 4096,
    seed: int = 0,
    save_every: int | None = None,
    save_dir: str | None = None,
    validate_every: int = 64,
    log_level_numeric: int | None = None,
    device_type: DeviceType = "cuda",
) -> tuple[list[th.Tensor], int, th.Tensor, int, int]:
    """
    Perform k-means clustering with Manhattan distance.

    Args:
        activations: Activations to cluster
        k_values: List of number of clusters
        effective_batch_size: Batch size for k-means updates. If None, use the batch size of the activations.
        max_iters: Maximum number of iterations
        minibatch_size: Batch size for processing data. If None, process all data at once.
        centroid_minibatch_size: Size of centroid chunks to avoid device limits (defaults to 32768)
        assignment_minibatch_size: Size of assignment data minibatches (default: 4096)
        seed: Random seed for initialization
        save_every: Save checkpoints every N iterations. If None, no checkpoints are saved.
        save_dir: Directory to save checkpoints. Required if save_every is specified.
        validate_every: Run centroid validation every N iterations. If None, only validate at the end.
        log_level_numeric: Numeric log level for conditional validation (if None, always run validation)
        device_type: Device type ("cuda" or "xpu", defaults to "cuda")

    Returns:
        centroid_sets: List of cluster centroids, each element of shape (K, D)
        top_k: Topk value of the model used to generate the activations
        losses: Losses for each iteration, shape (num_K, T)
        num_layers: Number of layers with routers
        num_experts: Number of experts per layer
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # CUDA contexts cannot be forked, so we must use spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        # Start method already set, which is fine
        logger.debug(f"Multiprocessing start method already set: {e}")

    # Get backend once and reuse throughout the function
    backend = get_backend(device_type)

    th.manual_seed(seed)
    backend.manual_seed(seed)

    num_gpus = backend.device_count()
    rank = dist.get_rank()
    num_nodes = dist.get_world_size()
    total_gpus = num_gpus * num_nodes

    logger.debug(f"Running kmeans with device type: {device_type}")
    logger.trace(f"Number of devices: {num_gpus}")
    logger.trace(f"Number of nodes: {num_nodes}")
    logger.trace(f"Total number of devices: {total_gpus}")

    assert backend.is_available() and num_gpus > 0, (
        f"CPU-only not supported yet :( Device {device_type} not available."
    )

    if effective_batch_size is None:
        effective_batch_size = (len(activations) // total_gpus) * total_gpus
        logger.trace(f"Size of activations: {len(activations)}")

    logger.trace(f"Effective batch size: {effective_batch_size}")

    if (leftover_batch_size := (effective_batch_size % total_gpus)) > 0:
        logger.warning(
            f"Effective batch size {effective_batch_size} is not divisible by total number of gpus {total_gpus}; {leftover_batch_size} left over"
        )
        effective_batch_size -= leftover_batch_size

    batch_size = effective_batch_size // total_gpus

    if minibatch_size is None:
        minibatch_size = batch_size

    if (leftover_minibatch_size := (batch_size % minibatch_size)) > 0:
        total_leftover_minibatch_size = leftover_minibatch_size * total_gpus
        logger.warning(
            f"Per-GPU batch size {batch_size} is not divisible by GPU minibatch size "
            f"{minibatch_size}; {leftover_minibatch_size} left over per GPU, "
            f"{total_leftover_minibatch_size} left over total"
        )
        batch_size -= leftover_minibatch_size
        effective_batch_size -= total_leftover_minibatch_size

    num_discarded_datapoints = (
        leftover_minibatch_size * total_gpus + leftover_batch_size
    )
    if num_discarded_datapoints > 0:
        logger.warning(
            f"{leftover_minibatch_size * total_gpus + leftover_batch_size} data points discarded"
        )

    assert minibatch_size > 0, "minibatch_size must be positive"
    assert batch_size > 0, "batch_size must be positive"
    assert effective_batch_size % total_gpus == 0, (
        f"effective_batch_size {effective_batch_size} must be a multiple of total_gpus {total_gpus}"
    )
    assert effective_batch_size / batch_size == total_gpus, (
        f"effective_batch_size {effective_batch_size} must be batch_size {batch_size} times total_gpus {total_gpus}"
    )

    assert batch_size % minibatch_size == 0, (
        f"batch_size {batch_size} must be a multiple of minibatch_size {minibatch_size}"
    )

    accumulation_size = batch_size // minibatch_size

    num_gpu_minibatches = len(activations) // minibatch_size

    logger.trace(f"Accumulation size: {accumulation_size}")
    logger.trace(f"Number of GPU minibatches: {num_gpu_minibatches}")

    all_gpu_data = [
        GPUData(
            synced_data=RunningKMeansData(
                centroid_sets=[
                    th.empty(
                        k, activation_dim, dtype=th.float32, device=th.device("cpu")
                    )
                    for k in k_values
                ],
                weight_sets=[
                    th.zeros(k, dtype=th.int64, device=th.device("cpu"))
                    for k in k_values
                ],
                losses=th.zeros(
                    len(k_values), dtype=th.float32, device=th.device("cpu")
                ),
            ),
            dirty_data=RunningKMeansData(
                centroid_sets=[
                    th.empty(
                        k, activation_dim, dtype=th.float32, device=th.device("cpu")
                    )
                    for k in k_values
                ],
                weight_sets=[
                    th.zeros(k, dtype=th.int64, device=th.device("cpu"))
                    for k in k_values
                ],
                losses=th.zeros(
                    len(k_values), dtype=th.float32, device=th.device("cpu")
                ),
            ),
            queue=mp.Queue(maxsize=GPU_QUEUE_MAXSIZE),
        )
        for gpu_idx in range(num_gpus)
    ]

    for gpu_idx, gpu_data in enumerate(all_gpu_data):
        # Mark tensors for shared memory across processes (required for spawn method)
        for centroids in gpu_data.synced_data.centroid_sets:
            centroids.share_memory_()
        for weights in gpu_data.synced_data.weight_sets:
            weights.share_memory_()
        gpu_data.synced_data.losses.share_memory_()

        for centroids in gpu_data.dirty_data.centroid_sets:
            centroids.share_memory_()
        for weights in gpu_data.dirty_data.weight_sets:
            weights.share_memory_()
        gpu_data.dirty_data.losses.share_memory_()
        logger.trace(
            f"GPU {gpu_idx} synced centroid sets {type(gpu_data.synced_data.centroid_sets)} {gpu_data.synced_data.centroid_sets[0].shape} {gpu_data.synced_data.centroid_sets[0].dtype} {gpu_data.synced_data.centroid_sets[0].device}"
        )
        logger.trace(
            f"GPU {gpu_idx} dirty centroid sets {type(gpu_data.dirty_data.centroid_sets)} {gpu_data.dirty_data.centroid_sets[0].shape} {gpu_data.dirty_data.centroid_sets[0].dtype} {gpu_data.dirty_data.centroid_sets[0].device}"
        )
        logger.trace(
            f"GPU {gpu_idx} synced weight sets {type(gpu_data.synced_data.weight_sets)} {gpu_data.synced_data.weight_sets[0].shape} {gpu_data.synced_data.weight_sets[0].dtype} {gpu_data.synced_data.weight_sets[0].device}"
        )
        logger.trace(
            f"GPU {gpu_idx} dirty weight sets {type(gpu_data.dirty_data.weight_sets)} {gpu_data.dirty_data.weight_sets[0].shape} {gpu_data.dirty_data.weight_sets[0].dtype} {gpu_data.dirty_data.weight_sets[0].device}"
        )
        logger.trace(
            f"GPU {gpu_idx} synced losses {type(gpu_data.synced_data.losses)} {gpu_data.synced_data.losses.shape} {gpu_data.synced_data.losses.dtype} {gpu_data.synced_data.losses.device}"
        )
        logger.trace(
            f"GPU {gpu_idx} dirty losses {type(gpu_data.dirty_data.losses)} {gpu_data.dirty_data.losses.shape} {gpu_data.dirty_data.losses.dtype} {gpu_data.dirty_data.losses.device}"
        )

    logger.trace(f"Initialized GPU data for {len(all_gpu_data)} GPUs")

    ### get top_k and initialize centroids from random data points

    # load a batch of activations to initialize the centroids and reserve validation data
    # Reserve extra data for validation (we'll use VALIDATION_SIZE_K_PROPORTION x the largest k value)
    max_k = max(k_values)
    validation_size = max_k * VALIDATION_SIZE_K_PROPORTION

    logger.trace(f"Validation size: {validation_size}")
    logger.trace(f"Max k: {max_k}")
    logger.trace(f"Total size: {max_k + validation_size}")

    data_iterable = activations(batch_size=max_k + validation_size)
    try:
        activation_batch = next(data_iterable)
    except StopIteration:
        logger.error(
            f"Not enough activations found for validation size {validation_size} and max k {max_k}"
        )
        raise
    data_iterable.close()
    router_activations = activation_batch[ActivationKeys.ROUTER_LOGITS]
    top_k = activation_batch["topk"]

    # Extract shape information for metadata
    # router_activations shape: (B, L, E) where L=layers, E=experts
    num_layers = router_activations.shape[1]
    num_experts = router_activations.shape[2]

    logger.debug(
        f"Extracted shape info: num_layers={num_layers}, num_experts={num_experts}"
    )
    logger.debug(
        f"Activation dim check: {num_layers * num_experts} == {activation_dim}"
    )

    assert num_layers * num_experts == activation_dim, (
        f"Shape mismatch: num_layers ({num_layers}) * num_experts ({num_experts}) = "
        f"{num_layers * num_experts} != activation_dim ({activation_dim})"
    )

    # Split into initialization and validation data
    init_activation_logits = router_activations[:max_k]
    validation_router_logits = router_activations[max_k:]

    logger.trace(
        f"Router logits {router_activations.shape} {router_activations.dtype} {router_activations.device}"
    )
    logger.trace(
        f"Init router logits {init_activation_logits.shape} {init_activation_logits.dtype} {init_activation_logits.device}"
    )
    logger.trace(
        f"Validation router logits {validation_router_logits.shape} {validation_router_logits.dtype} {validation_router_logits.device}"
    )

    logger.debug(
        f"Init router logits stats: "
        f"Min: {init_activation_logits.min()}, "
        f"Max: {init_activation_logits.max()}, "
        f"Mean: {init_activation_logits.mean()}, "
        f"Std: {init_activation_logits.std()}"
    )
    logger.debug(
        f"Validation router logits stats: "
        f"Min: {validation_router_logits.min()}, "
        f"Max: {validation_router_logits.max()}, "
        f"Mean: {validation_router_logits.mean()}, "
        f"Std: {validation_router_logits.std()}"
    )

    # Convert validation data to flat paths format (same as training data)
    validation_data = (
        convert_router_logits_to_paths(validation_router_logits, top_k)
        .view(validation_router_logits.shape[0], -1)
        .to(dtype=th.float32, device=th.device("cpu"))
    )
    init_activation_data = (
        convert_router_logits_to_paths(init_activation_logits, top_k)
        .view(init_activation_logits.shape[0], -1)
        .to(dtype=th.float32, device=th.device("cpu"))
    )

    logger.debug(
        f"Init activation data stats: "
        f"Min: {init_activation_data.min()}, "
        f"Max: {init_activation_data.max()}, "
        f"Mean: {init_activation_data.mean()}, "
        f"Std: {init_activation_data.std()}"
    )
    logger.debug(
        f"Validation data stats: "
        f"Min: {validation_data.min()}, "
        f"Max: {validation_data.max()}, "
        f"Mean: {validation_data.mean()}, "
        f"Std: {validation_data.std()}"
    )

    validate_centroids = partial(
        validate_centroid_distribution,
        validation_data,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        device_type=device_type,
    )

    logger.info(
        f"Reserved {validation_size} data points for validation (shape: {validation_data.shape})"
    )

    for k_idx, k in enumerate(k_values):
        for _gpu_idx, gpu_data in enumerate(all_gpu_data):
            current_device = gpu_data.dirty_data.centroid_sets[k_idx].device
            current_dtype = gpu_data.dirty_data.centroid_sets[k_idx].dtype
            current_shape = gpu_data.dirty_data.centroid_sets[k_idx].shape

            logger.trace(
                f"Initializing centroid set {k_idx} for k {k} with shape {current_shape} on device {current_device} and dtype {current_dtype}"
            )

            gpu_data.dirty_data.centroid_sets[k_idx] = (
                init_activation_data[:k]
                .view(current_shape)
                .to(device=current_device, dtype=current_dtype)
            )

            initialized_centroids = gpu_data.dirty_data.centroid_sets[k_idx]
            if th.isnan(initialized_centroids).any():
                logger.error(f"NaN values in initialized centroids for k={k}!")
            if not th.isfinite(initialized_centroids).all():
                logger.error(f"Non-finite values in initialized centroids for k={k}!")

            centroid_norms = th.norm(initialized_centroids, dim=1)
            zero_norm_count = (centroid_norms == 0).sum().item()
            if zero_norm_count > 0:
                logger.debug(
                    f"Initialized centroids for k={k}: {zero_norm_count} have zero norm"
                )
                logger.debug(
                    f"Centroid norm stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}"
                )

    trace_level_numeric = logger.level("TRACE").no

    # Only run detailed validation at TRACE level
    if log_level_numeric is None or log_level_numeric <= trace_level_numeric:
        logger.debug("🔍 VALIDATION: Checking centroids AFTER initialization...")
        for k_idx, centroid_set in enumerate(all_gpu_data[0].dirty_data.centroid_sets):
            _is_valid, _stats = validate_centroids(centroid_set.cpu())
            logger.debug(
                f"📊 POST-INIT VALIDATION k_idx={k_idx}: Empty={_stats.num_empty_centroids}, Norms: min={_stats.min_norm:.6f}, max={_stats.max_norm:.6f}, mean={_stats.mean_norm:.6f}"
            )

    logger.trace(f"Initialized centroids for {len(k_values)} clusters")

    logger.debug("🔄 Copying initialized centroids to synced_data...")
    for gpu_data in all_gpu_data:
        for k_idx in range(len(k_values)):
            gpu_data.synced_data.centroid_sets[k_idx].copy_(
                gpu_data.dirty_data.centroid_sets[k_idx]
            )

    # Validate that synced_data now has proper centroids (only at TRACE level)
    if log_level_numeric is None or log_level_numeric <= trace_level_numeric:
        logger.debug("🔍 VALIDATION: Checking synced_data after initial sync...")
        for k_idx, centroid_set in enumerate(all_gpu_data[0].synced_data.centroid_sets):
            _is_valid, _stats = validate_centroids(centroid_set.cpu())
            logger.debug(
                f"📊 POST-SYNC VALIDATION k_idx={k_idx}: Empty={_stats.num_empty_centroids}, Norms: min={_stats.min_norm:.6f}, max={_stats.max_norm:.6f}, mean={_stats.mean_norm:.6f}"
            )

    # clean up the background workers and queue
    backend.empty_cache()
    gc.collect()
    ### end

    # track losses for each iteration
    losses_over_time = []

    # Validate save_every parameter
    if save_every is not None:
        assert save_dir is not None, (
            "save_dir must be specified if save_every is provided"
        )
        logger.info(f"Checkpointing enabled: saving every {save_every} batches")

    iterator = range(max_iters)
    if dist.get_rank() == 0:
        iterator = tqdm(
            iterator, desc="Kmeans iterations", leave=False, total=max_iters, position=0
        )

    logger.trace("Created iterator")

    synchronization_barrier = th.multiprocessing.Barrier(num_gpus)
    workers = [
        mp.Process(
            target=gpu_worker,
            args=(
                gpu_idx,
                all_gpu_data,
                top_k,
                losses_over_time,
                synchronization_barrier,
                rank,  # rank
                num_nodes,  # world_size
                save_dir,
                validate_every,
                centroid_minibatch_size,
                assignment_minibatch_size,
                device_type,
            ),
            name=str(gpu_idx),
        )
        for gpu_idx in range(num_gpus)
    ]

    # Start all workers
    for worker in workers:
        worker.start()
    logger.trace(f"Created {len(workers)} workers")

    # distributed kmeans
    for iter_idx in iterator:
        # process data in batches, parallelized over devices and nodes
        logger.trace(f"🚀 Starting k-means iteration {iter_idx}")

        logger.trace(f"Running iteration {iter_idx}")

        # Check worker health at start of each iteration
        workers_dict = {f"GPU {i}": worker for i, worker in enumerate(workers)}
        check_worker_health(workers_dict, context=f"iteration {iter_idx}")

        # Log queue states for observability
        queue_sizes = [gpu_data.queue.qsize() for gpu_data in all_gpu_data]
        logger.trace(f"Iteration {iter_idx} - Queue sizes: {queue_sizes}")

        # skip over the first validation_size data points for validation
        minibatch_iterator = activations(
            batch_size=minibatch_size, start_idx=validation_size
        )

        distributed_iterator = islice(
            minibatch_iterator,
            rank,
            None,
            num_nodes,
        )
        logger.trace("Created distributed iterator")

        num_local_minibatches = len(
            range(
                rank,
                num_gpu_minibatches,
                num_nodes,
            )
        )
        distributed_iterator = tqdm(
            distributed_iterator,
            desc=f"Rank {rank}",
            total=num_local_minibatches,
            leave=False,
            position=num_nodes + 1,
        )

        concurrent_minibatch_iterator = batched(distributed_iterator, num_gpus)

        logger.trace("Created concurrent minibatch iterator")

        for distributed_batch_idx, gpu_minibatches in enumerate(
            concurrent_minibatch_iterator
        ):
            logger.trace(f"Running distributed batch {distributed_batch_idx}")

            # Periodic worker health check during long iterations
            if distributed_batch_idx % 10 == 0:
                check_worker_health(
                    workers_dict, context=f"batch {distributed_batch_idx}"
                )

            should_sync = (distributed_batch_idx % accumulation_size) == (
                accumulation_size - 1
            )

            # Determine if we should save at this batch
            # save_idx is the batch number if we should save, None otherwise
            if save_every is not None and should_save_checkpoint(
                distributed_batch_idx, save_every
            ):
                save_idx = distributed_batch_idx
            else:
                save_idx = None

            logger.trace(f"Should sync: {should_sync}")
            logger.trace(f"Accumulation size: {accumulation_size}")
            logger.trace(f"Distributed batch index: {distributed_batch_idx}")
            logger.trace(f"Save index: {save_idx}")

            for gpu_idx, (gpu_data, gpu_minibatch) in enumerate(
                zip(all_gpu_data, gpu_minibatches, strict=False)
            ):
                logger.trace(
                    f"Putting data on GPU {gpu_idx} with queue size {gpu_data.queue.qsize()}"
                )

                gpu_data.queue.put(
                    (
                        gpu_minibatch[ActivationKeys.ROUTER_LOGITS],
                        should_sync,
                        save_idx,
                    )
                )

        # log avg of last n losses from the last iteration
        num_losses_to_log = 10
        if len(losses_over_time) > 0:
            losses_tensor = th.stack(losses_over_time, dim=1)
            last_n_losses = losses_tensor[:, -num_losses_to_log:]
            avg_losses = last_n_losses.mean(dim=1)
            avg_loss_strings = [
                f"k={k_values[k_idx]}: {avg_loss:.6f}"
                for k_idx, avg_loss in enumerate(avg_losses)
            ]
            logger.info(
                f"Average losses over last {num_losses_to_log} iterations:\n\t{'\n\t'.join(avg_loss_strings)}"
            )

    for gpu_data in all_gpu_data:
        logger.trace(
            f"Putting stop signal on GPU with queue size {gpu_data.queue.qsize()}"
        )

        gpu_data.queue.put((None, False, None))

    if losses_over_time:
        logger.trace("Stacking losses over time")

        losses = th.stack(losses_over_time, dim=1)
    else:
        logger.trace("No losses over time, creating empty tensor")

        num_k_values = len(all_gpu_data[0].synced_data.centroid_sets)
        losses = th.empty((num_k_values, 0))

    logger.trace("Returning results")

    return (
        all_gpu_data[0].synced_data.centroid_sets,
        top_k,
        losses,
        num_layers,
        num_experts,
    )


def cluster_paths_main(
    model_name: str,
    dataset_name: str,
    activations: Activations,
    activation_dim: int,  # router activation dimension
    k: tuple[int, ...],
    batch_size: int,
    max_iters: int,
    seed: int,
    tokens_per_file: int,
    minibatch_size: int,
    centroid_minibatch_size: int = 32768,
    assignment_minibatch_size: int = 4096,
    save_every: int | None = None,
    validate_every: int = 64,
    log_level_numeric: int | None = None,
    device_type: DeviceType = "cuda",
) -> None:
    kmeans_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        tokens_per_file=tokens_per_file,
        type=KMEANS_TYPE,
    )

    logger.debug(f"Running kmeans with experiment name: {kmeans_experiment_name}")

    save_dir = os.path.join(OUTPUT_DIR, kmeans_experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger.trace(f"Save directory: {save_dir}")

    centroids, top_k, losses, num_layers, num_experts = kmeans_manhattan(
        activations=activations,
        activation_dim=activation_dim,
        k_values=k,
        effective_batch_size=batch_size,
        max_iters=max_iters,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        assignment_minibatch_size=assignment_minibatch_size,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        validate_every=validate_every,
        log_level_numeric=log_level_numeric,
        device_type=device_type,
    )

    if dist.get_rank() == 0:
        logger.info("Saving...")

        out = {
            "centroids": centroids,
            "top_k": top_k,
            "losses": losses,
        }
        out_path = os.path.join(save_dir, KMEANS_FILENAME)
        th.save(out, out_path)

        out_metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "activation_dim": activation_dim,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "k": k,
            "max_iters": max_iters,
            "seed": seed,
            "tokens_per_file": tokens_per_file,
            "minibatch_size": minibatch_size,
            "centroid_minibatch_size": centroid_minibatch_size,
            "save_every": save_every,
            "type": KMEANS_TYPE,
            "kmeans_experiment_name": kmeans_experiment_name,
        }
        out_metadata_path = os.path.join(save_dir, METADATA_FILENAME)
        with open(out_metadata_path, "w") as f:
            yaml.dump(out_metadata, f)

        logger.info("done :)")


def cluster_paths(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    k: tuple[int, ...] | int | None = None,
    expansion_factor: tuple[int, ...] | int | None = None,
    batch_size: int = 40_000,
    max_iters: int = 16,
    save_every: int | None = None,
    validate_every: int = 64,
    seed: int = 0,
    minibatch_size: int = 100_000,
    centroid_minibatch_size: int = 32768,
    assignment_minibatch_size: int = 4096,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
    device_type: DeviceType = "cuda",
) -> None:
    print(f"Running with log level: {log_level}")

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.debug(f"Running with log level: {log_level}")

    log_level_numeric = logger.level(log_level).no
    debug_level_numeric = logger.level("DEBUG").no

    activations, activation_dims = asyncio.run(
        load_activations_and_init_dist(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            submodule_names=[ActivationKeys.ROUTER_LOGITS, ActivationKeys.MLP_OUTPUT],
            context_length=context_length,
            num_workers=num_workers,
            debug=log_level_numeric <= debug_level_numeric,
            device_type=device_type,
        )
    )

    residual_activation_dim = activation_dims[ActivationKeys.MLP_OUTPUT]
    router_activation_dim = activation_dims[ActivationKeys.ROUTER_LOGITS]

    assert residual_activation_dim > 0, (
        "Residual activation dimension must be greater than 0"
    )
    assert router_activation_dim > 0, (
        "Router activation dimension must be greater than 0"
    )

    match k, expansion_factor:
        case None, None:
            # 1 to 131072
            k = tuple(2**i for i in range(17))
        case None, int(ef):
            k = (ef * residual_activation_dim,)
        case int(k_val), None:
            k = (k_val,)
        case ((None | tuple()), tuple(ef_tuple)):
            k = tuple(
                current_expansion_factor * residual_activation_dim
                for current_expansion_factor in ef_tuple
            )
        case tuple(), None:
            pass
        case _, _:
            raise ValueError("Cannot specify both k and expansion_factor")

    # At this point, k is guaranteed to be a tuple[int, ...]
    assert isinstance(k, tuple), "k must be a tuple after processing"

    # Validate device type
    assert_device_type(device_type)

    logger.info(f"Running with device type: {device_type}")

    cluster_paths_main(
        model_name=model_name,
        dataset_name=dataset_name,
        activations=activations,
        activation_dim=router_activation_dim,
        k=k,
        batch_size=batch_size,
        max_iters=max_iters,
        seed=seed,
        tokens_per_file=reshuffled_tokens_per_file,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        assignment_minibatch_size=assignment_minibatch_size,
        save_every=save_every,
        validate_every=validate_every,
        log_level_numeric=log_level_numeric,
        device_type=device_type,
    )


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *args: Any,
    k: tuple[int, ...] | None = None,
    expansion_factor: tuple[int, ...] | None = None,
    batch_size: int = 40_000,
    max_iters: int = 16,
    save_every: int | None = None,
    validate_every: int = 64,
    seed: int = 0,
    minibatch_size: int = 10_000,
    centroid_minibatch_size: int = 32768,
    assignment_minibatch_size: int = 4096,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
    device_type: str = "cuda",
) -> None:
    cluster_paths(
        model_name,
        dataset_name,
        *args,
        k=k,
        expansion_factor=expansion_factor,
        batch_size=batch_size,
        max_iters=max_iters,
        save_every=save_every,
        validate_every=validate_every,
        seed=seed,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        assignment_minibatch_size=assignment_minibatch_size,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        context_length=context_length,
        log_level=log_level,
        num_workers=num_workers,
        device_type=assert_device_type(device_type),
    )


if __name__ == "__main__":
    arguably.run()
