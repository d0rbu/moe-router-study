import asyncio
from asyncio import Barrier
from collections.abc import Awaitable
from dataclasses import dataclass
from functools import partial
import gc
from itertools import batched, islice
import os
import sys
from typing import Any, TypeVar

import arguably
from loguru import logger
import torch as th
import torch.distributed as dist
from tqdm import tqdm
import yaml

from core.async_utils import handle_exceptions
from core.device import DeviceType, assert_device_type, get_backend, get_device, get_device_type_from_backend, get_device_from_backend
from core.moe import convert_router_logits_to_paths
from core.training import exponential_to_linear_save_steps
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

# Type alias for PyTorch device backends
DeviceBackend = th.cuda | th.xpu


def check_worker_health(workers: dict[str, asyncio.Task], *, context: str = "") -> None:
    """Check if any workers have failed and raise appropriate exceptions."""
    for worker_name, worker in workers.items():
        if worker.done():
            exception = worker.exception()
            context_str = f" [{context}]" if context else ""
            if exception:
                logger.error(f"{worker_name} worker failed{context_str}: {exception}")
                raise RuntimeError(f"{worker_name} worker failed") from exception
            else:
                logger.error(
                    f"{worker_name} worker completed unexpectedly{context_str}"
                )
                raise RuntimeError(f"{worker_name} worker completed unexpectedly")


async def safe_await_with_worker_check[T](
    awaitable: Awaitable[T],
    *,
    workers: dict[str, asyncio.Task] | None = None,
    timeout: float = 30.0,
    operation_name: str = "",
) -> T:
    """Safely await an operation with timeout and worker health checking on error."""
    if workers is None:
        workers = {}

    try:
        result = await asyncio.wait_for(awaitable, timeout=timeout)
        if operation_name:
            logger.trace(f"Successfully completed {operation_name}")
        else:
            logger.trace("Successfully completed")
        return result
    except TimeoutError:
        if operation_name:
            logger.error(f"Timeout during {operation_name} - workers may have failed!")
        else:
            logger.error("Operation timed out - workers may have failed!")
        # Check if any workers have failed
        for worker_name, worker in workers.items():
            if worker.done():
                exception = worker.exception()
                if exception:
                    logger.error(
                        f"{worker_name} worker failed with exception: {exception}"
                    )
                else:
                    logger.error(f"{worker_name} worker completed unexpectedly")
            else:
                logger.error(f"{worker_name} worker appears to be hanging")
        raise
    except Exception as e:
        if operation_name:
            logger.error(f"Unexpected error during {operation_name}: {e}")
        else:
            logger.error(f"Unexpected error: {e}")
        raise


@dataclass
class RunningKMeansData:
    # list of centroids of shape (K, D)
    centroid_sets: list[th.Tensor]
    # list of weights of shape (K) for online running updates
    weight_sets: list[th.Tensor]
    losses: th.Tensor

    def clone(self) -> "RunningKMeansData":
        return RunningKMeansData(
            centroid_sets=[centroids.clone() for centroids in self.centroid_sets],
            weight_sets=[weights.clone() for weights in self.weight_sets],
            losses=self.losses.clone(),
        )

    def to(self, device: th.device) -> "RunningKMeansData":
        return RunningKMeansData(
            centroid_sets=[centroids.to(device) for centroids in self.centroid_sets],
            weight_sets=[weights.to(device) for weights in self.weight_sets],
            losses=self.losses.to(device),
        )

    def __add__(self, other: "RunningKMeansData") -> "RunningKMeansData":
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

            # Check if all weights are zero (which would cause issues)
            if new_weights.sum() == 0:
                logger.warning(
                    "All weights are zero in __add__ - this might cause centroid issues"
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

            # Debug: Check for zero centroids after weighted averaging
            new_centroid_norms = th.norm(new_centroid_values, dim=1)
            zero_norms_after_add = (new_centroid_norms == 0).sum().item()
            if zero_norms_after_add > 0:
                logger.warning(
                    f"__add__ produced {zero_norms_after_add} zero-norm centroids"
                )
                logger.warning(
                    f"base_weight_proportion sum: {base_weight_proportion.sum():.6f}, other_weight_proportion sum: {other_weight_proportion.sum():.6f}"
                )
                logger.warning(
                    f"base_centroids norms: min={th.norm(base_centroids, dim=1).min():.6f}, max={th.norm(base_centroids, dim=1).max():.6f}"
                )
                logger.warning(
                    f"other_centroids norms: min={th.norm(other_centroids, dim=1).min():.6f}, max={th.norm(other_centroids, dim=1).max():.6f}"
                )

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
    queue: asyncio.Queue | None = None


async def compute_centroid_from_assignment(
    data: th.Tensor,
    assignments: th.Tensor,
    centroid_idx: int,
) -> tuple[th.Tensor, th.Tensor]:
    centroid_mask = assignments == centroid_idx
    num_assigned = centroid_mask.sum()

    if num_assigned == 0:
        logger.trace(
            f"Centroid {centroid_idx} has no assigned points, returning zero vector"
        )
        return th.zeros_like(data[0]), num_assigned

    new_centroid = data[centroid_mask].mean(dim=0)

    if th.isnan(new_centroid).any():
        logger.error(f"NaN detected in centroid {centroid_idx} computation!")
        logger.error(f"num_assigned: {num_assigned}, data shape: {data.shape}")
        logger.error(
            f"assigned data stats: min={data[centroid_mask].min()}, max={data[centroid_mask].max()}"
        )

    return new_centroid, num_assigned


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


async def kmeans_step(
    data: th.Tensor,  # (B, L * E)
    centroids: th.Tensor,  # (K, L * E)
    centroid_minibatch_size: int = 65536,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    logger.trace(
        f"Running kmeans step with {data.shape[0]} data points and {centroids.shape[0]} centroids"
    )
    logger.trace(f"Data: {data.dtype} {data.device} {data.shape}")
    logger.trace(f"Centroids: {centroids.dtype} {centroids.device} {centroids.shape}")

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

    # (B)
    assignments = th.argmin(distances, dim=1)
    logger.trace(f"Computed assignments with shape {assignments.shape}")

    # for calculating loss, we get the distances from each data point to the closest centroid
    centroid_distances = th.gather(distances, 1, assignments.unsqueeze(1))
    logger.trace(f"Computed centroid distances with shape {centroid_distances.shape}")

    centroid_awaitables = [
        compute_centroid_from_assignment(data, assignments, i)
        for i in range(centroids.shape[0])
    ]
    centroids_and_weights = await asyncio.gather(*centroid_awaitables)
    logger.trace(
        f"Computed centroids and weights with shape {len(centroids_and_weights)}"
    )

    new_loss = centroid_distances.mean()
    logger.trace(f"Computed new loss with shape {new_loss.shape}")

    new_centroids_raw, new_weights_raw = zip(*centroids_and_weights, strict=True)
    new_centroids = th.stack(new_centroids_raw, dim=0)
    new_weights = th.tensor(new_weights_raw, dtype=th.int64, device=data.device)

    # Log update statistics
    update_norms = th.norm(new_centroids, dim=1)
    zero_update_norms = (update_norms == 0).sum().item()
    total_weight = new_weights.sum().item()
    logger.debug(
        f"📊 UPDATE: Computed updates - zero_norms={zero_update_norms}/{len(update_norms)}, norm_stats: min={update_norms.min():.6f}, max={update_norms.max():.6f}, mean={update_norms.mean():.6f}, total_weight={total_weight}"
    )

    logger.trace(
        f"New centroids: {new_centroids.dtype} {new_centroids.device} {new_centroids.shape}"
    )
    logger.trace(
        f"New weights: {new_weights.dtype} {new_weights.device} {new_weights.shape}"
    )
    logger.trace(f"New loss: {new_loss.dtype} {new_loss.device} {new_loss.shape}")

    return new_centroids, new_weights, new_loss


async def sync(
    gpu_idx: int,
    all_gpu_data: list[GPUData],
    losses_over_time: list[th.Tensor],
    barrier: Barrier,
    group: dist.ProcessGroup | None = None,
    backend: DeviceBackend | None = None,
) -> None:
    if backend is None:
        raise ValueError("backend parameter is required")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpu_data = all_gpu_data[gpu_idx]

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
    all_losses = (
        th.empty_like(gpu_data.dirty_data.losses).unsqueeze(0).repeat(world_size, 1)
    )

    # N, num_K
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

        # (K)
        loss_proportion = all_weights.sum(dim=1) / weights_total.sum()
        # ()
        new_loss = (all_losses[:, losses_idx] * loss_proportion).sum()

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
            loss_proportion,
            new_loss,
        )

    # now do an all-gather along gpus (among entries in all_gpu_data)
    device = backend.get_device(gpu_idx)
    empty_data = RunningKMeansData(
        centroid_sets=[
            th.zeros_like(centroids) for centroids in gpu_data.synced_data.centroid_sets
        ],
        weight_sets=[
            th.zeros_like(weights) for weights in gpu_data.synced_data.weight_sets
        ],
        losses=th.zeros_like(gpu_data.synced_data.losses),
    )

    # Debug: Log synced data before update
    for k_idx, centroids in enumerate(gpu_data.synced_data.centroid_sets):
        centroid_norms = th.norm(centroids, dim=1)
        zero_norms = (centroid_norms == 0).sum().item()
        logger.debug(
            f"🔄 SYNC GPU {gpu_idx} k_idx={k_idx} BEFORE GPU AGGREGATION: Synced centroids zero_norms={zero_norms}/{len(centroid_norms)}, norm_stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}"
        )

    gpu_data.synced_data += sum(
        (current_gpu_data.dirty_data.to(device) for current_gpu_data in all_gpu_data),
        start=empty_data,
    )

    backend.empty_cache()

    await barrier.wait()

    # reset dirty data now that it has been synced
    logger.debug(f"🔄 SYNC GPU {gpu_idx}: Resetting dirty data to zero...")
    for weights in gpu_data.dirty_data.weight_sets:
        weights.zero_()

    for centroids in gpu_data.dirty_data.centroid_sets:
        centroids.zero_()

    # Log synced data state after sync
    for k_idx, centroids in enumerate(gpu_data.synced_data.centroid_sets):
        centroid_norms = th.norm(centroids, dim=1)
        zero_norms = (centroid_norms == 0).sum().item()
        logger.debug(
            f"🔄 SYNC GPU {gpu_idx} k_idx={k_idx} FINAL: Synced centroids zero_norms={zero_norms}/{len(centroid_norms)}, norm_stats: min={centroid_norms.min():.6f}, max={centroid_norms.max():.6f}, mean={centroid_norms.mean():.6f}"
        )

    if rank == 0 and gpu_idx == 0:
        losses_over_time.append(gpu_data.synced_data.losses.detach().cpu().clone())

        # Check for monotonically increasing loss windows
        if (
            len(losses_over_time) >= WARNING_WINDOW_SIZE
        ):  # Need at least window_size iterations
            losses_tensor = th.stack(losses_over_time, dim=1)
            has_problem, start_idx = check_monotonic_increasing_window(
                losses_tensor, window_size=WARNING_WINDOW_SIZE
            )
            if has_problem:
                logger.warning(
                    f"Detected monotonically increasing loss window starting at "
                    f"iteration {start_idx}. This may indicate a training problem."
                )


async def gpu_worker(
    gpu_idx: int,
    all_gpu_data: list[GPUData],
    top_k: int,
    losses_over_time: list[th.Tensor],
    barrier: Barrier,
    group: dist.ProcessGroup | None = None,
    save_dir: str | None = None,
    validate_every: int = 64,
    centroid_minibatch_size: int = 65536,
    backend: DeviceBackend | None = None,
) -> None:
    """
    GPU worker for distributed k-means clustering.

    Args:
        gpu_idx: Index of the device this worker is responsible for
        all_gpu_data: List of GPU data objects for all devices
        top_k: Number of top experts to consider
        losses_over_time: Shared list to store losses over time
        barrier: Synchronization barrier for coordinating workers
        group: Distributed process group for communication
        save_dir: Directory to save checkpoints (if any)
        validate_every: Validate centroid synchronization every N sync operations (default: 1)
        centroid_minibatch_size: Size of centroid chunks to avoid device limits (default: 65536)
        backend: Device backend object (required)
    """
    if backend is None:
        raise ValueError("backend parameter is required")

    logger.info(f"Starting GPU worker {gpu_idx}")
    gpu_data = all_gpu_data[gpu_idx]
    sync_iteration = 0

    while True:
        logger.trace(f"GPU {gpu_idx} waiting for queue item...")
        try:
            queue_item = await asyncio.wait_for(gpu_data.queue.get(), timeout=60.0)
            logger.trace(f"GPU {gpu_idx} picked up item from queue")
        except TimeoutError:
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
        device = backend.get_device(gpu_idx)
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
            f"GPU {gpu_idx} running kmeans step with {len(gpu_data.synced_data.centroid_sets)} centroids"
        )

        updates = await asyncio.gather(
            *[
                kmeans_step(
                    flat_data,
                    centroids,
                    centroid_minibatch_size,
                )
                for centroids in gpu_data.synced_data.centroid_sets
            ]
        )
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

        gpu_data.dirty_data += RunningKMeansData(
            centroid_sets=list(new_centroid_sets),
            weight_sets=list(new_weight_sets),
            losses=th.stack(new_losses),
        )

        logger.trace(f"GPU {gpu_idx} updated synced data")

        if not should_sync:
            logger.trace(f"GPU {gpu_idx} skipping sync, continuing to next item")
            continue

        logger.trace(f"GPU {gpu_idx} starting sync operation")

        await safe_await_with_worker_check(
            sync(
                gpu_idx,
                all_gpu_data,
                losses_over_time,
                barrier,
                group,
                backend,
            ),
            timeout=300.0,
            operation_name=f"sync operation for GPU {gpu_idx}",
        )

        logger.trace(f"GPU {gpu_idx} completed sync operation")

        # Increment sync iteration counter
        sync_iteration += 1

        # Validate GPU synchronization after sync (only on GPU 0 to avoid redundant checks)
        # Only validate every validate_every iterations
        if (
            gpu_idx == 0
            and (len(all_gpu_data) > 1 or dist.get_world_size() > 1)
            and sync_iteration % validate_every == 0
        ):
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

        # save checkpoint if save_idx is not None and we're on rank 0 gpu 0
        if (
            save_idx is not None
            and dist.get_rank() == 0
            and gpu_idx == 0
            and save_dir is not None
        ):
            logger.trace(f"GPU {gpu_idx} saving checkpoint")

            checkpoint_data = {
                "centroids": all_gpu_data[0].synced_data.centroid_sets,
                "top_k": top_k,
                "losses": th.stack(losses_over_time, dim=1)
                if losses_over_time
                else all_gpu_data[0].synced_data.losses,
                "iteration": save_idx,
            }
            checkpoint_path = os.path.join(
                save_dir, CHECKPOINT_FILENAME.format(iteration=save_idx)
            )
            logger.trace(f"GPU {gpu_idx} saved checkpoint to {checkpoint_path}")

            th.save(checkpoint_data, checkpoint_path)
            logger.info(
                f"Saved checkpoint at iteration {save_idx} to {checkpoint_path}"
            )


GPU_QUEUE_MAXSIZE = 4


async def kmeans_manhattan(
    activations: Activations,
    activation_dim: int,
    k_values: tuple[int, ...],
    effective_batch_size: int | None = None,
    max_iters: int = 128,
    minibatch_size: int | None = None,
    centroid_minibatch_size: int = 16384,
    seed: int = 0,
    save_every: int | None = None,
    save_dir: str | None = None,
    validate_every: int = 64,
    group: dist.ProcessGroup | None = None,
    backend: DeviceBackend | None = None,
) -> tuple[list[th.Tensor], int, th.Tensor]:
    """
    Perform k-means clustering with Manhattan distance.

    Args:
        activations: Activations to cluster
        k_values: List of number of clusters
        effective_batch_size: Batch size for k-means updates. If None, use the batch size of the activations.
        max_iters: Maximum number of iterations
        minibatch_size: Batch size for processing data. If None, process all data at once.
        centroid_minibatch_size: Size of centroid chunks to avoid device limits (defaults to 16384)
        seed: Random seed for initialization
        save_every: Save checkpoints every N iterations. If None, no checkpoints are saved.
        save_dir: Directory to save checkpoints. Required if save_every is specified.
        validate_every: Run centroid validation every N iterations. If None, only validate at the end.
        backend: Device backend object (if None, defaults to CUDA backend)

    Returns:
        centroid_sets: List of cluster centroids, each element of shape (K, D)
        top_k: Topk value of the model used to generate the activations
        losses: Losses for each iteration, shape (num_K, T)
    """
    # Get backend once and reuse throughout the function
    if backend is None:
        backend = get_backend("cuda")  # Default to CUDA for backward compatibility

    th.manual_seed(seed)
    backend.manual_seed(seed)

    num_gpus = backend.device_count()
    rank = dist.get_rank()
    num_nodes = dist.get_world_size()
    total_gpus = num_gpus * num_nodes

    logger.trace(f"Number of devices: {num_gpus}")
    logger.trace(f"Number of nodes: {num_nodes}")
    logger.trace(f"Total number of devices: {total_gpus}")

    assert backend.is_available() and num_gpus > 0, (
        f"CPU-only not supported yet :( Device {get_device_type_from_backend(backend)} not available."
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
                    th.empty(k, activation_dim, dtype=th.float32, device=gpu_idx)
                    for k in k_values
                ],
                weight_sets=[
                    th.zeros(k, dtype=th.int64, device=gpu_idx) for k in k_values
                ],
                losses=th.zeros(len(k_values), dtype=th.float32, device=gpu_idx),
            ),
            dirty_data=RunningKMeansData(
                centroid_sets=[
                    th.empty(k, activation_dim, dtype=th.float32, device=gpu_idx)
                    for k in k_values
                ],
                weight_sets=[
                    th.zeros(k, dtype=th.int64, device=gpu_idx) for k in k_values
                ],
                losses=th.zeros(len(k_values), dtype=th.float32, device=gpu_idx),
            ),
            queue=asyncio.Queue(maxsize=GPU_QUEUE_MAXSIZE),
        )
        for gpu_idx in range(num_gpus)
    ]

    for gpu_idx, gpu_data in enumerate(all_gpu_data):
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
        .to(dtype=th.float32, device="cpu")
    )
    init_activation_data = (
        convert_router_logits_to_paths(init_activation_logits, top_k)
        .view(init_activation_logits.shape[0], -1)
        .to(dtype=th.float32, device="cpu")
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

    # Validate that synced_data now has proper centroids
    logger.debug("🔍 VALIDATION: Checking synced_data after initial sync...")
    for k_idx, centroid_set in enumerate(all_gpu_data[0].synced_data.centroid_sets):
        _is_valid, _stats = validate_centroids(centroid_set.cpu())
        logger.debug(
            f"📊 POST-SYNC VALIDATION k_idx={k_idx}: Empty={_stats.num_empty_centroids}, Norms: min={_stats.min_norm:.6f}, max={_stats.max_norm:.6f}, mean={_stats.mean_norm:.6f}"
        )

    # clean up the background workers and queue
    th.cuda.empty_cache()
    gc.collect()
    ### end

    # track losses for each iteration
    losses_over_time = []

    # calculate save steps for checkpointing
    save_steps = set()
    if save_every is not None:
        assert save_dir is not None, (
            "save_dir must be specified if save_every is provided"
        )
        save_steps = exponential_to_linear_save_steps(
            total_steps=max_iters, save_every=save_every
        )

    logger.trace(f"Save steps: {save_steps}")

    iterator = range(max_iters)
    if dist.get_rank() == 0:
        iterator = tqdm(
            iterator, desc="Kmeans iterations", leave=False, total=max_iters, position=0
        )

    logger.trace("Created iterator")

    synchronization_barrier = Barrier(num_gpus)
    workers = [
        asyncio.create_task(
            gpu_worker(
                gpu_idx,
                all_gpu_data,
                top_k,
                losses_over_time,
                synchronization_barrier,
                group,
                save_dir,
                validate_every,
                centroid_minibatch_size,
                backend,
            ),
            name=str(gpu_idx),
        )
        for gpu_idx in range(num_gpus)
    ]

    for worker in workers:
        worker.add_done_callback(handle_exceptions)

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

            # compute effective step index and determine if we should save
            effective_step_idx = distributed_batch_idx // accumulation_size
            save_idx = effective_step_idx if effective_step_idx in save_steps else None

            logger.trace(f"Should sync: {should_sync}")
            logger.trace(f"Accumulation size: {accumulation_size}")
            logger.trace(f"Distributed batch index: {distributed_batch_idx}")
            logger.trace(f"Effective step index: {effective_step_idx}")
            logger.trace(f"Save index: {save_idx}")

            for gpu_idx, (gpu_data, gpu_minibatch) in enumerate(
                zip(all_gpu_data, gpu_minibatches, strict=False)
            ):
                logger.trace(
                    f"Putting data on GPU {gpu_idx} with queue size {gpu_data.queue.qsize()}"
                )

                await safe_await_with_worker_check(
                    gpu_data.queue.put(
                        (
                            gpu_minibatch[ActivationKeys.ROUTER_LOGITS],
                            should_sync,
                            save_idx,
                        )
                    ),
                    workers={f"GPU {gpu_idx}": workers[gpu_idx]},
                    timeout=3600.0,
                    operation_name=f"queue put for GPU {gpu_idx}",
                )

    for gpu_data in all_gpu_data:
        logger.trace(
            f"Putting stop signal on GPU with queue size {gpu_data.queue.qsize()}"
        )

        await gpu_data.queue.put((None, False, None))

    if losses_over_time:
        logger.trace("Stacking losses over time")

        losses = th.stack(losses_over_time, dim=1)
    else:
        logger.trace("No losses over time, creating empty tensor")

        num_k_values = len(all_gpu_data[0].synced_data.centroid_sets)
        losses = th.empty((num_k_values, 0))

    logger.trace("Returning results")

    return all_gpu_data[0].synced_data.centroid_sets, top_k, losses


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


async def cluster_paths_async(
    model_name: str,
    dataset_name: str,
    activations: Activations,
    activation_dim: int,  # router activation dimension
    k: tuple[int, ...],
    max_iters: int,
    seed: int,
    tokens_per_file: int,
    minibatch_size: int,
    centroid_minibatch_size: int = 16384,
    save_every: int | None = None,
    validate_every: int = 64,
    group: dist.ProcessGroup | None = None,
    backend: DeviceBackend | None = None,
) -> None:
    if backend is None:
        raise ValueError("backend parameter is required")

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

    centroids, top_k, losses = await kmeans_manhattan(
        activations=activations,
        activation_dim=activation_dim,
        k_values=k,
        max_iters=max_iters,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        validate_every=validate_every,
        group=group,
        backend=backend,
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
    max_iters: int = 128,
    save_every: int | None = None,
    validate_every: int = 64,
    seed: int = 0,
    minibatch_size: int = 100_000,
    centroid_minibatch_size: int = 16384,
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

    activations, activation_dims, gpu_process_group = asyncio.run(
        load_activations_and_init_dist(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            submodule_names=[ActivationKeys.ROUTER_LOGITS, ActivationKeys.MLP_OUTPUT],
            context_length=context_length,
            num_workers=num_workers,
            debug=log_level_numeric <= debug_level_numeric,
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
        case None, tuple(ef_tuple):
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

    # Get backend once for the entire operation
    backend = get_backend(device_type)

    asyncio.run(
        cluster_paths_async(
            model_name=model_name,
            dataset_name=dataset_name,
            activations=activations,
            activation_dim=router_activation_dim,
            k=k,
            max_iters=max_iters,
            seed=seed,
            tokens_per_file=reshuffled_tokens_per_file,
            minibatch_size=minibatch_size,
            centroid_minibatch_size=centroid_minibatch_size,
            save_every=save_every,
            validate_every=validate_every,
            group=gpu_process_group,
            backend=backend,
        )
    )


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *args: Any,
    k: tuple[int, ...] | None = None,
    expansion_factor: tuple[int, ...] | None = None,
    max_iters: int = 128,
    save_every: int | None = None,
    validate_every: int = 64,
    seed: int = 0,
    minibatch_size: int = 10_000,
    centroid_minibatch_size: int = 16384,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
    device_type: DeviceType = "cuda",
) -> None:
    cluster_paths(
        model_name,
        dataset_name,
        *args,
        k=k,
        expansion_factor=expansion_factor,
        max_iters=max_iters,
        save_every=save_every,
        validate_every=validate_every,
        seed=seed,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        context_length=context_length,
        log_level=log_level,
        num_workers=num_workers,
        device_type=device_type,
    )


if __name__ == "__main__":
    arguably.run()
