import asyncio
from asyncio import Barrier
from dataclasses import dataclass
import gc
from itertools import batched, islice, pairwise
import os
import sys
from typing import Any

import arguably
from loguru import logger
import torch as th
import torch.distributed as dist
from tqdm import tqdm
import yaml

from core.async_utils import handle_exceptions
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
from exp.training import exponential_to_linear_save_steps, get_experiment_name


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
            base_weight_proportion = th.zeros_like(base_weights, dtype=th.float32)
            base_weight_proportion[mask] = base_weights[mask] / new_weights[mask]
            other_weight_proportion = 1 - base_weight_proportion

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

            new_centroids.copy_(
                base_weight_proportion.unsqueeze(-1) * base_centroids
                + other_weight_proportion.unsqueeze(-1) * other_centroids
            )

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
        return th.zeros_like(data[0]), 0

    return data[centroid_mask].mean(dim=0), num_assigned


async def kmeans_step(
    data: th.Tensor,  # (B, L * E)
    centroids: th.Tensor,  # (K, L * E)
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    logger.trace(
        f"Running kmeans step with {data.shape[0]} data points and {centroids.shape[0]} centroids"
    )
    logger.trace(f"Data: {data.dtype} {data.device} {data.shape}")
    logger.trace(f"Centroids: {centroids.dtype} {centroids.device} {centroids.shape}")

    # (B, K)
    distances = th.cdist(data.to(th.float32), centroids.to(th.float32), p=1)
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
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpu_data = all_gpu_data[gpu_idx]

    # gather across nodes
    all_losses = (
        th.empty_like(gpu_data.dirty_data.losses).unsqueeze(0).repeat(world_size, 1)
    )

    # N, num_K
    dist.all_gather_into_tensor(all_losses, gpu_data.dirty_data.losses, group=group)

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
        # (N, K)
        weights_proportion = all_weights / weights_total

        # (K, D)
        new_centroids = (all_centroids * weights_proportion.unsqueeze(-1)).sum(dim=0)

        # (K)
        loss_proportion = all_weights.sum(dim=1) / weights_total.sum()
        # ()
        new_loss = (all_losses[:, losses_idx] * loss_proportion).sum()

        gpu_data.dirty_data.centroid_sets[losses_idx] = new_centroids
        gpu_data.dirty_data.weight_sets[losses_idx] = weights_total
        gpu_data.dirty_data.losses[losses_idx] = new_loss

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
    device = th.device(f"cuda:{gpu_idx}")
    empty_data = RunningKMeansData(
        centroid_sets=[
            th.empty_like(centroids) for centroids in gpu_data.synced_data.centroid_sets
        ],
        weight_sets=[
            th.zeros_like(weights) for weights in gpu_data.synced_data.weight_sets
        ],
        losses=th.empty_like(gpu_data.synced_data.losses),
    )
    gpu_data.synced_data += sum(
        (current_gpu_data.dirty_data.to(device) for current_gpu_data in all_gpu_data),
        start=empty_data,
    )

    th.cuda.empty_cache()

    await barrier.wait()

    # reset dirty data now that it has been synced
    for weights in gpu_data.dirty_data.weight_sets:
        weights.zero_()

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
) -> None:
    logger.info(f"Starting GPU worker {gpu_idx}")
    gpu_data = all_gpu_data[gpu_idx]

    while True:
        queue_item = await gpu_data.queue.get()
        logger.trace(f"GPU {gpu_idx} picked up item from queue")

        if queue_item is None:
            logger.trace(f"GPU {gpu_idx} received stop signal")
            break

        router_logits, should_sync, save_idx = queue_item

        logger.trace(f"GPU {gpu_idx} received queue item")

        # assert that if save_idx is not None, then should_sync is also true
        if save_idx is not None:
            assert should_sync, "save_idx can only be set when should_sync is true"

        logger.trace(f"GPU {gpu_idx} converting router logits to paths")

        # (B, L, E)
        device = th.device(f"cuda:{gpu_idx}")
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

        th.cuda.empty_cache()

        logger.trace(
            f"GPU {gpu_idx} running kmeans step with {len(gpu_data.synced_data.centroid_sets)} centroids"
        )

        updates = await asyncio.gather(
            *[
                kmeans_step(
                    flat_data,
                    centroids,
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
            continue

        logger.trace(f"GPU {gpu_idx} syncing data")

        await sync(gpu_idx, all_gpu_data, losses_over_time, barrier, group)

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
    k_values: list[int],
    effective_batch_size: int | None = None,
    max_iters: int = 128,
    gpu_minibatch_size: int | None = None,
    seed: int = 0,
    save_every: int | None = None,
    save_dir: str | None = None,
    validate_every: int | None = None,
    group: dist.ProcessGroup | None = None,
) -> tuple[list[th.Tensor], int, th.Tensor]:
    """
    Perform k-means clustering with Manhattan distance.

    Args:
        activations: Activations to cluster
        k_values: List of number of clusters
        effective_batch_size: Batch size for k-means updates. If None, use the batch size of the activations.
        max_iters: Maximum number of iterations
        minibatch_size: Batch size for processing data. If None, process all data at once.
        seed: Random seed for initialization
        save_every: Save checkpoints every N iterations. If None, no checkpoints are saved.
        save_dir: Directory to save checkpoints. Required if save_every is specified.
        validate_every: Run centroid validation every N iterations. If None, only validate at the end.

    Returns:
        centroid_sets: List of cluster centroids, each element of shape (K, D)
        top_k: Topk value of the model used to generate the activations
        losses: Losses for each iteration, shape (num_K, T)
    """
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    num_gpus = th.cuda.device_count()
    rank = dist.get_rank()
    num_nodes = dist.get_world_size()
    total_gpus = num_gpus * num_nodes

    logger.trace(f"Number of GPUs: {num_gpus}")
    logger.trace(f"Number of nodes: {num_nodes}")
    logger.trace(f"Total number of GPUs: {total_gpus}")

    assert th.cuda.is_available() and num_gpus > 0, "CPU-only not supported yet :("

    if effective_batch_size is None:
        effective_batch_size = (len(activations) // total_gpus) * total_gpus

    if (leftover_batch_size := (effective_batch_size % total_gpus)) > 0:
        logger.warning(
            f"Effective batch size {effective_batch_size} is not divisible by total number of gpus {total_gpus}; {leftover_batch_size} left over"
        )
        effective_batch_size -= leftover_batch_size

    batch_size = effective_batch_size // total_gpus

    if gpu_minibatch_size is None:
        gpu_minibatch_size = batch_size

    if (leftover_minibatch_size := (batch_size % gpu_minibatch_size)) > 0:
        total_leftover_minibatch_size = leftover_minibatch_size * total_gpus
        logger.warning(
            f"Per-GPU batch size {batch_size} is not divisible by GPU minibatch size "
            f"{gpu_minibatch_size}; {leftover_minibatch_size} left over per GPU, "
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

    assert gpu_minibatch_size > 0, "gpu_minibatch_size must be positive"
    assert batch_size > 0, "batch_size must be positive"
    assert effective_batch_size % total_gpus == 0, (
        f"effective_batch_size {effective_batch_size} must be a multiple of total_gpus {total_gpus}"
    )
    assert effective_batch_size / batch_size == total_gpus, (
        f"effective_batch_size {effective_batch_size} must be batch_size {batch_size} times total_gpus {total_gpus}"
    )
    assert batch_size % gpu_minibatch_size == 0, (
        f"batch_size {batch_size} must be a multiple of gpu_minibatch_size {gpu_minibatch_size}"
    )

    accumulation_size = batch_size // gpu_minibatch_size

    num_gpu_minibatches = len(activations) // gpu_minibatch_size

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
    k_ranges = th.cat(
        [
            th.tensor([0], device=0),
            th.cumsum(th.tensor(k_values, device=0), dim=0),
        ],
        dim=0,
    )

    # load a batch of activations to initialize the centroids and reserve validation data
    # Reserve extra data for validation (we'll use VALIDATION_SIZE_K_PROPORTION x the largest k value)
    validation_size = max(k_values) * VALIDATION_SIZE_K_PROPORTION
    num_total_centroids = k_ranges[-1].item()
    total_init_size = num_total_centroids + validation_size

    data_iterable = activations(batch_size=total_init_size)
    activation_batch = next(data_iterable)
    router_activations = activation_batch[ActivationKeys.ROUTER_LOGITS]
    top_k = activation_batch["topk"]

    # Split into initialization and validation data
    init_activations = router_activations[:num_total_centroids]
    validation_router_logits = router_activations[num_total_centroids:]

    # Convert validation data to flat paths format (same as training data)
    validation_data = convert_router_logits_to_paths(
        validation_router_logits, top_k
    ).to(dtype=th.float32, device="cpu")

    logger.info(
        f"Reserved {validation_size} data points for validation (shape: {validation_data.shape})"
    )

    for k_idx, (k_start, k_end) in enumerate(pairwise(k_ranges)):
        k = k_values[k_idx]
        assert k_end - k_start == k, "k_end - k_start must be equal to k"

        for _gpu_idx, gpu_data in enumerate(all_gpu_data):
            current_device = gpu_data.dirty_data.centroid_sets[k_idx].device
            current_dtype = gpu_data.dirty_data.centroid_sets[k_idx].dtype
            current_shape = gpu_data.dirty_data.centroid_sets[k_idx].shape

            gpu_data.dirty_data.centroid_sets[k_idx] = (
                init_activations[k_start:k_end]
                .view(current_shape)
                .to(device=current_device, dtype=current_dtype)
            )

    logger.trace(f"Initialized centroids for {len(k_values)} clusters")

    # clean up the background workers and queue
    data_iterable.close()
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
            ),
            name=str(gpu_idx),
        )
        for gpu_idx in range(num_gpus)
    ]

    # Add exception handling to all worker tasks
    for worker in workers:
        worker.add_done_callback(handle_exceptions)

    logger.trace(f"Created {len(workers)} workers")

    # distributed kmeans
    for iter_idx in iterator:
        # process data in batches, parallelized over devices and nodes
        logger.trace(f"Running iteration {iter_idx}")
        minibatch_iterator = activations(batch_size=gpu_minibatch_size)

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

            should_sync = distributed_batch_idx % accumulation_size == (
                accumulation_size - 1
            )

            # compute effective step index and determine if we should save
            effective_step_idx = distributed_batch_idx // accumulation_size
            save_idx = effective_step_idx if effective_step_idx in save_steps else None

            logger.trace(f"Should sync: {should_sync}")
            logger.trace(f"Effective step index: {effective_step_idx}")
            logger.trace(f"Save index: {save_idx}")

            for gpu_data, gpu_minibatch in zip(
                all_gpu_data, gpu_minibatches, strict=False
            ):
                logger.trace(
                    f"Putting data on GPU with queue size {gpu_data.queue.qsize()}"
                )

                await gpu_data.queue.put(
                    (
                        gpu_minibatch[ActivationKeys.ROUTER_LOGITS],
                        should_sync,
                        save_idx,
                    )
                )

        # Intermittent validation during training
        if (
            validate_every is not None
            and (iter_idx + 1) % validate_every == 0
            and rank == 0
        ):
            logger.info(
                f"Running intermittent validation at iteration {iter_idx + 1}..."
            )
            for _k_idx, centroid_set in enumerate(
                all_gpu_data[0].synced_data.centroid_sets
            ):
                is_valid, stats = validate_centroid_distribution(
                    validation_data,
                    centroid_set.cpu(),
                )
                k_value = centroid_set.shape[0]
                if is_valid:
                    logger.success(
                        f"✓ K={k_value} (iter {iter_idx + 1}): Valid. {stats}"
                    )
                else:
                    logger.warning(
                        f"✗ K={k_value} (iter {iter_idx + 1}): Issues. {stats}"
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
    activation_dim: int,
    k: list[int],
    max_iters: int,
    seed: int,
    tokens_per_file: int,
    gpu_minibatch_size: int,
    save_every: int | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    kmeans_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        tokens_per_file=tokens_per_file,
    )

    logger.debug(f"Running kmeans with experiment name: {kmeans_experiment_name}")

    save_dir = None
    if save_every is not None:
        save_dir = os.path.join(OUTPUT_DIR, kmeans_experiment_name)
        os.makedirs(save_dir, exist_ok=True)

    logger.trace(f"Save directory: {save_dir}")

    centroids, top_k, losses = await kmeans_manhattan(
        activations=activations,
        activation_dim=activation_dim,
        k_values=k,
        max_iters=max_iters,
        gpu_minibatch_size=gpu_minibatch_size,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        group=group,
    )

    if dist.get_rank() == 0:
        logger.info("Saving...")

        out = {
            "centroids": centroids,
            "top_k": top_k,
            "losses": losses,
        }
        out_path = os.path.join(OUTPUT_DIR, kmeans_experiment_name, KMEANS_FILENAME)
        th.save(out, out_path)

        out_metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "activation_dim": activation_dim,
            "k": k,
            "max_iters": max_iters,
            "seed": seed,
            "tokens_per_file": tokens_per_file,
            "gpu_minibatch_size": gpu_minibatch_size,
            "save_every": save_every,
            "type": KMEANS_TYPE,
        }
        out_metadata_path = os.path.join(
            OUTPUT_DIR, kmeans_experiment_name, METADATA_FILENAME
        )
        with open(out_metadata_path, "w") as f:
            yaml.dump(out_metadata, f)

        logger.info("done :)")


def cluster_paths(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    k: list[int] | int | None = None,
    expansion_factor: list[int] | int | None = None,
    max_iters: int = 128,
    save_every: int | None = None,
    seed: int = 0,
    gpu_minibatch_size: int = 100000,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
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
            submodule_names=[ActivationKeys.ROUTER_LOGITS],
            context_length=context_length,
            num_workers=num_workers,
            debug=log_level_numeric <= debug_level_numeric,
        )
    )
    activation_dim = activation_dims[ActivationKeys.ROUTER_LOGITS]

    assert activation_dim > 0, "Activation dimension must be greater than 0"

    match k, expansion_factor:
        case None, None:
            # 1 to 131072
            k = [2**i for i in range(17)]
        case None, int(ef):
            k = [ef * activation_dim]
        case int(k_val), None:
            k = [k_val]
        case None, list(ef_list):
            k = [
                current_expansion_factor * activation_dim
                for current_expansion_factor in ef_list
            ]
        case list(), None:
            pass
        case _, _:
            raise ValueError("Cannot specify both k and expansion_factor")

    # At this point, k is guaranteed to be a list[int]
    assert isinstance(k, list), "k must be a list after processing"

    asyncio.run(
        cluster_paths_async(
            model_name=model_name,
            dataset_name=dataset_name,
            activations=activations,
            activation_dim=activation_dim,
            k=k,
            max_iters=max_iters,
            seed=seed,
            tokens_per_file=reshuffled_tokens_per_file,
            gpu_minibatch_size=gpu_minibatch_size,
            save_every=save_every,
            group=gpu_process_group,
        )
    )


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *args: Any,
    k: list[int] | None = None,
    expansion_factor: list[int] | None = None,
    max_iters: int = 128,
    save_every: int | None = None,
    seed: int = 0,
    gpu_minibatch_size: int = 100000,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
) -> None:
    if not k:
        k = None

    cluster_paths(
        model_name,
        dataset_name,
        *args,
        k=k,
        expansion_factor=expansion_factor,
        max_iters=max_iters,
        save_every=save_every,
        seed=seed,
        gpu_minibatch_size=gpu_minibatch_size,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        context_length=context_length,
        log_level=log_level,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    arguably.run()
