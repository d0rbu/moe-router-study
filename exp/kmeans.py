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

from exp import OUTPUT_DIR
from exp.activations import Activations, load_activations_and_init_dist
from exp.get_activations import ActivationKeys
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
                strict=False,
            )
        ):
            new_weights.copy_(base_weights + other_weights)
            base_weight_proportion = base_weights / new_weights
            other_weight_proportion = 1 - base_weight_proportion

            new_centroids.copy_(
                base_weight_proportion * base_centroids
                + other_weight_proportion * other_centroids
            )

            base_loss_proportion = base_weights.sum() / new_weights.sum()
            other_loss_proportion = 1 - base_loss_proportion

            new_data.losses[losses_idx] = (
                base_loss_proportion * self.losses
                + other_loss_proportion * other.losses
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

    return data[centroid_mask].mean(dim=0), num_assigned


async def kmeans_step(
    data: th.Tensor,  # (B, L * E)
    centroids: th.Tensor,  # (K, L * E)
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    # (B, K)
    distances = th.cdist(data, centroids, p=1)
    # (B)
    assignments = th.argmin(distances, dim=1)

    # for calculating loss, we get the distances from each data point to the closest centroid
    centroid_distances_awaitable = asyncio.to_thread(
        th.gather, distances, 1, assignments.unsqueeze(1)
    )

    centroid_awaitables = [
        asyncio.to_thread(compute_centroid_from_assignment, data, assignments, i)
        for i in range(centroids.shape[0])
    ]
    centroids_and_weights_awaitable = asyncio.gather(*centroid_awaitables)

    centroid_distances, centroids_and_weights = await asyncio.gather(
        centroid_distances_awaitable, centroids_and_weights_awaitable
    )
    new_loss = centroid_distances.mean()

    new_centroids, new_weights = zip(*centroids_and_weights, strict=True)

    return new_centroids, new_weights, new_loss


async def sync(
    gpu_idx: int,
    all_gpu_data: list[GPUData],
    losses_over_time: list[th.Tensor],
    barrier: Barrier,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpu_data = all_gpu_data[gpu_idx]

    # gather across nodes
    all_losses = (
        th.empty_like(gpu_data.dirty_data.losses).unsqueeze(0).repeat(world_size, 1)
    )

    # N, num_K
    dist.all_gather_into_tensor(all_losses, gpu_data.dirty_data.losses)

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

        dist.all_gather_into_tensor(all_centroids, centroids)
        dist.all_gather_into_tensor(all_weights, weights)

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
    gpu_data.synced_data += sum(
        current_gpu_data.dirty_data.to(gpu_idx) for current_gpu_data in all_gpu_data
    )

    await barrier.wait()

    # reset dirty data now that it has been synced
    for weights in gpu_data.dirty_data.weight_sets:
        weights.zero_()

    if rank == 0 and gpu_idx == 0:
        losses_over_time.append(gpu_data.synced_data.losses.detach().cpu().clone())


async def gpu_worker(
    gpu_idx: int,
    all_gpu_data: list[GPUData],
    top_k: int,
    losses_over_time: list[th.Tensor],
    barrier: Barrier,
    save_dir: str | None = None,
) -> None:
    logger.info(f"Starting GPU worker {gpu_idx}")
    gpu_data = all_gpu_data[gpu_idx]

    while True:
        queue_item = await gpu_data.queue.get()
        if queue_item is None:
            break

        data, should_sync, save_idx = queue_item

        # assert that if save_idx is not None, then should_sync is also true
        if save_idx is not None:
            assert should_sync, "save_idx can only be set when should_sync is true"

        # (B, L, E)
        data = data.to(gpu_idx)

        # convert from logits to paths
        paths = th.topk(data, k=top_k, dim=2).indices
        data.zero_()
        data.scatter_(2, paths, 1)

        # (B, L, E) -> (B, L * E)
        flat_data = data.view(data.shape[0], -1)

        del paths, data
        th.cuda.empty_cache()

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
        gpu_data.dirty_data += RunningKMeansData(
            centroid_sets=new_centroid_sets,
            weight_sets=new_weight_sets,
            losses=new_losses,
        )

        if not should_sync:
            continue

        await sync(gpu_idx, all_gpu_data, losses_over_time, barrier)

        # save checkpoint if save_idx is not None and we're on rank 0 gpu 0
        if (
            save_idx is not None
            and dist.get_rank() == 0
            and gpu_idx == 0
            and save_dir is not None
        ):
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
        logger.warning(
            f"Per-GPU batch size {batch_size} is not divisible by GPU minibatch size {gpu_minibatch_size}; {leftover_minibatch_size} left over"
        )
        batch_size -= leftover_minibatch_size

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

    all_gpu_data = [
        GPUData(
            synced_data=RunningKMeansData(
                centroid_sets=[
                    th.empty(k, activation_dim, device=gpu_idx) for k in k_values
                ],
                weight_sets=[th.zeros(k) for k in k_values],
                losses=th.zeros(len(k_values)),
            ),
            dirty_data=RunningKMeansData(
                centroid_sets=[
                    th.empty(k, activation_dim, device=gpu_idx) for k in k_values
                ],
                weight_sets=[th.zeros(k) for k in k_values],
                losses=th.zeros(len(k_values)),
            ),
            queue=asyncio.Queue(maxsize=GPU_QUEUE_MAXSIZE),
        )
        for gpu_idx in range(num_gpus)
    ]

    ### get top_k and initialize centroids from random data points
    k_ranges = th.cat(
        [
            th.tensor([0], device=0),
            th.cumsum(th.tensor(k_values, device=0), dim=0),
        ],
        dim=0,
    )

    # load a batch of activations to initialize the centroids
    data_iterable = activations(batch_size=k_ranges[-1].item())
    activation_batch = next(data_iterable)
    router_activations = activation_batch[ActivationKeys.ROUTER_LOGITS]
    top_k = activation_batch["topk"]

    for k_idx, (k_start, k_end) in enumerate(pairwise(k_ranges)):
        k = k_values[k_idx]
        assert k_end - k_start == k, "k_end - k_start must be equal to k"

        for gpu_idx, gpu_data in enumerate(all_gpu_data):
            gpu_data.dirty_data.centroid_sets[k_idx] = router_activations[
                k_start:k_end
            ].to(gpu_idx)

    # clean up the background workers and queue
    data_iterable.send("STOP!")
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
        # ensure the save directory exists
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok=True)

    iterator = range(max_iters)
    if dist.get_rank() == 0:
        iterator = tqdm(
            iterator, desc="Kmeans iterations", leave=False, total=max_iters, position=0
        )

    synchronization_barrier = Barrier(num_gpus)
    _workers = [
        asyncio.create_task(
            gpu_worker(
                gpu_idx,
                all_gpu_data,
                top_k,
                losses_over_time,
                synchronization_barrier,
                save_dir,
            )
        )
        for gpu_idx in range(num_gpus)
    ]

    # distributed kmeans
    for _iter_idx in iterator:
        # process data in batches, parallelized over devices and nodes
        minibatch_iterator = activations(batch_size=gpu_minibatch_size)

        distributed_iterator = islice(
            minibatch_iterator,
            start=rank,
            stop=None,
            step=num_nodes,
        )
        num_local_minibatches = len(
            range(
                start=rank,
                stop=num_gpu_minibatches,
                step=num_nodes,
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

        for distributed_batch_idx, gpu_minibatches in enumerate(
            concurrent_minibatch_iterator
        ):
            should_sync = distributed_batch_idx % accumulation_size == (
                accumulation_size - 1
            )

            # compute effective step index and determine if we should save
            effective_step_idx = distributed_batch_idx // accumulation_size
            save_idx = effective_step_idx if effective_step_idx in save_steps else None

            for gpu_data, gpu_minibatch in zip(
                all_gpu_data, gpu_minibatches, strict=False
            ):
                await gpu_data.queue.put(
                    (
                        gpu_minibatch[ActivationKeys.ROUTER_LOGITS],
                        should_sync,
                        save_idx,
                    )
                )

    for gpu_data in all_gpu_data:
        await gpu_data.queue.put((None, False))

    losses = th.stack(losses_over_time, dim=1)

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
) -> None:
    kmeans_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        tokens_per_file=tokens_per_file,
    )

    save_dir = (
        os.path.join(OUTPUT_DIR, kmeans_experiment_name)
        if save_every is not None
        else None
    )

    centroids, top_k, losses = await kmeans_manhattan(
        activations=activations,
        activation_dim=activation_dim,
        k_values=k,
        max_iters=max_iters,
        gpu_minibatch_size=gpu_minibatch_size,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
    )

    if dist.get_rank() == 0:
        logger.info("saving...")

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
    gpu_minibatch_size: int = 1024,
    tokens_per_file: int = 10_000,
    log_level: str = "INFO",
) -> None:
    print(f"Running with log level: {log_level}")

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.debug(f"Running with log level: {log_level}")

    activations, activation_dims = load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        submodule_names=[ActivationKeys.ROUTER_LOGITS],
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
            tokens_per_file=tokens_per_file,
            gpu_minibatch_size=gpu_minibatch_size,
            save_every=save_every,
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
    gpu_minibatch_size: int = 1024,
    tokens_per_file: int = 10_000,
    log_level: str = "INFO",
    **kwargs: Any,
) -> None:
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
        log_level=log_level,
        **kwargs,
    )


if __name__ == "__main__":
    arguably.run()
