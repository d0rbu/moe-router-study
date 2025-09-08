import asyncio
from dataclasses import dataclass
import gc
from itertools import batched, islice, pairwise
import os

import arguably
from loguru import logger
import torch as th
import torch.distributed as dist
from tqdm import tqdm

from exp import OUTPUT_DIR
from exp.activations import Activations, load_activations_and_init_dist
from exp.get_activations import ActivationKeys, get_experiment_name


@dataclass
class RunningKMeansData:
    # list of centroids of shape (K, D)
    centroid_sets: list[th.tensor]
    # list of weights of shape (K) for online running updates
    weight_sets: list[th.tensor]
    losses: th.tensor

    def to(self, device: th.Device) -> "RunningKMeansData":
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
            loss=th.empty_like(self.losses),
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
                self.weights,
                other.centroid_sets,
                other.weights,
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


async def update_centroid_data(
    data: th.Tensor,
    centroids: th.Tensor,
    weights: th.Tensor,
    losses: th.Tensor,
    losses_idx: int,
) -> None:
    ### do a step of kmeans on this data batch and get loss
    distances = th.cdist(data, centroids, p=1)
    assignments = th.argmin(distances, dim=1)

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

    new_centroids, new_weights = list(zip(*centroids_and_weights, strict=True))
    ### end

    ### update the tensors
    old_weights = weights.clone()
    weights += weights_delta
    old_weights_proportion = old_weights / weights
    new_weights_proportion = 1 - old_weights_proportion

    centroids *= old_weights_proportion
    centroids += new_centroids * new_weights_proportion

    old_loss_proportion = old_weights.sum() / weights.sum()
    new_loss_proportion = 1 - old_loss_proportion

    losses[losses_idx] *= old_loss_proportion
    losses[losses_idx] += new_loss * new_loss_proportion
    ### end


async def sync_gpu_data():
    pass


async def gpu_worker(
    gpu_idx: int, gpu_data: GPUData, top_k: int, losses_over_time: list[th.Tensor]
) -> None:
    logger.info(f"Starting GPU worker {gpu_idx}")

    while True:
        data: th.tensor = await gpu_data.gpu_queue.get()
        if data is None:
            break

        data = data.to(gpu_idx)

        # convert from logits to paths
        paths = th.topk(data, k=top_k, dim=2).indices
        data.zero_()
        data.scatter_(2, paths.indices, 1)

        del paths
        th.cuda.empty_cache()

        await asyncio.gather(
            *[
                update_centroid_data(
                    data,
                    centroids,
                    weights,
                    gpu_data.dirty_losses,
                    centroid_set_idx,
                )
                for centroid_set_idx, (centroids, weights) in enumerate(
                    zip(
                        gpu_data.dirty_centroid_sets,
                        gpu_data.dirty_weights,
                        strict=True,
                    )
                )
            ]
        )


async def sync_gpu_data(
    all_gpu_data: list[GPUData],
) -> None:
    pass


GPU_QUEUE_MAXSIZE = 4


async def kmeans_manhattan(
    activations: Activations,
    activation_dim: int,
    k_values: list[int],
    effective_batch_size: int | None = None,
    max_iters: int = 128,
    gpu_minibatch_size: int | None = None,
    seed: int = 0,
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

    Returns:
        centroid_sets: List of cluster centroids, each element of shape (K, D)
        top_k: Topk value of the model used to generate the activations
        losses: Losses for each iteration, shape (num_K, T)
    """
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    num_gpus = th.cuda.device_count()
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
                weights=[th.zeros(k) for k in k_values],
                losses=th.zeros(len(k_values)),
            ),
            dirty_data=RunningKMeansData(
                centroid_sets=[
                    th.zeros(k, activation_dim, device=gpu_idx) for k in k_values
                ],
                weights=[th.zeros(k) for k in k_values],
                losses=th.zeros(len(k_values)),
            ),
            queue=asyncio.Queue(maxsize=GPU_QUEUE_MAXSIZE),
        )
        for gpu_idx in num_gpus
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

    iterator = range(max_iters)
    if dist.get_rank() == 0:
        iterator = tqdm(
            iterator, desc="Kmeans iterations", leave=False, total=max_iters, position=0
        )

    _workers = [
        asyncio.create_task(gpu_worker(gpu_idx, all_gpu_data, top_k, losses_over_time))
        for gpu_idx in range(num_gpus)
    ]

    # distributed kmeans
    for _iter_idx in iterator:
        # process data in batches, parallelized over devices and nodes
        minibatch_iterator = activations(batch_size=gpu_minibatch_size)

        distributed_iterator = islice(
            minibatch_iterator,
            start=dist.get_rank(),
            stop=None,
            step=dist.get_world_size(),
        )
        num_local_minibatches = len(
            range(
                start=dist.get_rank(),
                stop=num_gpu_minibatches,
                step=dist.get_world_size(),
            )
        )
        distributed_iterator = tqdm(
            distributed_iterator,
            desc=f"Rank {dist.get_rank()}",
            total=num_local_minibatches,
            leave=False,
            position=dist.get_rank() + 1,
        )

        concurrent_minibatch_iterator = batched(distributed_iterator, num_gpus)

        for gpu_minibatch_idx, gpu_minibatches in enumerate(
            concurrent_minibatch_iterator
        ):
            for gpu_data, gpu_minibatch in zip(
                all_gpu_data, gpu_minibatches, strict=False
            ):
                gpu_data.queue.put(
                    (gpu_minibatch_idx, gpu_minibatch[ActivationKeys.ROUTER_LOGITS])
                )

        # gather across nodes
        all_centroid_sets = [
            [
                th.empty_like(centroids)
                .unsqueeze(0)
                .repeat(dist.get_world_size(), 1, 1)
                for centroids in gpu_data.centroid_sets
            ]
            for gpu_data in all_gpu_data
        ]
        all_weights = [
            [
                th.empty_like(weights).unsqueeze(0).repeat(dist.get_world_size(), 1)
                for weights in gpu_data.weights
            ]
            for gpu_data in all_gpu_data
        ]
        all_losses = [
            th.empty_like(gpu_data.losses).unsqueeze(0).repeat(dist.get_world_size(), 1)
            for gpu_data in all_gpu_data
        ]

        for (
            gpu_data,
            centroid_setses,
            weights_sets,
            losses_sets,
        ) in zip(all_gpu_data, all_centroid_sets, all_weights, all_losses, strict=True):
            # N, num_K
            dist.all_gather(losses_sets, gpu_data.losses)

            for losses_idx, (
                centroids,
                weights,
                centroid_dist_set,
                weights_dist_set,
            ) in enumerate(
                zip(
                    gpu_data.centroid_sets,
                    gpu_data.weights,
                    centroid_setses,
                    weights_sets,
                    strict=True,
                )
            ):
                # (N, K, D)
                dist.all_gather_into_tensor(centroid_dist_set, centroids)
                # (N, K)
                dist.all_gather_into_tensor(weights_dist_set, weights)

                weights_total = weights_dist_set.sum(dim=0)
                weights_proportion = weights_dist_set / weights_total

                new_centroids = (
                    centroid_dist_set * weights_proportion.unsqueeze(-1)
                ).sum(dim=0)

                loss_proportion = weights_dist_set.sum(dim=1) / weights_total.sum()

                new_loss = (losses_sets[:, losses_idx] * loss_proportion).sum()

                gpu_data.centroid_sets[losses_idx] = new_centroids
                gpu_data.weights[losses_idx] = weights_total
                gpu_data.losses[losses_idx] = new_loss

        # now do an all-gather among entries in gpu_data
        # we start by just gathering to cpu
        base_data = GPUData(
            centroid_sets=[
                centroids.cpu() for centroids in all_gpu_data[0].centroid_sets
            ],
            weights=[weights.cpu() for weights in all_gpu_data[0].weights],
            losses=all_gpu_data[0].losses,
        )
        for gpu_data in all_gpu_data[1]:
            for losses_idx, (
                centroids,
                weights,
                new_centroids,
                weights_delta,
                new_loss,
            ) in enumerate(
                zip(
                    base_data.centroid_sets,
                    base_data.weights,
                    gpu_data.centroid_sets,
                    gpu_data.weights,
                    gpu_data.losses,
                    strict=True,
                )
            ):
                new_weights = weights + weights_delta.cpu()
                old_weights_proportion = weights / new_weights
                new_weights_proportion = 1 - old_weights_proportion

                centroids *= old_weights_proportion
                centroids += new_centroids.cpu() * new_weights_proportion

                old_loss_proportion = weights.sum() / new_weights.sum()
                new_loss_proportion = 1 - old_loss_proportion

                base_data.losses[losses_idx] *= old_loss_proportion
                base_data.losses[losses_idx] += new_loss * new_loss_proportion

        # now we broadcast to all gpus
        for gpu_data in all_gpu_data:
            for centroids, weights, new_centroids, new_weights in enumerate(
                zip(
                    gpu_data.centroid_sets,
                    gpu_data.weights,
                    base_data.centroid_sets,
                    base_data.weights,
                    strict=True,
                )
            ):
                centroids.copy_(new_centroids)
                weights.copy_(new_weights)

            gpu_data.losses.copy_(base_data.losses)

    for gpu_data in all_gpu_data:
        gpu_data.put(None)
        gpu_data.queue.join()

    losses = th.stack(losses_over_time, dim=1)

    return all_gpu_data[0].centroid_sets, top_k, losses


def get_top_circuits(
    centroids: th.Tensor, num_layers: int, top_k: int
) -> tuple[th.Tensor, th.Tensor]:
    num_centroids = centroids.shape[0]
    circuit_centroids = centroids.view(num_centroids, num_layers, -1)

    circuits = th.topk(circuit_centroids, k=top_k, dim=2)
    circuit_mask = th.zeros_like(circuit_centroids)
    circuit_mask.scatter_(2, circuits.indices, 1)

    return circuits.indices, circuit_mask


async def cluster_paths(
    model_name: str,
    dataset_name: str,
    activations: Activations,
    activation_dim: int,
    k: list[int],
    max_iters: int,
    seed: int,
    tokens_per_file: int,
    gpu_minibatch_size: int,
) -> None:
    kmeans_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        tokens_per_file=tokens_per_file,
    )

    centroids, top_k, losses = kmeans_manhattan(
        activations=activations,
        activation_dim=activation_dim,
        k_values=k,
        max_iters=max_iters,
        gpu_minibatch_size=gpu_minibatch_size,
        seed=seed,
    )

    # save centroids
    out = {
        "centroids": centroids,
        "top_k": top_k,
    }
    out_path = os.path.join(OUTPUT_DIR, kmeans_experiment_name, "kmeans.pt")
    th.save(out, out_path)


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    k: list[int] | int | None = None,
    expansion_factor: list[int] | int | None = None,
    max_iters: int = 128,
    seed: int = 0,
    gpu_minibatch_size: int = 1024,
    tokens_per_file: int = 10_000,
) -> None:
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
        case None, int():
            k = [expansion_factor * activation_dim]
        case int(), None:
            k = [k]
        case None, list():
            k = [
                current_expansion_factor * activation_dim
                for current_expansion_factor in expansion_factor
            ]
        case list(), None:
            pass
        case _, _:
            raise ValueError("Cannot specify both k and expansion_factor")

    asyncio.run(
        cluster_paths(
            model_name=model_name,
            dataset_name=dataset_name,
            activations=activations,
            activation_dim=activation_dim,
            k=k,
            max_iters=max_iters,
            seed=seed,
            tokens_per_file=tokens_per_file,
            gpu_minibatch_size=gpu_minibatch_size,
        )
    )


if __name__ == "__main__":
    arguably.run()
