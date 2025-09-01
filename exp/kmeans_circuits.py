import os

import arguably
from loguru import logger
import matplotlib

matplotlib.use("WebAgg")  # Use GTK3Agg backend for interactive plots on Pop!_OS
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

from exp import (
    get_experiment_dir,
    get_experiment_name,
    save_config,
    verify_config,
)
from exp.activations import load_activations_and_topk
from viz import FIGURE_DIR


def kmeans_manhattan(
    data: th.Tensor,
    k: int,
    minibatch_size: int,
    max_iters: int = 100,
    seed: int = 0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Perform k-means clustering with Manhattan distance.

    Args:
        data: Data to cluster, shape (N, D)
        k: Number of clusters
        max_iters: Maximum number of iterations
        minibatch_size: Batch size for processing data. If None, process all data at once.
        seed: Random seed for initialization

    Returns:
        centroids: Cluster centroids, shape (k, D)
        assignments: Cluster assignments, shape (N,)
        losses: Loss at each iteration, shape (num_iters,)
    """
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    assert data.ndim == 2, "Data must be of dimensions (B, D)"

    dataset_size, dim = data.shape

    assert minibatch_size > 0 and minibatch_size <= dataset_size, (
        "Batch size must be > 0 and <= dataset_size"
    )

    # initialize the centroids - use first batch to initialize
    first_batch = data[: min(minibatch_size, dataset_size)]
    centroids = first_batch[th.randperm(first_batch.size(0))[:k]]

    # Initialize cluster sizes for weighted updates
    cluster_sizes = th.zeros(k, device=data.device)

    # Track losses for each iteration
    losses = []

    # Track assignments for final return
    assignments = th.zeros(dataset_size, dtype=th.long, device=data.device)

    # run kmeans
    for _ in tqdm(range(max_iters), desc="Running kmeans", leave=False):
        last_centroids = centroids.clone()

        # Reset cluster sizes for this iteration
        cluster_sizes.zero_()

        # Create new centroids tensor filled with zeros
        new_centroids = th.zeros_like(centroids)

        # Track loss for this iteration
        current_loss = 0.0

        # Process data in batches
        num_batches = (dataset_size + minibatch_size - 1) // minibatch_size

        for batch_idx in tqdm(
            range(num_batches),
            desc="Processing minibatches",
            leave=False,
            total=num_batches,
        ):
            start_idx = batch_idx * minibatch_size
            end_idx = min((batch_idx + 1) * minibatch_size, dataset_size)
            batch_data = data[start_idx:end_idx]

            # assign each point to the nearest centroid
            distances = th.cdist(batch_data, centroids, p=1)
            batch_clusters = th.argmin(distances, dim=1)

            # Store assignments for this batch
            assignments[start_idx:end_idx] = batch_clusters

            # Calculate loss for this batch (sum of distances to assigned centroids)
            batch_loss = (
                th.gather(distances, 1, batch_clusters.unsqueeze(1)).sum().item()
            )
            current_loss += batch_loss

            # update the centroids and cluster sizes for this batch
            for i in range(k):
                batch_mask = batch_clusters == i
                batch_count = batch_mask.sum().item()

                if batch_count > 0:
                    # Accumulate sum of points in this cluster
                    new_centroids[i] += batch_data[batch_mask].sum(dim=0)
                    # Update cluster size
                    cluster_sizes[i] += batch_count

        # Compute final centroids by dividing by cluster sizes
        # Avoid division by zero
        for i in range(k):
            if cluster_sizes[i] > 0:
                centroids[i] = new_centroids[i] / cluster_sizes[i]

        # Record loss for this iteration
        losses.append(current_loss)

        # Check for convergence
        if th.allclose(centroids, last_centroids):
            break

    return centroids, assignments, th.tensor(losses)


def elbow(
    data: th.Tensor,
    minibatch_size: int,
    start: int = 32,
    stop: int = 1024,
    step: int = 32,
    seed: int = 0,
    experiment_name: str | None = None,
) -> None:
    assert data.ndim == 2, "Data must be of dimensions (B, D)"

    dataset_size, dim = data.shape

    total_iters = (stop - start) // step

    # run kmeans for each k
    sse_collection = []
    for k in tqdm(
        range(start, stop, step),
        desc="Running elbow method",
        leave=False,
        total=total_iters,
    ):
        centroids, assignments, _ = kmeans_manhattan(
            data, k, minibatch_size=minibatch_size, seed=seed
        )

        # compute the sum of squared manhattan distances in batches
        sse = 0.0

        # Process in batches to avoid OOM
        num_batches = (
            (dataset_size + minibatch_size - 1) // minibatch_size
            if minibatch_size > 0
            else 1
        )

        for batch_idx in tqdm(
            range(num_batches),
            desc="Processing minibatches",
            leave=False,
            total=num_batches,
        ):
            start_idx = batch_idx * minibatch_size
            end_idx = min((batch_idx + 1) * minibatch_size, dataset_size)
            batch_data = data[start_idx:end_idx]
            batch_assignments = assignments[start_idx:end_idx]

            batch_sse = (
                (batch_data - centroids[batch_assignments])
                .abs()
                .sum(dim=1)
                .pow(2)
                .sum()
            )
            sse += batch_sse.item()

        sse_collection.append(sse)

        del centroids, assignments
        th.cuda.empty_cache()

    sse = th.tensor(sse_collection).cpu()

    # Create experiment directory if name is provided
    if experiment_name:
        experiment_dir = get_experiment_dir(experiment_name)
        figure_path = os.path.join(experiment_dir, "elbow_method.png")
    else:
        figure_path = os.path.join(FIGURE_DIR, "elbow_method.png")

    # plot the sse
    plt.plot(range(start, stop, step), sse)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_circuits(
    centroids: th.Tensor, num_layers: int, top_k: int
) -> tuple[th.Tensor, th.Tensor]:
    num_centroids = centroids.shape[0]
    circuit_centroids = centroids.view(num_centroids, num_layers, -1)

    circuits = th.topk(circuit_centroids, k=top_k, dim=2)
    circuit_mask = th.zeros_like(circuit_centroids)
    circuit_mask.scatter_(2, circuits.indices, 1)

    return circuits.indices, circuit_mask


@arguably.command()
def cluster_circuits(
    model_name: str = "gpt",
    dataset_name: str = "lmsys",
    *_args,
    k: int | None = None,
    seed: int = 0,
    minibatch_size: int | None = None,
    name: str | None = None,
    source_experiment: str | None = None,
) -> None:
    """Perform k-means clustering on router activations to identify circuits.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        k: Number of clusters (if None, runs elbow method)
        seed: Random seed for initialization
        minibatch_size: Batch size for processing data
        name: Custom name for the experiment
        source_experiment: Name of the experiment to load activations from
    """
    # Generate experiment name if not provided
    if name is None:
        name = get_experiment_name(
            model_name=model_name,
            dataset_name=dataset_name,
            k=k,
            seed=seed,
        )

    # Create experiment directory
    experiment_dir = get_experiment_dir(name)

    # Create config dictionary
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "k": k,
        "seed": seed,
        "minibatch_size": minibatch_size,
        "source_experiment": source_experiment,
    }

    # Verify config against saved config
    verify_config(config, experiment_dir)

    # Save config
    save_config(config, experiment_dir)

    # Load activations
    activated_experts, top_k = load_activations_and_topk(
        experiment_name=source_experiment
    )

    # Flatten the activations for clustering
    num_tokens, num_layers, num_experts = activated_experts.shape
    flattened_activations = activated_experts.view(num_tokens, -1).float()

    # Set default minibatch size if not provided
    if minibatch_size is None:
        minibatch_size = min(1024, num_tokens)

    # Run elbow method if k is not provided
    if k is None:
        logger.info("Running elbow method to determine optimal k")
        elbow(
            flattened_activations,
            minibatch_size=minibatch_size,
            seed=seed,
            experiment_name=name,
        )
        return

    # Run k-means clustering
    logger.info(f"Running k-means clustering with k={k}")
    centroids, _, _ = kmeans_manhattan(
        flattened_activations, k=k, minibatch_size=minibatch_size, seed=seed
    )

    # Save the circuits
    output_path = os.path.join(experiment_dir, "kmeans_circuits.pt")
    th.save({"circuits": centroids, "top_k": top_k}, output_path)
    logger.info(f"Saved circuits to {output_path}")


if __name__ == "__main__":
    arguably.run()
