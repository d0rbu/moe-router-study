import os

import arguably
import matplotlib

matplotlib.use("WebAgg")  # Use GTK3Agg backend for interactive plots on Pop!_OS
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_topk
from viz import FIGURE_DIR


def kmeans_manhattan(
    data: th.Tensor,
    k: int,
    max_iters: int = 100,
    batch_size: int | None = None,
    seed: int = 0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Perform k-means clustering with Manhattan distance.

    Args:
        data: Data to cluster, shape (N, D)
        k: Number of clusters
        max_iters: Maximum number of iterations
        batch_size: Batch size for processing data. If None, process all data at once.
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

    if batch_size is None:
        batch_size = dataset_size
    else:
        assert batch_size > 0 and batch_size < dataset_size, (
            "Batch size must be > 0 and < dataset_size"
        )

    # initialize the centroids - use first batch to initialize
    first_batch = data[: min(batch_size, dataset_size)]
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
        num_batches = (dataset_size + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, dataset_size)
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
    batch_size: int = 0,
    start: int = 32,
    stop: int = 256,
    step: int = 32,
    seed: int = 0,
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
            data, k, batch_size=batch_size, seed=seed
        )

        # compute the sum of squared manhattan distances in batches
        sse = 0.0

        # Process in batches to avoid OOM
        num_batches = (
            (dataset_size + batch_size - 1) // batch_size if batch_size > 0 else 1
        )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, dataset_size)
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

    # plot the sse
    plt.plot(range(start, stop, step), sse)
    plt.savefig(
        os.path.join(FIGURE_DIR, "elbow_method.png"), dpi=300, bbox_inches="tight"
    )
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
def cluster_circuits(k: int | None = None, seed: int = 0, batch_size: int = 0) -> None:
    activated_experts, top_k = load_activations_and_topk()

    batch_size, num_layers, num_experts = activated_experts.shape

    # (B, L, E) -> (B, L * E)
    activated_experts = activated_experts.view(activated_experts.shape[0], -1).float()

    if k is None:
        elbow(activated_experts, batch_size=batch_size, seed=seed)
        return

    centroids, _, _ = kmeans_manhattan(
        activated_experts, k, batch_size=batch_size, seed=seed
    )

    # save circuits
    out = {
        "circuits": centroids,
        "top_k": top_k,
    }
    out_path = os.path.join(OUTPUT_DIR, "kmeans_circuits.pt")
    th.save(out, out_path)


if __name__ == "__main__":
    arguably.run()
