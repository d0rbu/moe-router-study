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
    data: th.Tensor, k: int, max_iters: int = 1_000, seed: int = 0
) -> th.Tensor:
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    assert data.ndim == 2, "Data must be of dimensions (B, D)"

    batch_size, dim = data.shape

    # initialize the centroids
    centroids = data[th.randperm(batch_size)[:k]]

    # run kmeans
    for _ in tqdm(range(max_iters), desc="Running kmeans", leave=False):
        # assign each point to the nearest centroid
        distances = th.cdist(data, centroids, p=1)
        clusters = th.argmin(distances, dim=1)

        last_centroids = centroids.clone()

        # update the centroids
        for i in range(k):
            centroids[i] = data[clusters == i].mean(dim=0)

        centroid_delta = th.norm(centroids - last_centroids, p=2)
        if th.allclose(centroids, last_centroids):
            break

    return centroids


def elbow(data: th.Tensor, start: int = 32, stop: int = 1024, step: int = 32, seed: int = 0) -> None:
    assert data.ndim == 2, "Data must be of dimensions (B, D)"

    batch_size, dim = data.shape

    total_iters = (stop - start) // step

    # run kmeans for each k
    sse_collection = []
    for k in tqdm(range(start, stop, step), desc="Running elbow method", leave=False, total=total_iters):
        centroids = kmeans_manhattan(data, k, seed=seed)

        # compute the sum of squared manhattan distances
        distances = th.cdist(data, centroids, p=1)
        clusters = th.argmin(distances, dim=1)
        sse = (data - centroids[clusters]).abs().sum(dim=1).pow(2).sum()
        sse_collection.append(sse)

    sse = th.stack(sse_collection).cpu()

    # plot the sse
    plt.plot(range(start, stop, step), sse)
    plt.savefig(os.path.join(FIGURE_DIR, "elbow_method.png"), dpi=300, bbox_inches="tight")
    plt.close()


def get_top_circuits(centroids: th.Tensor, num_layers: int, top_k: int) -> tuple[th.Tensor, th.Tensor]:
    num_centroids = centroids.shape[0]
    circuit_centroids = centroids.view(num_centroids, num_layers, -1)

    circuits = th.topk(circuit_centroids, k=top_k, dim=2)
    circuit_mask = th.zeros_like(circuit_centroids)
    circuit_mask.scatter_(2, circuits.indices, 1)

    return circuits.indices, circuit_mask


@arguably.command()
def cluster_circuits(k: int | None = None, seed: int = 0) -> None:
    activated_experts, top_k = load_activations_and_topk()

    batch_size, num_layers, num_experts = activated_experts.shape

    # (B, L, E) -> (B, L * E)
    activated_experts = (
        activated_experts.view(activated_experts.shape[0], -1).float().cuda()
    )

    if k is None:
        elbow(activated_experts, seed=seed)
        return

    centroids = kmeans_manhattan(activated_experts, k, seed=seed)

    # save circuits
    out = {
        "circuits": centroids,
        "top_k": top_k,
    }
    out_path = os.path.join(OUTPUT_DIR, "kmeans_circuits.pt")
    th.save(out, out_path)


if __name__ == "__main__":
    arguably.run()
