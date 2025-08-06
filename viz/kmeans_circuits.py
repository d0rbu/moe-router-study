import os

import arguably
import matplotlib

matplotlib.use("WebAgg")  # Use GTK3Agg backend for interactive plots on Pop!_OS
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

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


def visualize_top_circuits(circuit_mask: th.Tensor) -> None:
    circuit_mask = circuit_mask.cpu()
    num_centroids, num_layers, num_experts = circuit_mask.shape

    # matplotlib interactive mode that lets us scroll through the circuits
    # each circuit will be shown as a num_layers x num_experts matrix
    # where each row is a layer and each column is an expert
    # a cell in the grid will be empty when the mask is 0 and filled when the mask is 1

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize with the first circuit
    current_circuit = 0
    im = ax.imshow(circuit_mask[current_circuit], cmap="Blues", aspect="auto")
    ax.set_title(f"Circuit {current_circuit + 1}/{num_centroids}")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Layer Index")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    def on_scroll(event):
        nonlocal current_circuit
        if event.button == "up" and current_circuit < num_centroids - 1:
            current_circuit += 1
        elif event.button == "down" and current_circuit > 0:
            current_circuit -= 1

        # Update the image
        im.set_array(circuit_mask[current_circuit])
        ax.set_title(f"Circuit {current_circuit + 1}/{num_centroids}")
        fig.canvas.draw()

    # Connect the scroll event
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Add instructions
    plt.figtext(0.5, 0.02, "Use mouse scroll to navigate through circuits",
                ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.show()


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
    # top_circuits, circuit_mask = get_top_circuits(centroids, num_layers, top_k)
    circuit_mask = centroids > 0.5
    visualize_top_circuits(circuit_mask)


if __name__ == "__main__":
    arguably.run()
