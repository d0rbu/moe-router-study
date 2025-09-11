from itertools import count
import os

import arguably
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

from exp import OUTPUT_DIR
from viz import FIGURE_DIR

# Use hardcoded directory name since ROUTER_LOGITS_DIRNAME was removed
ROUTER_LOGITS_DIRNAME = "activations"


@arguably.command()
def router_correlations(experiment_name: str) -> None:
    """Generate router correlation plots for an experiment."""
    activated_experts_collection = []
    top_k: int | None = None

    for file_idx in tqdm(count(), desc="Loading router logits"):
        file_path = os.path.join(
            OUTPUT_DIR, experiment_name, ROUTER_LOGITS_DIRNAME, f"{file_idx}.pt"
        )
        if not os.path.exists(file_path):
            break

        output = th.load(file_path)
        top_k = output["topk"]
        router_logits = output["router_logits"]

        num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
        total_experts = num_layers * num_experts
        top_k_indices = th.topk(router_logits, k=top_k, dim=2).indices
        activated_experts = th.zeros_like(router_logits)
        activated_experts.scatter_(2, top_k_indices, 1)

        # (B, L, E) -> (L * E, B)
        activated_experts_collection.append(
            activated_experts.reshape(-1, total_experts).T
        )

    if top_k is None:
        raise ValueError("No data files found")

    # (L * E, B)
    activated_experts = th.cat(activated_experts_collection, dim=-1)
    batch_size = activated_experts.shape[-1]

    # Build control by shuffling along the batch dimension per-layer to preserve within-layer statistics
    # Reconstruct to (B, L, E)
    activated_experts_ble = activated_experts.T.view(
        batch_size, num_layers, num_experts
    )

    # Initialize with a copy and shuffle batch indices independently for each layer
    random_activated_experts_ble = activated_experts_ble.clone()
    for layer_idx in range(num_layers):
        perm = th.randperm(batch_size, device=activated_experts.device)
        random_activated_experts_ble[:, layer_idx, :] = activated_experts_ble[
            perm, layer_idx, :
        ]

    # (B, L, E) -> (L * E, B)
    random_activated_experts = random_activated_experts_ble.reshape(batch_size, -1).T

    # (L * E, L * E)
    correlation = th.corrcoef(activated_experts)
    random_correlation = th.corrcoef(random_activated_experts)

    # only consider the upper triangle since it's hermitian and we don't want to consider an expert's correlation with itself
    upper_triangular_mask = th.triu(th.ones_like(correlation).bool(), diagonal=1).view(
        -1
    )

    # get the most correlated experts
    # (L * E * L * E), (L * E * L * E)
    correlations_raw, indices_raw = th.sort(correlation.view(-1))
    random_correlations_raw, random_indices_raw = th.sort(random_correlation.view(-1))

    # filter for upper triangle
    sorted_upper_triangular_mask = upper_triangular_mask[indices_raw]

    # (((L * E) * (L * E - 1)) // 2,)
    indices = indices_raw[sorted_upper_triangular_mask]
    correlations = correlations_raw[sorted_upper_triangular_mask]
    random_indices = random_indices_raw[sorted_upper_triangular_mask]
    random_correlations = random_correlations_raw[sorted_upper_triangular_mask]

    # set default fig size
    plt.rcParams["figure.figsize"] = (16, 12)

    # plot a bar chart of the correlations
    print("Plotting bar chart...")
    plt.bar(range(len(correlations)), correlations)
    plt.savefig(os.path.join(FIGURE_DIR, "router_correlations.png"))
    plt.close()

    # plot a bar chart of the random correlations
    print("Plotting random bar chart...")
    plt.bar(range(len(random_correlations)), random_correlations)
    plt.savefig(os.path.join(FIGURE_DIR, "router_correlations_random.png"))
    plt.close()

    first_layer_indices = (indices // total_experts) // num_experts
    second_layer_indices = (indices % total_experts) // num_experts
    first_expert_indices = (indices % (num_experts * total_experts)) // total_experts
    second_expert_indices = indices % num_experts
    rolled_indices = th.stack(
        [
            first_layer_indices,
            second_layer_indices,
            first_expert_indices,
            second_expert_indices,
        ],
        dim=1,
    )

    first_layer_random_indices = (random_indices // total_experts) // num_experts
    second_layer_random_indices = (random_indices % total_experts) // num_experts

    # now we want to filter out within-layer correlations
    within_layer_mask = first_layer_indices == second_layer_indices
    within_layer_random_mask = first_layer_random_indices == second_layer_random_indices

    cross_layer_rolled_indices = rolled_indices[~within_layer_mask]
    cross_layer_correlations = correlations[~within_layer_mask]
    cross_layer_random_correlations = random_correlations[~within_layer_random_mask]

    # plot a bar chart of the cross-layer correlations
    plt.bar(range(len(cross_layer_correlations)), cross_layer_correlations)
    plt.savefig(os.path.join(FIGURE_DIR, "router_correlations_cross_layer.png"))
    plt.close()

    # plot a bar chart of the cross-layer random correlations
    plt.bar(
        range(len(cross_layer_random_correlations)), cross_layer_random_correlations
    )
    plt.savefig(os.path.join(FIGURE_DIR, "router_correlations_cross_layer_random.png"))
    plt.close()

    # print the top 10 correlations
    print("Top 10 cross-layer correlations:")
    for i in range(10):
        first_layer_idx, second_layer_idx, first_expert_idx, second_expert_idx = (
            cross_layer_rolled_indices[-i - 1]
        )
        correlation = cross_layer_correlations[-i - 1]
        print(
            f"layer {first_layer_idx} expert {first_expert_idx} -> layer {second_layer_idx} expert {second_expert_idx}: {correlation}"
        )
    print()

    # print the bottom 10 correlations
    print("Bottom 10 cross-layer correlations:")
    for i in range(10):
        first_layer_idx, second_layer_idx, first_expert_idx, second_expert_idx = (
            cross_layer_rolled_indices[i]
        )
        correlation = cross_layer_correlations[i]
        print(
            f"layer {first_layer_idx} expert {first_expert_idx} -> layer {second_layer_idx} expert {second_expert_idx}: {correlation}"
        )
    print()


if __name__ == "__main__":
    arguably.run()
