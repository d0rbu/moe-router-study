import asyncio
import os

import arguably
from loguru import logger
import matplotlib.pyplot as plt
import torch as th

from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from viz import FIGURE_DIR


async def _router_correlations_async(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 5_000,
    context_length: int = 2048,
    batch_size: int = 4096,
    reshuffled_tokens_per_file: int = 100000,
    num_workers: int = 8,
    debug: bool = False,
) -> None:
    """Async implementation of router correlation analysis."""
    logger.info(f"Loading activations for model: {model_name}, dataset: {dataset_name}")

    # Input validation
    assert model_name, "Model name cannot be empty"
    assert dataset_name, "Dataset name cannot be empty"
    assert tokens_per_file > 0, (
        f"Tokens per file must be positive, got {tokens_per_file}"
    )
    assert context_length > 0, f"Context length must be positive, got {context_length}"
    assert batch_size > 0, f"Batch size must be positive, got {batch_size}"

    logger.debug("Loading activations and initializing distributed...")
    (
        activations,
        _activation_dims,
    ) = await load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        submodule_names=[ActivationKeys.ROUTER_LOGITS],
        context_length=context_length,
        num_workers=num_workers,
        debug=debug,
    )

    activated_experts_collection = []
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None

    # Iterate through activation batches
    for batch in activations(batch_size=batch_size):
        router_logits = batch[ActivationKeys.ROUTER_LOGITS]

        if top_k is None:
            top_k = batch["topk"]
            num_layers, num_experts = router_logits.shape[1], router_logits.shape[2]
            total_experts = num_layers * num_experts
            logger.info(
                f"Router configuration: {num_layers} layers, {num_experts} experts per layer, top-k={top_k}"
            )

        # Apply logits postprocessor (default: convert to masks)
        from core.moe import convert_router_logits_to_paths

        logits_postprocessor = (
            convert_router_logits_to_paths  # Can be made configurable later
        )
        activated_experts = logits_postprocessor(router_logits, top_k)

        # (B, L, E) -> (L * E, B)
        activated_experts_collection.append(
            activated_experts.reshape(-1, total_experts).T
        )

    if top_k is None or num_layers is None or num_experts is None:
        raise ValueError("No activation data found")

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


@arguably.command
def router_correlations(
    *,
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    tokens_per_file: int = 5_000,
    context_length: int = 2048,
    batch_size: int = 4096,
    reshuffled_tokens_per_file: int = 100000,
    num_workers: int = 8,
    debug: bool = False,
) -> None:
    """Generate router correlation plots for an experiment.

    This script:
    1. Loads router activations using load_activations_and_init_dist
    2. Computes correlations between expert activations across layers
    3. Generates correlation matrices and visualizations
    4. Analyzes cross-layer correlations

    Args:
        model_name: Name of the model (e.g., "olmoe-i").
        dataset_name: Name of the dataset (e.g., "lmsys").
        tokens_per_file: Number of tokens per activation file.
        context_length: Context length used during activation collection.
        batch_size: Number of samples to process per batch (default: 4096).
        reshuffled_tokens_per_file: Number of tokens per reshuffled file (default: 100000).
        num_workers: Number of worker processes for data loading (default: 8).
        debug: Enable debug logging (default: False).
    """
    asyncio.run(
        _router_correlations_async(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            context_length=context_length,
            batch_size=batch_size,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            num_workers=num_workers,
            debug=debug,
        )
    )


if __name__ == "__main__":
    arguably.run()
