"""
Experiment to visualize router paths for all countries vs all others.

This script runs the pre_answer experiment for all countries,
computes the average route across prompts, and plots three heatmaps for each:
1. Target country mean route
2. Other countries mean route
3. Their difference

Usage:
    uv run python -m exp.capital_country_viz \\
        --model-name "olmoe-i"
"""

import gc
from itertools import batched
from pathlib import Path
import sys

import arguably
from loguru import logger
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor, get_postprocessor
from exp.capital_country import get_all_prompts
from viz import FIGURE_DIR


@th.no_grad()
def extract_pre_answer_paths(
    model: StandardizedTransformer,
    top_k: int,
    batch_size: int = 64,
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
) -> dict[str, th.Tensor]:
    """
    Extract only PRE_ANSWER router paths for all prompts in batches.
    More efficient than extracting all experiment types.

    Args:
        model: The MoE model
        top_k: Number of top experts
        batch_size: Number of prompts to process at once
        postprocessor: How to process router logits

    Returns:
        Dictionary mapping country -> path tensor (L, E)
    """
    prompts = get_all_prompts(model.tokenizer)

    postprocessor_fn = get_postprocessor(postprocessor)

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    assert all(p.token_info is not None for p in prompts), (
        "All prompts must have token info"
    )

    # Group prompts by country
    country_paths: dict[str, th.Tensor] = {}

    # Sort prompts by sequence length for more efficient batching
    sorted_prompts = sorted(prompts, key=lambda p: len(p.token_ids))

    batches = list(batched(sorted_prompts, batch_size))

    for batch_prompts in tqdm(
        batches, desc="Extracting PRE_ANSWER router paths", total=len(batches)
    ):
        # Get sequence lengths and find max length in batch
        seq_lengths = [len(p.token_ids) for p in batch_prompts]
        max_seq_len = max(seq_lengths)

        attn_mask = th.ones(
            (len(batch_prompts), max_seq_len),
            dtype=th.bool,
            device=batch_prompts[0].token_ids.device,
        )

        # Left pad sequences to max length and stack into batch
        padded_tokens = []
        for sample_idx, prompt in enumerate(batch_prompts):
            tokens = prompt.token_ids
            padding_amt = max_seq_len - len(tokens)

            if padding_amt > 0:
                padding = th.full(
                    (padding_amt,),
                    pad_token_id,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = th.cat([padding, tokens])
                attn_mask[sample_idx, :padding_amt] = False

            padded_tokens.append(tokens)

        # Stack into (B, T)
        batch_token_ids = th.stack(padded_tokens, dim=0)
        batch = {
            "input_ids": batch_token_ids,
            "attention_mask": attn_mask,
        }

        router_logits_list = []

        with model.trace(batch):
            for layer_idx in model.layers_with_routers:
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats
                if isinstance(router_output, tuple):
                    if len(router_output) == 2:
                        router_scores, _router_indices = router_output
                    else:
                        raise ValueError(
                            f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                        )
                else:
                    router_scores = router_output

                logits = router_scores.save()
                # Extract only the last token's logits (pre_answer is always the last token)
                # logits shape: (B, T, E), we want (B, E) for the last token
                last_token_logits = logits.reshape(*batch_token_ids.shape, -1)[
                    :, -1, :
                ]  # (B, E)
                router_logits_list.append(last_token_logits)

        # Stack into (B, L, E) - only last token for each prompt
        router_logits = th.stack(router_logits_list, dim=-2)  # (B, L, E)

        # Apply postprocessor to get paths
        router_paths_batch = postprocessor_fn(router_logits, top_k)  # (B, L, E)
        router_paths_batch = router_paths_batch.cpu()

        country_paths[prompt.country] = router_paths_batch

        # Clean up batch tensors
        del router_logits, router_paths_batch
        gc.collect()
        th.cuda.empty_cache()

    return country_paths


def compute_average_paths_pre_answer(
    country_paths: dict[str, th.Tensor],
) -> dict[str, th.Tensor]:
    """
    Compute average PRE_ANSWER path for each country.

    Returns:
        Dictionary mapping country -> average path tensor (L, E)
    """
    avg_paths: dict[str, th.Tensor] = {
        country: path.mean(dim=0) for country, path in country_paths.items()
    }

    return avg_paths


def plot_route_heatmaps(
    target_route: th.Tensor,  # (L, E)
    other_route: th.Tensor,  # (L, E)
    diff_route: th.Tensor,  # (L, E)
    target_country: str,
    output_country_path: Path,
) -> None:
    """
    Plot three heatmaps: target route, other route, and their difference.

    Args:
        target_route: Mean route for target country (L, E)
        other_route: Mean route for other countries (L, E)
        diff_route: Difference between target and other routes (L, E)
        target_country: Name of target country
        output_country_path: Path to save the country's figures
    """
    output_country_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 6))
    plt.imshow(
        target_route.cpu().numpy(),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    plt.title(f"{target_country} Mean Route", fontsize=14)
    plt.colorbar(label="Activation")
    plt.savefig(output_country_path / "target.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.imshow(
        other_route.cpu().numpy(),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    plt.title("Other Countries Mean Route", fontsize=14)
    plt.colorbar(label="Activation")
    plt.savefig(output_country_path / "other.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.imshow(
        diff_route.cpu().numpy(),
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    plt.title(f"{target_country} - Other Countries", fontsize=14)
    plt.colorbar(label="Difference")
    plt.savefig(output_country_path / "diff.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.tight_layout()

    logger.info(f"Saved heatmaps to {output_country_path}")


@arguably.command()
def capital_country_viz(
    *,
    model_name: str = "olmoe-i",
    model_dtype: str = "bf16",
    postprocessor: str = "masks",
    router_path_batch_size: int = 128,
    seed: int = 0,
    hf_token: str = "",
    output_dir: str = "out/capital_country_viz",
    log_level: str = "INFO",
) -> None:
    """
    Visualize router paths for all countries vs all others.

    Generates heatmaps for each country showing:
    1. Target country mean route
    2. Other countries mean route
    3. Their difference

    Args:
        model_name: Name of the model to use (olmoe-i, q3, gpt, etc.)
        model_dtype: Data type for model weights
        postprocessor: Router logits postprocessor (masks, identity, softmax, etc.)
        router_path_batch_size: Batch size for router path extraction
        seed: Random seed for reproducibility
        hf_token: Hugging Face API token
        output_dir: Directory to save results
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running capital_country_viz experiment with log level: {log_level}")

    # Set random seeds
    th.manual_seed(seed)

    # Parse postprocessor
    postprocessor_enum = RouterLogitsPostprocessor(postprocessor)
    logger.info(f"Postprocessor: {postprocessor_enum}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict()
    model_dtype_torch = get_dtype(model_dtype)

    logger.info(f"Loading model: {model_config.hf_name}")
    logger.info(f"Checkpoint: {model_ckpt}")

    # Load model
    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        torch_dtype=model_dtype_torch,
        token=hf_token,
    )

    logger.info("Model loaded successfully")
    logger.info(f"Number of layers with routers: {len(model.layers_with_routers)}")
    logger.info(f"Layers with routers: {model.layers_with_routers}")

    # Get model architecture info
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok

    logger.info(f"Number of experts: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Step 1: Extract PRE_ANSWER router paths for all prompts
    logger.info("=" * 80)
    logger.info("STEP 1: Extracting PRE_ANSWER router paths")
    logger.info("=" * 80)

    country_paths = extract_pre_answer_paths(
        model,
        top_k=top_k,
        batch_size=router_path_batch_size,
        postprocessor=postprocessor_enum,
    )

    # Step 2: Compute average paths
    logger.info("=" * 80)
    logger.info("STEP 2: Computing average paths")
    logger.info("=" * 80)

    avg_paths = compute_average_paths_pre_answer(country_paths)
    logger.info(f"Computed average paths for {len(avg_paths)} countries")

    # Step 3: Generate visualizations for all countries
    logger.info("=" * 80)
    logger.info("STEP 3: Generating visualizations for all countries")
    logger.info("=" * 80)

    all_countries = set(avg_paths.keys())
    logger.info(f"Generating heatmaps for {len(all_countries)} countries")

    for target_country, target_route in tqdm(
        avg_paths.items(), total=len(avg_paths), desc="Generating heatmaps"
    ):
        other_countries = set(all_countries) - {target_country}
        other_routes = th.stack(
            [avg_paths[country] for country in other_countries]
        )  # (N-1, L, E)
        other_route = other_routes.mean(dim=0)  # (L, E)
        diff_route = target_route - other_route  # (L, E)

        # Create output filename
        country_slug = target_country.lower().replace(" ", "_")
        output_country_path = Path(FIGURE_DIR) / "capital_country_viz" / country_slug

        plot_route_heatmaps(
            target_route=target_route,
            other_route=other_route,
            diff_route=diff_route,
            target_country=target_country,
            output_country_path=output_country_path,
        )

    # Print summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Generated heatmaps for {len(all_countries)} countries")
    logger.info(f"Figures saved to: {Path(FIGURE_DIR) / 'capital_country_viz'}")


if __name__ == "__main__":
    arguably.run()
