"""
Experiment to visualize expert selection differences before and after intervention.

This experiment:
1. Computes the mean difference (country-specific path) for a chosen country
2. Runs interventions with different alpha values
3. Captures binary masks of top-k chosen experts before and after intervention
4. Saves the masks for visualization

Usage:
    uv run python -m exp.capital_country_expert_diff \\
        --model-name "olmoe-i" \\
        --target-country "South Korea" \\
        --alpha 1.0
"""

from dataclasses import dataclass
import gc
from pathlib import Path
import sys
from typing import cast

import arguably
from loguru import logger
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import torch as th
import torch.nn.functional as F
from tqdm import tqdm

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor, convert_router_logits_to_paths
from exp.capital_country import (
    COUNTRY_TO_CAPITAL,
    CountryPrompt,
    ExperimentType,
    compute_average_paths,
    compute_country_specific_paths,
    extract_router_paths,
    get_all_prompts,
)
from viz import FIGURE_DIR


@dataclass(frozen=True)
class ExpertMasks:
    """Binary masks of chosen experts before and after intervention."""

    pre_intervention: th.Tensor  # (L, E) - binary mask before intervention
    post_intervention: th.Tensor  # (L, E) - binary mask after intervention
    forgetfulness: float  # (pre - post) / pre, normalized forgetfulness


@th.no_grad()
def extract_expert_masks_with_intervention(
    prompts: list[CountryPrompt],
    model: StandardizedTransformer,
    intervention_path: th.Tensor,  # (L, E) - the country-specific path to subtract
    alpha: float,
    top_k: int,
) -> ExpertMasks:
    """
    Extract binary masks of chosen experts before and after intervention, averaged over all prompts.

    Args:
        prompts: List of prompts to run (averaged over all)
        model: The model
        intervention_path: The path to subtract from router outputs (L, E)
        alpha: Scaling factor for the intervention
        top_k: Number of top experts to select

    Returns:
        ExpertMasks containing averaged pre and post intervention binary masks
    """
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    layers_with_routers = list(model.layers_with_routers)

    # Collect masks and probabilities for all prompts
    pre_intervention_masks_list = []
    post_intervention_masks_list = []
    pre_intervention_probs = []
    post_intervention_probs = []

    # Process each prompt
    for prompt in tqdm(prompts, desc="Processing prompts", leave=False):
        seq_len = len(prompt.token_ids)

        # Left-pad sequence
        tokens = prompt.token_ids
        batch_token_ids = tokens.unsqueeze(0)  # (1, T)
        attn_mask = th.ones((1, seq_len), dtype=th.bool, device=tokens.device)

        batch = {
            "input_ids": batch_token_ids,
            "attention_mask": attn_mask,
        }

        # Get the capital token ID for forgetfulness computation
        capital_tokens = model.tokenizer(
            prompt.capital, add_special_tokens=False
        ).input_ids
        assert isinstance(capital_tokens, list) and capital_tokens, (
            f"Capital '{prompt.capital}' not found in tokenizer"
        )
        capital_first_token_id = capital_tokens[0]
        assert isinstance(capital_first_token_id, int), (
            f"Capital '{prompt.capital}' first token ID is not an integer"
        )

        # First pass: get pre-intervention masks and probabilities
        pre_intervention_logits_list = []

        with model.trace(batch):
            for layer_idx in layers_with_routers:
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats
                if isinstance(router_output, tuple):
                    if len(router_output) == 2:
                        router_scores, _router_indices = router_output
                    elif len(router_output) == 3:
                        original_router_logits, _, _ = router_output
                        router_scores = original_router_logits
                    else:
                        raise ValueError(
                            f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                        )
                else:
                    router_scores = router_output

                # Save the traced tensor
                if hasattr(router_scores, "save"):
                    logits = router_scores.save()
                else:
                    logits = router_scores

                # Get logits for the last token: (1, T, E) -> (1, E)
                last_token_logits = logits.reshape(1, seq_len, -1)[:, -1, :]  # (1, E)
                pre_intervention_logits_list.append(last_token_logits)

            # Get pre-intervention probability for the capital
            final_logits = model.lm_head.output.save()  # (1, T, vocab_size)
            final_probs = F.softmax(final_logits.float(), dim=-1)
            pre_intervention_capital_prob = final_probs[
                0, -1, capital_first_token_id
            ].item()
            pre_intervention_probs.append(pre_intervention_capital_prob)

        # Stack pre-intervention logits: (1, L, E)
        pre_intervention_logits = th.stack(
            pre_intervention_logits_list, dim=1
        )  # (1, L, E)
        pre_intervention_logits = pre_intervention_logits.squeeze(0)  # (L, E)

        # Convert to binary mask
        pre_intervention_mask = convert_router_logits_to_paths(
            pre_intervention_logits.unsqueeze(0), top_k
        ).squeeze(0)  # (L, E)
        pre_intervention_masks_list.append(pre_intervention_mask)

        # Second pass: get post-intervention masks
        intervention_path_device = intervention_path.to(
            device=batch_token_ids.device, dtype=th.float32
        )

        post_intervention_logits_list = []

        with model.trace(batch):
            for i, layer_idx in enumerate(layers_with_routers):
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats
                router_output_is_tuple = isinstance(router_output, tuple)
                router_output_len = len(router_output) if router_output_is_tuple else 0

                if router_output_is_tuple:
                    if router_output_len == 2:
                        router_scores, _router_indices = router_output
                        raise ValueError(
                            "Cannot run this experiment on a model whose routers do not return raw logits"
                        )
                    elif router_output_len == 3:
                        (
                            original_router_logits,
                            original_router_weights,
                            original_router_indices,
                        ) = router_output

                        router_logits = cast("th.Tensor", original_router_logits.save())
                        router_logits = router_logits.reshape(1, seq_len, -1)
                        router_weights = cast(
                            "th.Tensor", original_router_weights.save()
                        )
                        router_weights = router_weights.reshape(1, seq_len, -1)
                        router_indices = cast(
                            "th.Tensor", original_router_indices.save()
                        )
                        router_indices = router_indices.reshape(1, seq_len, -1)
                    else:
                        raise ValueError(
                            f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                        )
                else:
                    router_scores = cast("th.Tensor", router_output.save())
                    router_logits = router_scores.reshape(1, seq_len, -1)  # (1, T, E)

                # Apply intervention to the last token's logits
                intervention_path_device = intervention_path_device.to(
                    device=router_logits.device, dtype=router_logits.dtype
                )
                layer_intervention = intervention_path_device[i]  # (E,)
                modified_logits = router_logits.clone()
                modified_logits[:, -1, :] -= alpha * layer_intervention

                if router_output_is_tuple:
                    if router_output_len == 3:
                        new_weights, new_indices = th.topk(
                            modified_logits[:, -1, :], k=top_k, dim=-1
                        )
                        new_weights = F.softmax(
                            new_weights, dim=-1, dtype=new_weights.dtype
                        )

                        modified_weights = cast("th.Tensor", router_weights.save())
                        modified_weights[:, -1, :] = new_weights
                        modified_indices = cast("th.Tensor", router_indices.save())
                        modified_indices[:, -1, :] = new_indices

                        model.routers_output[layer_idx] = (
                            modified_logits.reshape(-1, modified_logits.shape[-1]),
                            modified_weights.reshape(-1, modified_weights.shape[-1]),
                            modified_indices.reshape(-1, modified_indices.shape[-1]),
                        )
                    else:
                        raise ValueError(
                            f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                        )
                else:
                    model.routers_output[layer_idx] = modified_logits.reshape(
                        -1, modified_logits.shape[-1]
                    )

                # Save the modified logits for mask extraction
                last_token_logits = modified_logits[:, -1, :]  # (1, E)
                post_intervention_logits_list.append(last_token_logits)

            # Get post-intervention probability for the capital
            final_logits = model.lm_head.output.save()  # (1, T, vocab_size)
            final_probs = F.softmax(final_logits.float(), dim=-1)
            post_intervention_capital_prob = final_probs[
                0, -1, capital_first_token_id
            ].item()
            post_intervention_probs.append(post_intervention_capital_prob)

        # Stack post-intervention logits: (1, L, E)
        post_intervention_logits = th.stack(
            post_intervention_logits_list, dim=1
        )  # (1, L, E)
        post_intervention_logits = post_intervention_logits.squeeze(0)  # (L, E)

        # Convert to binary mask
        post_intervention_mask = convert_router_logits_to_paths(
            post_intervention_logits.unsqueeze(0), top_k
        ).squeeze(0)  # (L, E)
        post_intervention_masks_list.append(post_intervention_mask)

    # Average masks across all prompts
    # Stack all masks: (N, L, E) where N is number of prompts
    pre_intervention_masks_stacked = th.stack(
        pre_intervention_masks_list, dim=0
    )  # (N, L, E)
    post_intervention_masks_stacked = th.stack(
        post_intervention_masks_list, dim=0
    )  # (N, L, E)

    # Average: (N, L, E) -> (L, E)
    # This gives us the fraction of prompts where each expert was selected
    pre_intervention_mask_avg = pre_intervention_masks_stacked.float().mean(
        dim=0
    )  # (L, E)
    post_intervention_mask_avg = post_intervention_masks_stacked.float().mean(
        dim=0
    )  # (L, E)

    # Average forgetfulness across all prompts
    avg_pre_prob = sum(pre_intervention_probs) / len(pre_intervention_probs)
    avg_post_prob = sum(post_intervention_probs) / len(post_intervention_probs)
    forgetfulness = (
        (avg_pre_prob - avg_post_prob) / avg_pre_prob if avg_pre_prob > 0 else 0.0
    )

    return ExpertMasks(
        pre_intervention=pre_intervention_mask_avg.cpu(),
        post_intervention=post_intervention_mask_avg.cpu(),
        forgetfulness=forgetfulness,
    )


def visualize_expert_diff(
    masks: ExpertMasks,
    target_country: str,
    alpha: float,
    output_path: Path,
) -> None:
    """
    Visualize the difference in expert selection before and after intervention.

    Args:
        masks: ExpertMasks containing pre and post intervention masks
        target_country: Name of target country
        alpha: Alpha value used for intervention
        output_path: Path to save the visualization
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Pre-intervention mask
    ax1 = axes[0]
    im1 = ax1.imshow(
        masks.pre_intervention.cpu().float().numpy(),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax1.set_xlabel("Expert Index", fontsize=12)
    ax1.set_ylabel("Layer Index", fontsize=12)
    ax1.set_title(f"Pre-Intervention Expert Selection\n{target_country}", fontsize=14)
    plt.colorbar(im1, ax=ax1, label="Selected (1) / Not Selected (0)")

    # Plot 2: Post-intervention mask
    ax2 = axes[1]
    im2 = ax2.imshow(
        masks.post_intervention.cpu().float().numpy(),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax2.set_xlabel("Expert Index", fontsize=12)
    ax2.set_ylabel("Layer Index", fontsize=12)
    ax2.set_title(
        f"Post-Intervention Expert Selection\n{target_country} (α={alpha})", fontsize=14
    )
    plt.colorbar(im2, ax=ax2, label="Selected (1) / Not Selected (0)")

    # Plot 3: Difference (post - pre)
    ax3 = axes[2]
    diff = masks.post_intervention - masks.pre_intervention
    im3 = ax3.imshow(
        diff.cpu().float().numpy(),
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )
    ax3.set_xlabel("Expert Index", fontsize=12)
    ax3.set_ylabel("Layer Index", fontsize=12)
    ax3.set_title(f"Difference (Post - Pre)\n{target_country} (α={alpha})", fontsize=14)
    plt.colorbar(im3, ax=ax3, label="Change (-1: removed, 0: unchanged, +1: added)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


@arguably.command()
def capital_country_expert_diff(
    *,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    target_country: str = "South Korea",
    postprocessor: str = "masks",
    router_path_batch_size: int = 128,
    seed: int = 0,
    hf_token: str = "",
    output_dir: str = "out/capital_country_expert_diff",
    log_level: str = "INFO",
) -> None:
    """
    Visualize expert selection differences before and after intervention.

    For a chosen country, computes the mean difference and runs interventions
    with different alpha values, capturing binary masks of chosen experts.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, gpt, etc.)
        model_step_ckpt: Checkpoint step to load (None for latest)
        model_dtype: Data type for model weights
        target_country: Country to analyze (default: "South Korea")
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
    logger.info(
        f"Running capital_country_expert_diff experiment with log level: {log_level}"
    )

    # Validate target country
    if target_country not in COUNTRY_TO_CAPITAL:
        logger.error(
            f"Target country '{target_country}' not found in COUNTRY_TO_CAPITAL"
        )
        logger.info(f"Available countries: {sorted(COUNTRY_TO_CAPITAL.keys())}")
        sys.exit(1)

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
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
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
    tokenizer = model.tokenizer

    logger.info("Model loaded successfully")
    logger.info(f"Number of layers with routers: {len(model.layers_with_routers)}")
    logger.info(f"Layers with routers: {model.layers_with_routers}")

    # Get model architecture info
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok

    logger.info(f"Number of experts: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Step 1: Extract router paths for all prompts
    logger.info("=" * 80)
    logger.info("STEP 1: Extracting router paths")
    logger.info("=" * 80)

    country_paths = extract_router_paths(
        model,
        top_k=top_k,
        batch_size=router_path_batch_size,
        postprocessor=postprocessor_enum,
    )

    # Step 2: Compute average paths
    logger.info("=" * 80)
    logger.info("STEP 2: Computing average paths")
    logger.info("=" * 80)

    avg_paths = compute_average_paths(country_paths)
    logger.info(f"Computed average paths for {len(avg_paths)} countries")

    # Step 3: Compute country-specific path (mean difference)
    logger.info("=" * 80)
    logger.info("STEP 3: Computing country-specific path")
    logger.info("=" * 80)

    country_specific_paths = compute_country_specific_paths(avg_paths, target_country)
    logger.info(f"Computed country-specific paths for {target_country}")

    # Use PRE_ANSWER experiment type for consistency
    intervention_path = country_specific_paths[ExperimentType.PRE_ANSWER]
    logger.info(f"Intervention path shape: {intervention_path.shape}")

    # Step 4: Get prompts for the target country
    logger.info("=" * 80)
    logger.info("STEP 4: Getting prompts for target country")
    logger.info("=" * 80)

    all_prompts = get_all_prompts(tokenizer)
    target_prompts = [p for p in all_prompts if p.country == target_country]
    logger.info(f"Found {len(target_prompts)} prompts for {target_country}")

    # Step 5: Run interventions for different alpha values
    logger.info("=" * 80)
    logger.info("STEP 5: Running interventions for different alpha values")
    logger.info("=" * 80)

    # Alpha values: 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    country_slug = target_country.lower().replace(" ", "_")
    country_output_dir = output_path / country_slug
    country_output_dir.mkdir(parents=True, exist_ok=True)

    for alpha in tqdm(alphas, desc="Processing alpha values"):
        logger.info(f"Processing alpha={alpha}")

        # Extract expert masks
        masks = extract_expert_masks_with_intervention(
            target_prompts,
            model,
            intervention_path,
            alpha,
            top_k,
        )

        # Save masks
        alpha_str = f"{alpha:.2f}".replace(".", "_")
        mask_file = country_output_dir / f"{alpha_str}.pt"
        th.save(
            {
                "pre_intervention": masks.pre_intervention,
                "post_intervention": masks.post_intervention,
                "forgetfulness": masks.forgetfulness,
                "alpha": alpha,
                "target_country": target_country,
            },
            mask_file,
        )
        logger.info(f"Saved masks to {mask_file}")

        # Create visualization in fig directory
        fig_country_dir = (
            Path(FIGURE_DIR) / "capital_country_expert_diff" / country_slug
        )
        viz_file = fig_country_dir / f"{alpha_str}.png"
        visualize_expert_diff(masks, target_country, alpha, viz_file)

        # Clean up
        gc.collect()
        th.cuda.empty_cache()

    # Print summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target country: {target_country}")
    logger.info(f"Alpha values tested: {alphas}")
    logger.info(f"Results saved to: {country_output_dir}")


if __name__ == "__main__":
    arguably.run()
