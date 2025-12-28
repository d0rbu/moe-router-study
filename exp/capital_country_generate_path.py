"""
Generate and save intervention paths for country-capital knowledge suppression.

This script computes the "country-specific path" - the difference between a target
country's average expert activation pattern and the average of all other countries.
This path can be used with capital_country_chat.py to steer the model's behavior.

Usage:
    # Generate intervention path for South Korea
    uv run python -m exp.capital_country_generate_path \\
        --model-name "olmoe-i" \\
        --target-country "South Korea" \\
        --output-file "out/intervention_paths/south_korea.pt"

    # Then use it in chat:
    uv run python -m exp.capital_country_chat \\
        --model-name "olmoe-i" \\
        --intervention-path "out/intervention_paths/south_korea.pt" \\
        --alpha 1.0
"""

from pathlib import Path
import sys

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor
from exp.capital_country import (
    COUNTRY_TO_CAPITAL,
    ExperimentType,
    compute_average_paths,
    compute_country_specific_paths,
    extract_router_paths,
)


@arguably.command()
def capital_country_generate_path(
    *,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    target_country: str = "South Korea",
    experiment_type: str = "pre_answer",
    postprocessor: str = "masks",
    batch_size: int = 128,
    output_file: str = "",
    hf_token: str = "",
    log_level: str = "INFO",
) -> None:
    """
    Generate and save an intervention path for a specific country.

    The intervention path represents the difference between the target country's
    average expert activation pattern and the average of all other countries.
    Higher values in the path indicate experts more specific to the target country.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, etc.)
        model_step_ckpt: Checkpoint step to load (None for latest)
        model_dtype: Data type for model weights
        target_country: Country to generate path for (e.g., "South Korea")
        experiment_type: Experiment type for path extraction (pre_answer, assistant_response, from_country_mention)
        postprocessor: Router logits postprocessor (masks, identity, softmax, etc.)
        batch_size: Batch size for router path extraction
        output_file: Path to save the intervention path (default: out/intervention_paths/<country>.pt)
        hf_token: Hugging Face API token
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running capital_country_generate_path with log level: {log_level}")

    # Validate target country
    if target_country not in COUNTRY_TO_CAPITAL:
        raise ValueError(
            f"Unknown target country: {target_country}. "
            f"Valid countries: {list(COUNTRY_TO_CAPITAL.keys())}"
        )

    # Parse experiment type
    try:
        exp_type = ExperimentType(experiment_type)
    except ValueError:
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. "
            f"Valid types: {[e.value for e in ExperimentType]}"
        ) from None

    # Parse postprocessor
    postprocessor_enum = RouterLogitsPostprocessor(postprocessor)
    logger.info(f"Postprocessor: {postprocessor_enum}")

    # Set default output file if not provided
    if not output_file:
        country_slug = target_country.lower().replace(" ", "_")
        output_file = f"out/intervention_paths/{country_slug}.pt"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)

    logger.info(f"Loading model: {model_config.hf_name}")
    logger.info(f"Checkpoint: {model_ckpt}")

    # Load model
    print(f"⏳ Loading model {model_config.hf_name}...")
    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        torch_dtype=model_dtype_torch,
        token=hf_token,
    )
    print("✅ Model loaded!")

    # Get model architecture info
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok
    layers_with_routers = list(model.layers_with_routers)

    logger.info(f"Number of layers with routers: {len(layers_with_routers)}")
    logger.info(f"Number of experts per layer: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Step 1: Extract router paths for all countries
    logger.info("=" * 80)
    logger.info("STEP 1: Extracting router paths for all countries")
    logger.info("=" * 80)

    print("⏳ Extracting router paths...")
    country_paths = extract_router_paths(
        model,
        top_k=top_k,
        batch_size=batch_size,
        postprocessor=postprocessor_enum,
    )
    print("✅ Router paths extracted!")

    # Step 2: Compute average paths
    logger.info("=" * 80)
    logger.info("STEP 2: Computing average paths")
    logger.info("=" * 80)

    avg_paths = compute_average_paths(country_paths)
    logger.info(f"Computed average paths for {len(avg_paths)} countries")

    # Step 3: Compute country-specific path
    logger.info("=" * 80)
    logger.info("STEP 3: Computing country-specific path")
    logger.info("=" * 80)

    country_specific_paths = compute_country_specific_paths(avg_paths, target_country)
    intervention_path = country_specific_paths[exp_type]

    logger.info(f"Intervention path shape: {intervention_path.shape}")
    logger.info(f"Path min: {intervention_path.min().item():.4f}")
    logger.info(f"Path max: {intervention_path.max().item():.4f}")
    logger.info(f"Path mean: {intervention_path.mean().item():.4f}")

    # Step 4: Save the intervention path
    logger.info("=" * 80)
    logger.info("STEP 4: Saving intervention path")
    logger.info("=" * 80)

    th.save(
        {
            "intervention_path": intervention_path,
            "target_country": target_country,
            "target_capital": COUNTRY_TO_CAPITAL[target_country],
            "experiment_type": exp_type.value,
            "model_name": model_name,
            "num_layers": len(layers_with_routers),
            "num_experts": num_experts,
        },
        output_path,
    )

    print(f"✅ Saved intervention path to {output_path}")
    print("\nTo use this path in chat:")
    print("  uv run python -m exp.capital_country_chat \\")
    print(f"      --model-name {model_name} \\")
    print(f"      --intervention-path {output_path} \\")
    print("      --alpha 1.0")

    # Print summary
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target country: {target_country}")
    logger.info(f"Target capital: {COUNTRY_TO_CAPITAL[target_country]}")
    logger.info(f"Experiment type: {exp_type.value}")
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    arguably.run()
