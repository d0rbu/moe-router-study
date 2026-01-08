"""
Generate and save intervention paths using Contrastive Activation Addition (CAA).

This script implements the CAA method from "Steering Llama 2 via Contrastive Activation Addition"
by computing steering vectors from contrastive pairs of prompts. Instead of averaging paths
for different countries, it computes the difference between positive and negative examples
and averages these differences to get a steering vector.

Usage:
    # Generate steering vector for sycophancy behavior
    uv run python -m exp.capital_country_generate_path_caa \\
        --model-name "olmoe-i" \\
        --dataset-file "CAA/datasets/generate/sycophancy/generate_dataset.json" \\
        --output-file "out/intervention_paths/sycophancy_caa.pt" \\
        --layer 13

    # Then use it in chat:
    uv run python -m exp.capital_country_chat \\
        --model-name "olmoe-i" \\
        --intervention-path "out/intervention_paths/sycophancy_caa.pt" \\
        --alpha 1.0
"""

import json
from pathlib import Path
import sys

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor, get_postprocessor


def load_contrastive_dataset(dataset_path: Path) -> list[dict]:
    """
    Load contrastive pairs from a JSON dataset file.

    Expected format:
    [
        {
            "question": "...",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)"
        },
        ...
    ]

    Args:
        dataset_path: Path to the JSON dataset file

    Returns:
        List of contrastive pair dictionaries
    """
    with open(dataset_path) as f:
        data = json.load(f)

    # Validate format
    for item in data:
        if not all(
            key in item
            for key in [
                "question",
                "answer_matching_behavior",
                "answer_not_matching_behavior",
            ]
        ):
            raise ValueError(
                "Invalid dataset format. Each item must have 'question', "
                "'answer_matching_behavior', and 'answer_not_matching_behavior' keys"
            )

    return data


def tokenize_prompt_with_answer(
    tokenizer,
    question: str,
    answer: str,
    use_chat_format: bool = True,
) -> th.Tensor:
    """
    Tokenize a question with an answer appended.

    Args:
        tokenizer: The tokenizer to use
        question: The question text
        answer: The answer text (e.g., "(A)" or "(B)")
        use_chat_format: Whether to use chat format (for chat models)

    Returns:
        Token IDs as a tensor of shape (1, T)
    """
    if use_chat_format:
        # Use the tokenizer's chat template (same as existing code)
        messages = [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        tokens = tokenizer(formatted, return_tensors="pt").input_ids
    else:
        # Use base format: question + answer
        prompt = f"{question.strip()} {answer.strip()}"
        tokens = tokenizer.encode(prompt, return_tensors="pt")

    return tokens


def extract_router_paths_at_answer_token(
    model: StandardizedTransformer,
    question: str,
    answer: str,
    top_k: int,
    postprocessor_fn,
    use_chat_format: bool = True,
) -> th.Tensor:
    """
    Extract router paths at the answer token position.

    Similar to CAA, we extract activations at the position of the answer token
    (the last token, which is the answer letter like "A" or "B").

    Args:
        model: The MoE model
        question: The question text
        answer: The answer text (e.g., "(A)")
        top_k: Number of top experts
        postprocessor_fn: Function to postprocess router logits
        use_chat_format: Whether to use chat format

    Returns:
        Router paths at answer token position, shape (L, E)
    """
    # Tokenize the prompt
    tokens = tokenize_prompt_with_answer(
        model.tokenizer,
        question,
        answer,
        use_chat_format=use_chat_format,
    )
    tokens = tokens.to(model.device)

    # Get sequence length
    seq_len = tokens.shape[1]

    # Extract router logits
    router_logits_list = []
    layers_with_routers = list(model.layers_with_routers)

    batch = {"input_ids": tokens}

    with model.trace(batch):
        for layer_idx in layers_with_routers:
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
            # Reshape to (1, T, E)
            router_logits = logits.reshape(1, seq_len, -1)
            router_logits_list.append(router_logits)

    # Stack into (1, T, L, E)
    router_logits = th.stack(router_logits_list, dim=2)

    # Apply postprocessor to get paths
    router_paths = postprocessor_fn(router_logits, top_k)  # (1, T, L, E)

    # Extract paths at the answer token position
    # Following CAA, we use position -2 (second-to-last token) because the answer
    # like "(A)" is typically tokenized as two tokens: "(" and "A", and we want
    # the activation at the letter token position. If the answer is a single token,
    # this will still work (position -2 will be the token before the answer).
    # For safety, we check the sequence length and use -1 if sequence is too short.
    answer_token_idx = -2 if seq_len >= 2 else -1

    # Shape: (1, L, E) -> (L, E)
    answer_token_paths = router_paths[0, answer_token_idx, :, :].cpu()

    return answer_token_paths


def compute_caa_steering_vector(
    model: StandardizedTransformer,
    contrastive_pairs: list[dict],
    top_k: int,
    postprocessor: RouterLogitsPostprocessor,
    use_chat_format: bool = True,
) -> th.Tensor:
    """
    Compute CAA steering vector from contrastive pairs.

    For each pair:
    1. Extract router paths for positive example (answer_matching_behavior)
    2. Extract router paths for negative example (answer_not_matching_behavior)
    3. Compute difference: positive - negative
    4. Average all differences to get steering vector

    Args:
        model: The MoE model
        contrastive_pairs: List of contrastive pair dictionaries
        top_k: Number of top experts
        postprocessor: Router logits postprocessor
        batch_size: Batch size for processing (not used for now, but kept for consistency)
        use_chat_format: Whether to use chat format

    Returns:
        Steering vector of shape (L, E)
    """
    postprocessor_fn = get_postprocessor(postprocessor)

    # Store differences for each pair
    differences = []

    logger.info(f"Processing {len(contrastive_pairs)} contrastive pairs...")

    for pair in tqdm(contrastive_pairs, desc="Processing contrastive pairs"):
        question = pair["question"]
        pos_answer = pair["answer_matching_behavior"]
        neg_answer = pair["answer_not_matching_behavior"]

        # Extract paths for positive example
        pos_paths = extract_router_paths_at_answer_token(
            model,
            question,
            pos_answer,
            top_k,
            postprocessor_fn,
            use_chat_format=use_chat_format,
        )  # (L, E)

        # Extract paths for negative example
        neg_paths = extract_router_paths_at_answer_token(
            model,
            question,
            neg_answer,
            top_k,
            postprocessor_fn,
            use_chat_format=use_chat_format,
        )  # (L, E)

        # Compute difference: positive - negative
        diff = pos_paths - neg_paths  # (L, E)
        differences.append(diff)

    # Stack all differences: (N, L, E)
    all_differences = th.stack(differences, dim=0)

    # Average over all pairs: (N, L, E) -> (L, E)
    steering_vector = all_differences.mean(dim=0)

    return steering_vector


@arguably.command()
def capital_country_generate_path_caa(
    *,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    dataset_file: str = "",
    behavior_name: str = "",
    postprocessor: str = "masks",
    batch_size: int = 32,
    output_file: str = "",
    hf_token: str = "",
    use_chat_format: bool = True,
    log_level: str = "INFO",
) -> None:
    """
    Generate and save a CAA steering vector from contrastive pairs.

    This implements Contrastive Activation Addition (CAA) by computing the mean
    difference between router paths for positive and negative examples.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, etc.)
        model_step_ckpt: Checkpoint step to load (None for latest)
        model_dtype: Data type for model weights
        dataset_file: Path to JSON file with contrastive pairs
        behavior_name: Name of the behavior (for metadata, e.g., "sycophancy")
        postprocessor: Router logits postprocessor (masks, identity, softmax, etc.)
        batch_size: Batch size (currently not used, kept for future optimization)
        output_file: Path to save the steering vector (default: out/intervention_paths/<behavior>_caa.pt)
        hf_token: Hugging Face API token
        use_chat_format: Whether to use chat format for tokenization
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(
        f"Running capital_country_generate_path_caa with log level: {log_level}"
    )

    # Validate dataset file
    if not dataset_file:
        raise ValueError("--dataset-file is required")

    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}")

    # Parse postprocessor
    postprocessor_enum = RouterLogitsPostprocessor(postprocessor)
    logger.info(f"Postprocessor: {postprocessor_enum}")

    # Set default output file if not provided
    if not output_file:
        if behavior_name:
            behavior_slug = behavior_name.lower().replace(" ", "_")
        else:
            behavior_slug = dataset_path.stem
        output_file = f"out/intervention_paths/{behavior_slug}_caa.pt"

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

    # Step 1: Load contrastive pairs
    logger.info("=" * 80)
    logger.info("STEP 1: Loading contrastive pairs")
    logger.info("=" * 80)

    print(f"⏳ Loading contrastive pairs from {dataset_path}...")
    contrastive_pairs = load_contrastive_dataset(dataset_path)
    print(f"✅ Loaded {len(contrastive_pairs)} contrastive pairs!")

    # Step 2: Compute CAA steering vector
    logger.info("=" * 80)
    logger.info("STEP 2: Computing CAA steering vector")
    logger.info("=" * 80)

    print("⏳ Computing steering vector...")
    steering_vector = compute_caa_steering_vector(
        model,
        contrastive_pairs,
        top_k=top_k,
        postprocessor=postprocessor_enum,
        batch_size=batch_size,
        use_chat_format=use_chat_format,
    )
    print("✅ Steering vector computed!")

    logger.info(f"Steering vector shape: {steering_vector.shape}")
    logger.info(f"Vector min: {steering_vector.min().item():.4f}")
    logger.info(f"Vector max: {steering_vector.max().item():.4f}")
    logger.info(f"Vector mean: {steering_vector.mean().item():.4f}")
    logger.info(f"Vector std: {steering_vector.std().item():.4f}")

    # Step 3: Save the steering vector
    logger.info("=" * 80)
    logger.info("STEP 3: Saving steering vector")
    logger.info("=" * 80)

    th.save(
        {
            "intervention_path": steering_vector,
            "behavior_name": behavior_name or dataset_path.stem,
            "dataset_file": str(dataset_path),
            "num_pairs": len(contrastive_pairs),
            "postprocessor": postprocessor,
            "model_name": model_name,
            "num_layers": len(layers_with_routers),
            "num_experts": num_experts,
            "method": "caa",
        },
        output_path,
    )

    print(f"✅ Saved steering vector to {output_path}")
    print("\nTo use this vector in chat:")
    print("  uv run python -m exp.capital_country_chat \\")
    print(f"      --model-name {model_name} \\")
    print(f"      --intervention-path {output_path} \\")
    print("      --alpha 1.0")

    # Print summary
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Behavior: {behavior_name or dataset_path.stem}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Number of pairs: {len(contrastive_pairs)}")
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    arguably.run()
