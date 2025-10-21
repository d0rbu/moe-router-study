"""
Experiment to determine the causal impact of MoE routing.

This experiment tests whether specific routing patterns have causal effects on model behavior by:
1. Loading pre-computed k-means centroids representing routing patterns
2. Running baseline intruder detection on naturally activating samples
3. Generating samples with causally-modified routing based on centroids
4. Comparing natural vs causal activation patterns via intruder detection

Usage:
    uv run python -m exp.eval_causal_routing \\
        --experiment-dir "path/to/kmeans/experiment" \\
        --num-centroids 64 \\
        --influence 0.8 \\
        --model-name "olmoe-i"
"""

import asyncio
from collections.abc import Callable
import json
from multiprocessing import cpu_count
from pathlib import Path
import random
import sys
from typing import Any, cast

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import yaml

from core.data import get_dataset_fn
from core.dtype import get_dtype
from core.model import get_model_config
from delphi.config import (  # type: ignore
    CacheConfig,
    ConstructorConfig,
    RunConfig,
    SamplerConfig,
)
from exp import OUTPUT_DIR
from exp.eval_intruder import (
    load_hookpoints,
    populate_cache,
    process_cache,
)
from exp.kmeans import KMEANS_FILENAME


def load_seed_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    sample_length: int,
    max_samples: int,
    seed: int = 0,
) -> list[th.Tensor]:
    """
    Load and filter a dataset to create seed tensors for generation.

    Args:
        dataset_name: Name of the dataset to load (e.g., "lmsys")
        tokenizer: Tokenizer to use for tokenization
        sample_length: Target length for each sample (samples will be filtered to this length)
        max_samples: Maximum number of samples to collect
        seed: Random seed for reproducible sampling

    Returns:
        List of tokenized tensors, each of shape (sample_length,) with int token IDs
    """
    logger.debug(
        f"Loading seed dataset '{dataset_name}' with max_samples={max_samples}, sample_length={sample_length}"
    )

    # Get the dataset function
    dataset_fn = get_dataset_fn(dataset_name)

    # Create dataset iterator
    dataset_iter = dataset_fn(cast("PreTrainedTokenizer", tokenizer))

    # Set random seed for reproducible sampling
    random.seed(seed)

    seed_tensors = []
    processed_count = 0

    # Create progress bar for seed collection
    pbar = tqdm(total=max_samples, desc="Collecting seed samples", leave=False)

    for text in dataset_iter:
        if len(seed_tensors) >= max_samples:
            break

        processed_count += 1

        # Tokenize the text
        tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]

        # Filter by length: must be at least sample_length tokens
        if len(tokens) < sample_length:
            continue

        # Truncate to exact sample_length
        truncated_tokens = tokens[:sample_length]

        # Validate that tokens are integers
        assert truncated_tokens.dtype in [th.int32, th.int64, th.long], (
            f"Expected int tensor, got {truncated_tokens.dtype}"
        )

        seed_tensors.append(truncated_tokens)
        pbar.update(1)

    pbar.close()

    logger.debug(
        f"Collected {len(seed_tensors)} seed samples from {processed_count} texts"
    )

    if len(seed_tensors) == 0:
        raise ValueError(
            f"No samples found with length >= {sample_length} in dataset '{dataset_name}'"
        )

    return seed_tensors


def load_and_select_centroid(
    experiment_dir: Path,
    num_centroids: int,
    dtype: th.dtype,
    centroid_idx: int | None = None,
    seed: int = 0,
) -> tuple[th.Tensor, int, int]:
    """
    Load k-means centroids and select one based on num_centroids parameter.

    Args:
        experiment_dir: Directory containing kmeans.pt file
        num_centroids: Number of centroids to select (k value)
        dtype: Data type for the centroid tensor
        centroid_idx: Specific centroid index to select (None for random selection)
        seed: Random seed for reproducible centroid selection when centroid_idx is None

    Returns:
        Tuple of (centroid_tensor, centroid_idx, top_k):
            - centroid_tensor: Reshaped centroid of shape (L, E) where L=layers, E=experts
            - centroid_idx: Index of the selected centroid within its set
            - top_k: Top-k value used for routing
    """
    kmeans_path = experiment_dir / KMEANS_FILENAME
    if not kmeans_path.is_file():
        raise FileNotFoundError(f"K-means file not found at {kmeans_path}")

    logger.debug(f"Loading centroids from {kmeans_path}")
    with open(kmeans_path, "rb") as f:
        data = th.load(f)

    centroid_sets: list[th.Tensor] = data["centroids"]
    top_k: int = data["top_k"]

    # Find the centroid set with the matching number of centroids
    available_sizes = [centroids.shape[0] for centroids in centroid_sets]
    try:
        matching_set_idx = available_sizes.index(num_centroids)
    except ValueError:
        logger.critical(
            f"No centroid set found with {num_centroids} centroids. "
            f"Available sizes: {available_sizes}"
        )
        raise

    centroids = centroid_sets[matching_set_idx]
    logger.debug(
        f"Found centroid set at index {matching_set_idx} with shape {centroids.shape}"
    )

    # Select specific centroid or pick one randomly
    if centroid_idx is None:
        # Set random seed for reproducible selection
        random.seed(seed)
        selected_centroid_idx = random.randint(0, num_centroids - 1)
        logger.debug(
            f"Randomly selected centroid {selected_centroid_idx} (seed={seed})"
        )
    else:
        assert 0 <= centroid_idx < num_centroids, (
            f"centroid_idx {centroid_idx} out of range [0, {num_centroids})"
        )
        selected_centroid_idx = centroid_idx
        logger.debug(f"Using specified centroid {selected_centroid_idx}")

    selected_centroid_flat = centroids[selected_centroid_idx]  # Shape: (L * E,)

    # Determine L and E dimensions
    # We need to know the number of experts per layer and number of layers
    # This should be derivable from the flat dimension and model architecture
    # For now, we'll infer from the data structure
    flat_dim = selected_centroid_flat.shape[0]

    # Load metadata to get activation dimensions and shape information
    metadata_path = experiment_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    activation_dim = metadata.get("activation_dim")
    num_layers = metadata.get("num_layers")
    num_experts = metadata.get("num_experts")

    assert activation_dim is not None, (
        f"activation_dim not found in metadata at {metadata_path}"
    )
    assert num_layers is not None, (
        f"num_layers not found in metadata at {metadata_path}"
    )
    assert num_experts is not None, (
        f"num_experts not found in metadata at {metadata_path}"
    )

    # Verify dimensions are consistent
    expected_dim = num_layers * num_experts
    assert flat_dim == activation_dim, (
        f"Centroid dimension ({flat_dim}) doesn't match "
        f"activation dimension ({activation_dim}) from metadata"
    )
    assert activation_dim == expected_dim, (
        f"Activation dimension ({activation_dim}) doesn't match "
        f"num_layers * num_experts ({num_layers} * {num_experts} = {expected_dim})"
    )

    # Reshape from (L * E,) to (L, E)
    centroid_reshaped = selected_centroid_flat.reshape(num_layers, num_experts).to(
        dtype=dtype
    )

    logger.info(
        f"Selected centroid {selected_centroid_idx} from set with {num_centroids} centroids. "
        f"Reshaped from {selected_centroid_flat.shape} to {centroid_reshaped.shape}"
    )

    return centroid_reshaped, selected_centroid_idx, top_k


def create_causal_forward_fn(
    model: StandardizedTransformer, centroid: th.Tensor, top_k: int
) -> Callable:
    """
    Create a causally-modified forward pass function for generation.

    Args:
        model: The transformer model
        centroid: Centroid tensor of shape (L, E) for router modulation
        top_k: Top-k value for routing

    Returns:
        Function that performs causally-modified forward pass
    """

    def causal_forward(input_ids: th.Tensor) -> th.Tensor:
        with model.trace(input_ids):
            # Loop through layers with routers and modify only the last token
            for layer_idx in model.layers_with_routers:
                # Get router logits for this layer: shape (B, T, E)
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats and get tensor
                if isinstance(router_output, tuple):
                    if len(router_output) == 2:
                        router_scores, _router_indices = router_output
                    else:
                        raise ValueError(
                            f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                        )
                else:
                    router_scores = router_output

                # Convert to tensor using .save() method
                router_logits = cast("th.Tensor", router_scores.save())

                # Only modify the last token in the sequence: shape (B, E)
                last_token_logits = router_logits[:, -1, :]

                # Apply softmax to get probabilities
                router_probs = F.softmax(last_token_logits, dim=-1)

                # Multiply by centroid values for this layer
                modulated_probs = router_probs * centroid[layer_idx]  # (B, E)

                # Apply top-k: zero out all but top-k values
                topk_values, topk_indices = th.topk(modulated_probs, k=top_k, dim=-1)

                # Create mask with zeros everywhere except top-k positions
                topk_probs = th.zeros_like(modulated_probs)
                topk_probs.scatter_(dim=-1, index=topk_indices, src=topk_values)

                # Renormalize to maintain valid probability distribution
                renormalized_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

                # Set the modified router probabilities for this layer
                # We need to update the full tensor, keeping other positions unchanged
                full_router_probs = F.softmax(router_logits, dim=-1)
                full_router_probs[:, -1, :] = (
                    renormalized_probs  # Update only last token
                )

                # Assign back to model using copy_ to update in-place
                layer_probs = model.router_probabilities[layer_idx].save()
                layer_probs.copy_(full_router_probs)

            # Get final logits after all router modifications
            logits = model.lm_head.output.save()

        return logits

    return causal_forward


def generate_causal_samples(
    model: StandardizedTransformer,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    centroid: th.Tensor,
    top_k: int,
    influence: float,
    num_samples: int = 10,
    max_length: int = 256,
    temperature: float = 1.0,
    seed: int = 0,
    seed_dataset: list[th.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate samples with causally-modulated routing using nnterp tracing.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for the model
        centroid: Centroid tensor of shape (L, E)
        top_k: Top-k value for routing
        influence: Probability of applying modulation per token (0.0-1.0)
        num_samples: Number of samples to generate
        max_length: Maximum length of generated sequences
        temperature: Temperature for sampling
        seed: Random seed for reproducibility
        seed_dataset: Optional list of tokenized seed tensors to start generation from.
                     If None, generates from BOS token. If provided, samples
                     are selected randomly from this dataset as starting points.
                     Each tensor should be a 1D int tensor of token IDs.

    Returns:
        List of dictionaries containing generated samples and metadata
    """
    # Validate influence parameter
    assert 0.0 <= influence <= 1.0, (
        f"influence must be between 0.0 and 1.0, got {influence}"
    )

    logger.debug(f"Generating {num_samples} causal samples with influence={influence}")

    # Set random seed for reproducibility
    th.manual_seed(seed)
    rng = random.Random(seed)

    # Create causal forward pass function (standard forward is just model)
    causal_forward = create_causal_forward_fn(model, centroid, top_k)

    # Generate samples using autoregressive generation with probabilistic forward selection
    samples = []
    for i in tqdm(range(num_samples), desc="Generating causal samples"):
        # Determine starting point for generation
        seed_tensor = None
        if seed_dataset is not None:
            # Select a random seed tensor from the dataset
            seed_tensor = rng.choice(seed_dataset)
            # Ensure it's a 2D tensor (batch_size=1, seq_len)
            if seed_tensor.dim() == 1:
                input_ids = seed_tensor.unsqueeze(0).to(model.device)
            else:
                input_ids = seed_tensor.to(model.device)
            logger.debug(
                f"Sample {i}: Starting from seed tensor with {input_ids.shape[1]} tokens"
            )
        else:
            # Use BOS token or empty string as prompt
            if tokenizer.bos_token_id is not None:
                input_ids = th.tensor([[tokenizer.bos_token_id]], device=model.device)
            else:
                # Use empty string as prompt
                input_ids = tokenizer("", return_tensors="pt").input_ids.to(
                    model.device
                )
            logger.debug(
                f"Sample {i}: Starting from BOS/empty with {input_ids.shape[1]} tokens"
            )

        # Autoregressive generation with probabilistic forward pass selection
        generated_tokens = input_ids.clone()

        for _step in range(max_length - input_ids.shape[1]):
            # Decide whether to use causal or standard forward pass for this token
            use_causal = rng.random() < influence

            if use_causal:
                logits = causal_forward(generated_tokens)
            else:
                logits = model(generated_tokens)

            # Apply temperature and sample next token
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = th.multinomial(probs, num_samples=1)

            # Append to sequence
            generated_tokens = th.cat(
                [generated_tokens, next_token.unsqueeze(0)], dim=1
            )

            # Stop if we hit EOS token
            if (
                tokenizer.eos_token_id is not None
                and next_token.item() == tokenizer.eos_token_id
            ):
                break

        # Decode generated sequence
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        samples.append(
            {
                "sample_idx": i,
                "tokens": generated_tokens[0].cpu().tolist(),
                "text": generated_text,
                "influence": influence,
                "seed": seed + i,
                "seed_tensor": seed_tensor.cpu().tolist()
                if seed_dataset is not None
                else None,
                "initial_tokens": input_ids.shape[1],
                "generated_tokens": generated_tokens.shape[1] - input_ids.shape[1],
            }
        )

        logger.trace(f"Generated sample {i}: {generated_text[:100]}...")

    return samples


def run_baseline_intruder_detection(
    run_cfg: RunConfig,
    model: StandardizedTransformer,
    hookpoint_to_sparse_encode: dict[str, Callable],
    root_dir: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    top_k: int,
    dtype: th.dtype,
    centroid_idx: int,
) -> dict[str, Any]:
    """
    Run baseline intruder detection on naturally activating samples.

    Args:
        run_cfg: Configuration for the run
        model: The transformer model
        hookpoint_to_sparse_encode: Dictionary mapping hookpoints to encoding functions
        root_dir: Root directory for the experiment
        tokenizer: Tokenizer for the model
        top_k: Top-k value for routing
        dtype: Data type for tensors
        centroid_idx: Index of the centroid being evaluated

    Returns:
        Dictionary containing baseline results
    """
    logger.info("Running baseline intruder detection on natural samples")

    base_path = root_dir / "causal_routing" / "baseline"
    latents_path = base_path / "latents"
    scores_path = base_path / "scores"

    # Populate cache with natural activations
    logger.debug("Populating cache with natural activations")
    populate_cache(
        run_cfg,
        model,
        hookpoint_to_sparse_encode,
        root_dir,
        latents_path,
        tokenizer,
        top_k=top_k,
        dtype=dtype,
    )

    # Process cache and run intruder detection
    logger.debug("Processing cache and running intruder detection")
    hookpoints = list(hookpoint_to_sparse_encode.keys())

    # Focus on the specific centroid/path
    latent_range = th.tensor([centroid_idx])

    asyncio.run(
        process_cache(
            run_cfg,
            latents_path,
            scores_path,
            hookpoints,
            tokenizer,
            latent_range,
        )
    )

    logger.info("Baseline intruder detection complete")

    return {
        "latents_path": str(latents_path),
        "scores_path": str(scores_path),
        "centroid_idx": centroid_idx,
    }


def run_causal_vs_nonactivating_intruder(
    causal_samples: list[dict[str, Any]],
    _run_cfg: RunConfig,
    root_dir: Path,
    centroid_idx: int,
) -> dict[str, Any]:
    """
    Run intruder detection with 1 causal sample vs many non-activating samples.

    Args:
        causal_samples: List of generated causal samples
        _run_cfg: Configuration for the run
        root_dir: Root directory for the experiment
        centroid_idx: Index of the centroid being evaluated

    Returns:
        Dictionary containing results
    """
    logger.info("Running intruder detection: causal vs non-activating")

    # TODO: Implement custom intruder detection with causal samples
    # This requires creating a custom dataset structure

    _base_path = root_dir / "causal_routing" / "causal_vs_nonactivating"

    results = {
        "num_causal_samples": len(causal_samples),
        "centroid_idx": centroid_idx,
        "status": "not_implemented",
    }

    logger.warning("Causal vs non-activating intruder detection not yet implemented")

    return results


def run_causal_vs_natural_intruder(
    causal_samples: list[dict[str, Any]],
    _run_cfg: RunConfig,
    root_dir: Path,
    centroid_idx: int,
) -> dict[str, Any]:
    """
    Run intruder detection with 1 causal sample vs many natural activating samples.

    Args:
        causal_samples: List of generated causal samples
        _run_cfg: Configuration for the run
        root_dir: Root directory for the experiment
        centroid_idx: Index of the centroid being evaluated

    Returns:
        Dictionary containing results
    """
    logger.info("Running intruder detection: causal vs natural activating")

    # TODO: Implement custom intruder detection comparing causal vs natural samples

    _base_path = root_dir / "causal_routing" / "causal_vs_natural"

    results = {
        "num_causal_samples": len(causal_samples),
        "centroid_idx": centroid_idx,
        "status": "not_implemented",
    }

    logger.warning("Causal vs natural intruder detection not yet implemented")

    return results


@arguably.command()
def eval_causal_routing(
    *,
    experiment_dir: str,
    num_centroids: int,
    influence: float = 0.8,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    dtype: str = "bf16",
    load_in_8bit: bool = False,
    # Generation settings
    num_causal_samples: int = 10,
    max_gen_length: int = 256,
    gen_temperature: float = 1.0,
    seed_dataset: str | None = None,
    seed_sample_length: int = 256,
    max_seed_samples: int = 1000,
    # Intruder detection settings
    ctxlen: int = 256,
    n_tokens: int = 10_000_000,
    batchsize: int = 8,
    n_latents: int = 1000,
    example_ctx_len: int = 32,
    min_examples: int = 200,
    num_non_activating: int = 50,
    num_examples: int = 50,
    n_quantiles: int = 10,
    explainer_model: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    explainer_model_max_len: int = 5120,
    filter_bos: bool = False,
    pipeline_num_proc: int = cpu_count() // 2,
    num_gpus: int = th.cuda.device_count(),
    verbose: bool = True,
    seed: int = 0,
    hf_token: str = "",
    log_level: str = "INFO",
) -> None:
    """
    Evaluate the causal impact of MoE routing patterns.

    This experiment:
    1. Loads k-means centroids from the experiment directory
    2. Selects a specific centroid based on num_centroids parameter
    3. Runs baseline intruder detection on naturally activating samples
    4. Generates samples with causally-modulated routing
    5. Runs intruder detection variants comparing natural vs causal activation

    Args:
        experiment_dir: Directory containing kmeans.pt and metadata files
        num_centroids: Number of centroids (k value) to select from
        influence: Probability of applying routing modulation (0.0 to 1.0)
        model_name: Name of the model to use
        model_step_ckpt: Checkpoint step to load (None for latest)
        model_dtype: Data type for model weights
        dtype: Data type for activations and computations
        load_in_8bit: Whether to load model in 8-bit quantization
        num_causal_samples: Number of causal samples to generate
        max_gen_length: Maximum length for generated sequences
        gen_temperature: Temperature for sampling during generation
        seed_dataset: Optional dataset name to load seed samples from (e.g., "lmsys").
                     If None, generates from BOS token. Dataset will be loaded, tokenized,
                     and filtered to create seed tensors for generation.
        seed_sample_length: Length to truncate/filter seed samples to
        max_seed_samples: Maximum number of seed samples to collect from dataset
        ctxlen: Context length for activation caching
        n_tokens: Number of tokens to process for caching
        batchsize: Batch size for processing
        n_latents: Number of latents to evaluate
        example_ctx_len: Context length for examples
        min_examples: Minimum number of examples per latent
        num_non_activating: Number of non-activating examples
        num_examples: Number of examples to show in prompts
        n_quantiles: Number of quantiles for sampling
        explainer_model: Model to use for explanation
        explainer_model_max_len: Maximum length for explainer model
        filter_bos: Whether to filter BOS tokens
        pipeline_num_proc: Number of processes for pipeline
        num_gpus: Number of GPUs to use
        verbose: Whether to enable verbose logging
        seed: Random seed for reproducibility
        hf_token: Hugging Face API token
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running causal routing evaluation with log level: {log_level}")

    # Set random seeds
    th.manual_seed(seed)
    random.seed(seed)

    # Get model configuration
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)
    dtype_torch = get_dtype(dtype)

    # Setup quantization if needed
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    # Setup paths
    root_dir = Path(OUTPUT_DIR, experiment_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {root_dir}")

    logger.info(f"Loading model from {model_config.hf_name} with revision {model_ckpt}")

    # Load model
    model = StandardizedTransformer(
        model_config.hf_name,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        torch_dtype=model_dtype_torch,
        token=hf_token,
    )
    tokenizer = model.tokenizer

    logger.info("Model loaded successfully")

    # Load and select centroid
    centroid, centroid_idx, top_k = load_and_select_centroid(
        root_dir, num_centroids, dtype_torch
    )

    # Load hookpoints for intruder detection
    hookpoint_to_sparse_encode, top_k_loaded = load_hookpoints(
        root_dir, dtype=dtype_torch
    )
    if top_k_loaded is not None:
        top_k = top_k_loaded
    hookpoints = list(hookpoint_to_sparse_encode.keys())

    # Setup run configuration for intruder detection
    run_cfg = RunConfig(
        max_latents=n_latents,
        cache_cfg=CacheConfig(
            cache_ctx_len=ctxlen,
            batch_size=batchsize,
            n_tokens=n_tokens,
        ),
        constructor_cfg=ConstructorConfig(
            example_ctx_len=example_ctx_len,
            min_examples=min_examples,
            n_non_activating=num_non_activating,
            non_activating_source="random",
        ),
        sampler_cfg=SamplerConfig(
            n_examples_train=0,
            n_examples_test=num_examples,
            n_quantiles=n_quantiles,
            train_type="quantiles",
            test_type="quantiles",
        ),
        model=model_name,
        sparse_model=str(root_dir),
        hookpoints=hookpoints,
        explainer_model=explainer_model,
        explainer_model_max_len=explainer_model_max_len,
        explainer_provider="offline",
        explainer="default",
        filter_bos=filter_bos,
        load_in_8bit=load_in_8bit,
        hf_token=hf_token,
        pipeline_num_proc=pipeline_num_proc,
        num_gpus=num_gpus,
        seed=seed,
        verbose=verbose,
    )

    # Step 1: Run baseline intruder detection
    logger.info("=" * 80)
    logger.info("STEP 1: Baseline intruder detection on natural samples")
    logger.info("=" * 80)

    baseline_results = run_baseline_intruder_detection(
        run_cfg,
        model,
        hookpoint_to_sparse_encode,
        root_dir,
        tokenizer,
        top_k,
        dtype_torch,
        centroid_idx,
    )

    # Step 2: Generate causal samples
    logger.info("=" * 80)
    logger.info("STEP 2: Generating causal samples")
    logger.info("=" * 80)

    # Load seed dataset if specified
    seed_tensors = None
    if seed_dataset is not None:
        logger.info(f"Loading seed dataset: {seed_dataset}")
        seed_tensors = load_seed_dataset(
            seed_dataset,
            tokenizer,
            seed_sample_length,
            max_seed_samples,
            seed=seed,
        )
        logger.info(f"Loaded {len(seed_tensors)} seed samples")

    causal_samples = generate_causal_samples(
        model,
        tokenizer,
        centroid,
        top_k,
        influence,
        num_samples=num_causal_samples,
        max_length=max_gen_length,
        temperature=gen_temperature,
        seed=seed,
        seed_dataset=seed_tensors,
    )

    # Save generated samples
    causal_samples_path = root_dir / "causal_routing" / "generated_samples.json"
    causal_samples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(causal_samples_path, "w") as f:
        json.dump(causal_samples, f, indent=2)
    logger.info(f"Saved generated samples to {causal_samples_path}")

    # Step 3: Run causal vs non-activating intruder detection
    logger.info("=" * 80)
    logger.info("STEP 3: Intruder detection - causal vs non-activating")
    logger.info("=" * 80)

    causal_vs_nonactivating_results = run_causal_vs_nonactivating_intruder(
        causal_samples,
        run_cfg,
        root_dir,
        centroid_idx,
    )

    # Step 4: Run causal vs natural intruder detection
    logger.info("=" * 80)
    logger.info("STEP 4: Intruder detection - causal vs natural activating")
    logger.info("=" * 80)

    causal_vs_natural_results = run_causal_vs_natural_intruder(
        causal_samples,
        run_cfg,
        root_dir,
        centroid_idx,
    )

    # Aggregate results
    summary = {
        "experiment_dir": str(root_dir),
        "num_centroids": num_centroids,
        "centroid_idx": centroid_idx,
        "influence": influence,
        "num_causal_samples": len(causal_samples),
        "baseline_results": baseline_results,
        "causal_vs_nonactivating_results": causal_vs_nonactivating_results,
        "causal_vs_natural_results": causal_vs_natural_results,
    }

    # Save summary
    summary_path = root_dir / "causal_routing" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Evaluated centroid {centroid_idx} with influence {influence}")
    logger.info(f"Generated {len(causal_samples)} causal samples")


if __name__ == "__main__":
    arguably.run()
