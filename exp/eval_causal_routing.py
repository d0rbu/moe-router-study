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
from typing import Any

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

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


def load_and_select_centroid(
    experiment_dir: Path,
    num_centroids: int,
    dtype: th.dtype,
) -> tuple[th.Tensor, int, int]:
    """
    Load k-means centroids and select one based on num_centroids parameter.

    Args:
        experiment_dir: Directory containing kmeans.pt file
        num_centroids: Number of centroids to select (k value)
        dtype: Data type for the centroid tensor

    Returns:
        Tuple of (centroid_tensor, centroid_idx, top_k):
            - centroid_tensor: Reshaped centroid of shape (L, E) where L=layers, E=experts
            - centroid_idx: Index of the selected centroid within its set
            - top_k: Top-k value used for routing
    """
    kmeans_path = experiment_dir / KMEANS_FILENAME
    if not kmeans_path.is_file():
        raise FileNotFoundError(f"K-means file not found at {kmeans_path}")

    logger.info(f"Loading centroids from {kmeans_path}")
    with open(kmeans_path, "rb") as f:
        data = th.load(f)

    centroid_sets: list[th.Tensor] = data["centroids"]
    top_k: int = data["top_k"]

    # Find the centroid set with the matching number of centroids
    matching_set_idx = None
    for idx, centroid_set in enumerate(centroid_sets):
        if centroid_set.shape[0] == num_centroids:
            matching_set_idx = idx
            break

    if matching_set_idx is None:
        available_sizes = [cs.shape[0] for cs in centroid_sets]
        raise ValueError(
            f"No centroid set found with {num_centroids} centroids. "
            f"Available sizes: {available_sizes}"
        )

    centroid_set = centroid_sets[matching_set_idx]
    logger.debug(
        f"Found centroid set at index {matching_set_idx} with shape {centroid_set.shape}"
    )

    # Randomly select one centroid from the set
    centroid_idx = random.randint(0, num_centroids - 1)
    selected_centroid_flat = centroid_set[centroid_idx]  # Shape: (L * E,)

    # Determine L and E dimensions
    # We need to know the number of experts per layer and number of layers
    # This should be derivable from the flat dimension and model architecture
    # For now, we'll infer from the data structure
    flat_dim = selected_centroid_flat.shape[0]

    # Load metadata to get activation dimensions
    metadata_path = experiment_dir / "metadata.yaml"
    if metadata_path.is_file():
        import yaml

        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        activation_dim = metadata.get("activation_dim")
        if activation_dim is None:
            raise ValueError(f"activation_dim not found in metadata at {metadata_path}")

        # activation_dim is E (number of experts)
        # flat_dim = L * E, so L = flat_dim / E
        E = activation_dim
        L = flat_dim // E

        if flat_dim % E != 0:
            raise ValueError(
                f"Flat dimension {flat_dim} is not divisible by activation_dim {E}"
            )
    else:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Reshape from (L * E,) to (L, E)
    centroid_reshaped = selected_centroid_flat.reshape(L, E).to(dtype=dtype)

    logger.info(
        f"Selected centroid {centroid_idx} from set with {num_centroids} centroids. "
        f"Reshaped from {selected_centroid_flat.shape} to {centroid_reshaped.shape}"
    )

    return centroid_reshaped, centroid_idx, top_k


class RouterModulationHook:
    """
    Forward hook that modulates router logits based on centroid values.

    This hook intercepts router logits during the forward pass and applies
    centroid-based modulation using the formula:
        modified_probs = renormalize(softmax(logits) * centroid[layer])

    The modulation is applied probabilistically based on the influence parameter.
    """

    def __init__(
        self,
        centroid: th.Tensor,  # Shape: (L, E)
        influence: float,
        layer_idx: int,
        seed: int = 0,
    ):
        """
        Args:
            centroid: Centroid tensor of shape (L, E)
            influence: Probability of applying modulation (0.0 to 1.0)
            layer_idx: Index of the layer this hook is attached to
            seed: Random seed for reproducibility
        """
        self.centroid = centroid  # (L, E)
        self.influence = influence
        self.layer_idx = layer_idx
        self.rng = random.Random(seed)
        self.apply_modulation = True  # Will be set per forward pass

    def __call__(
        self,
        module: nn.Module,
        input: tuple[th.Tensor, ...],
        output: th.Tensor,
    ) -> th.Tensor:
        """
        Hook function that modulates router logits.

        Args:
            module: The router module
            input: Input tensors to the module
            output: Router logits of shape (B, T, E) or (B, E)

        Returns:
            Modified router logits with same shape as output
        """
        # Decide whether to apply modulation for this forward pass
        if self.rng.random() > self.influence:
            return output

        # Get the centroid values for this layer
        layer_centroid = self.centroid[self.layer_idx]  # Shape: (E,)

        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=-1)  # Shape: (B, T, E) or (B, E)

        # Multiply by centroid values (broadcasting)
        modulated_probs = probs * layer_centroid.unsqueeze(0).unsqueeze(
            0
        )  # Broadcast to (B, T, E)

        # Renormalize to maintain valid probability distribution
        modulated_probs = modulated_probs / modulated_probs.sum(dim=-1, keepdim=True)

        # Convert back to logits (inverse softmax)
        # Use log to get logits from probabilities
        modulated_logits = th.log(modulated_probs + 1e-10)  # Add epsilon for stability

        return modulated_logits


def generate_causal_samples(
    model: StandardizedTransformer,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    centroid: th.Tensor,
    influence: float,
    num_samples: int = 10,
    max_length: int = 256,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Generate samples with causally-modulated routing.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for the model
        centroid: Centroid tensor of shape (L, E)
        influence: Probability of applying modulation per token
        num_samples: Number of samples to generate
        max_length: Maximum length of generated sequences
        temperature: Temperature for sampling
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries containing generated samples and metadata
    """
    logger.info(f"Generating {num_samples} causal samples with influence={influence}")

    # Set random seed for reproducibility
    th.manual_seed(seed)
    random.seed(seed)

    # Identify router layers in the model
    # Assuming model has attribute layers_with_routers or similar
    # For now, we'll try to find router modules by name
    router_modules = []
    router_layer_indices = []

    for name, module in model.named_modules():
        # Look for router/gate modules
        if "gate" in name.lower() or "router" in name.lower():
            # Extract layer index from name (e.g., "layers.5.mlp.gate")
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        router_modules.append(module)
                        router_layer_indices.append(layer_idx)
                        break
                    except ValueError:
                        continue

    logger.debug(
        f"Found {len(router_modules)} router modules at layers: {router_layer_indices}"
    )

    # Register hooks on all router modules
    hooks = []
    for router_module, layer_idx in zip(
        router_modules, router_layer_indices, strict=False
    ):
        hook = RouterModulationHook(
            centroid=centroid,
            influence=influence,
            layer_idx=layer_idx,
            seed=seed + layer_idx,  # Different seed per layer
        )
        handle = router_module.register_forward_hook(hook)
        hooks.append(handle)

    # Generate samples
    samples = []
    try:
        for i in tqdm(range(num_samples), desc="Generating causal samples"):
            # Use a simple prompt or BOS token
            if tokenizer.bos_token_id is not None:
                input_ids = th.tensor([[tokenizer.bos_token_id]], device=model.device)
            else:
                # Use empty string as prompt
                input_ids = tokenizer("", return_tensors="pt").input_ids.to(
                    model.device
                )

            # Generate
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore

            samples.append(
                {
                    "sample_idx": i,
                    "tokens": output_ids[0].cpu().tolist(),  # type: ignore
                    "text": generated_text,
                    "influence": influence,
                    "seed": seed + i,
                }
            )

            logger.trace(f"Generated sample {i}: {generated_text[:100]}...")

    finally:
        # Remove all hooks
        for handle in hooks:
            handle.remove()
        logger.debug("Removed all router hooks")

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

    causal_samples = generate_causal_samples(
        model,
        tokenizer,
        centroid,
        influence,
        num_samples=num_causal_samples,
        max_length=max_gen_length,
        temperature=gen_temperature,
        seed=seed,
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
