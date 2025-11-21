"""
Evaluation script for raw model activations using intruder detection.

This script evaluates raw model activations from specified layers and activation types
by collecting activations directly and running intruder detection on them.
"""

import asyncio
from multiprocessing import cpu_count
from pathlib import Path
import sys

import arguably
from delphi.__main__ import non_redundant_hookpoints
from delphi.config import (
    CacheConfig,
    ConstructorConfig,
    RunConfig,
    SamplerConfig,
)
from delphi.latents.cache import InMemoryCache, LatentCache
from delphi.log.result_analysis import log_results
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from core.device import DeviceType, get_backend
from core.dtype import get_dtype
from core.model import get_model_config
from core.type import assert_type
from exp import OUTPUT_DIR
from exp.eval_intruder import (
    ACTIVATION_KEYS_TO_HOOKPOINT,
    load_and_filter_tokens,
    process_cache,
)
from exp.get_activations import ActivationKeys


class RawActivationsCache(LatentCache):
    """Custom cache for raw model activations."""

    def __init__(
        self,
        model: StandardizedTransformer,
        activation_key: ActivationKeys,
        layers: set[int],
        batch_size: int,
        log_path: Path | None = None,
    ):
        self.model: StandardizedTransformer = model
        self.activation_key = activation_key
        self.layers_sorted = sorted(layers)  # Ensure consistent ordering
        self.batch_size = batch_size
        self.widths = {}
        self.cache = InMemoryCache(filters=None, batch_size=batch_size)
        self.log_path = log_path

    def run(self, n_tokens: int, tokens: th.Tensor, dtype: th.dtype):
        """Run the activation caching process."""
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        for batch_idx, batch in tqdm(
            enumerate(token_batches),
            total=total_batches,
            desc="Caching raw activations",
        ):
            total_tokens += tokens_per_batch

            with self.model.trace(batch):
                layer_activations = []

                for layer_idx in self.layers_sorted:
                    # Get the hookpoint for this layer and activation type
                    hookpoint_template = ACTIVATION_KEYS_TO_HOOKPOINT[
                        self.activation_key
                    ]
                    hookpoint_name = hookpoint_template.format(layer=layer_idx)

                    # Extract activation based on type
                    match self.activation_key:
                        case ActivationKeys.LAYER_OUTPUT:
                            activation = self.model.layers_output[layer_idx].save()
                        case ActivationKeys.MLP_OUTPUT:
                            activation = self.model.mlps_output[layer_idx].save()
                        case ActivationKeys.ATTN_OUTPUT:
                            activation = self.model.attentions_output[layer_idx].save()
                        case ActivationKeys.ROUTER_LOGITS:
                            activation = self.model.routers_output[layer_idx].save()
                        case _:
                            raise ValueError(
                                f"Unsupported activation key: {self.activation_key}"
                            )

                    layer_activations.append(activation)

                # Concatenate activations across layers: (B, T, sum(hidden_sizes))
                concat_activations = th.cat(layer_activations, dim=-1)

                # Create a synthetic hookpoint name for the concatenated activations
                hookpoint = f"raw_{self.activation_key}_layers_{'_'.join(map(str, self.layers_sorted))}"

                self.cache.add(concat_activations, batch, batch_idx, hookpoint)
                self.widths[hookpoint] = concat_activations.shape[2]

        logger.info(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()


def populate_cache(
    run_cfg: RunConfig,
    model: StandardizedTransformer,
    activation_key: ActivationKeys,
    layers: set[int],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dtype: th.dtype,
) -> None:
    """Populate cache with raw activations."""
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create log path
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_and_filter_tokens(tokenizer, cache_cfg, run_cfg)

    cache = RawActivationsCache(
        model,
        activation_key,
        layers,
        batch_size=cache_cfg.batch_size,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens, dtype=dtype)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


@arguably.command()
def eval_raw_activations(
    *,
    model_name: str = "olmoe-i",
    activation_key: ActivationKeys = ActivationKeys.LAYER_OUTPUT,
    layers: list[int] | None = None,
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    dtype: str = "bf16",
    ctxlen: int = 256,
    load_in_8bit: bool = False,
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
    explainer_provider: str = "offline",
    explainer: str = "default",
    filter_bos: bool = False,
    pipeline_num_proc: int = cpu_count() // 2,
    num_gpus: int | None = None,
    verbose: bool = True,
    seed: int = 0,
    hf_token: str = "",
    log_level: str = "INFO",
    device_type: DeviceType = "cuda",
) -> None:
    """
    Evaluate raw model activations using intruder detection.

    Args:
        model_name: Model name to evaluate
        activation_key: Type of activation (layer_output, attn_output, mlp_output, router_logits)
        layers: Set of layer indices to evaluate (if None or empty, uses all layers)
        model_step_ckpt: Model checkpoint step
        model_dtype: Model data type
        dtype: Activation data type
        ctxlen: Context length for caching
        load_in_8bit: Load model in 8-bit
        n_tokens: Number of tokens to process
        batchsize: Batch size for processing
        n_latents: Number of latents to evaluate
        example_ctx_len: Context length for examples
        min_examples: Minimum examples per latent
        num_non_activating: Number of non-activating examples
        num_examples: Number of examples to show
        n_quantiles: Number of quantiles for sampling
        explainer_model: Model for explanations
        explainer_model_max_len: Max length for explainer
        explainer_provider: Provider for explainer
        explainer: Type of explainer
        filter_bos: Filter BOS tokens
        pipeline_num_proc: Number of pipeline processes
        num_gpus: Number of GPUs
        verbose: Verbose output
        seed: Random seed
        hf_token: HuggingFace token
        log_level: Logging level
        device_type: Device type
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Get model config and setup
    model_config = get_model_config(model_name)

    # Set default layers to all layers if None or empty
    if layers is None or len(layers) == 0:
        hf_config = AutoConfig.from_pretrained(model_config.hf_name)
        num_layers = hf_config.num_hidden_layers

        layers = list(range(num_layers))

    layers_unique = set(layers)
    assert len(layers_unique) == len(layers), "Duplicate layers found in layers list"

    layers_sorted = sorted(layers)

    # Set GPU count dynamically
    if num_gpus is None:
        backend = get_backend(device_type)
        num_gpus = backend.device_count() if backend.is_available() else 0

    logger.info(
        f"Evaluating raw activations: {activation_key} from layers {layers_sorted} on {model_name}"
    )
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)
    dtype_torch = get_dtype(dtype)

    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    # Create synthetic hookpoint name for the concatenated activations
    hookpoint = f"raw_{activation_key}_layers_{'_'.join(map(str, layers_sorted))}"

    # Setup paths
    experiment_name = f"{hookpoint}_{model_name}"
    root_dir = Path(OUTPUT_DIR) / experiment_name
    base_path = root_dir / "delphi"
    latents_path = base_path / "latents"
    scores_path = base_path / "scores"
    visualize_path = base_path / "visualize"

    th.manual_seed(seed)

    logger.debug(
        f"Loading model from {model_config.hf_name} with revision {model_ckpt}"
    )

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

    logger.trace("Model and tokenizer initialized")

    hookpoints = [hookpoint]

    latent_range = th.arange(n_latents) if n_latents else None

    # Setup run config
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
        explainer_provider=explainer_provider,
        explainer=explainer,
        filter_bos=filter_bos,
        load_in_8bit=load_in_8bit,
        hf_token=hf_token,
        pipeline_num_proc=pipeline_num_proc,
        num_gpus=num_gpus,
        seed=seed,
        verbose=verbose,
    )

    # Check if we need to populate cache
    nrh = assert_type(
        non_redundant_hookpoints(
            {hookpoint: lambda x: x}, latents_path, overwrite=False
        ),
        dict,
    )
    if nrh:
        logger.info(
            f"Populating cache for {activation_key} activations from layers {layers_sorted}"
        )
        populate_cache(
            run_cfg,
            model,
            activation_key,
            layers_unique,
            latents_path,
            tokenizer,
            dtype_torch,
        )
    else:
        logger.debug("Cache already populated, skipping")

    del model

    # Process cache and run intruder detection
    nrh = assert_type(
        non_redundant_hookpoints(hookpoints, scores_path, overwrite=False),
        list,
    )
    if nrh:
        logger.info(f"Processing cache with hookpoints: {nrh}")
        asyncio.run(
            process_cache(
                run_cfg,
                latents_path,
                scores_path,
                nrh,
                tokenizer,
                latent_range,
            )
        )
    else:
        logger.debug("Scores already exist, skipping processing")

    if run_cfg.verbose:
        logger.debug("Logging results")
        log_results(scores_path, visualize_path, run_cfg.hookpoints, run_cfg.scorers)

    logger.success(
        f"🎉 Raw activations evaluation complete for {activation_key} layers {layers_sorted}!"
    )


if __name__ == "__main__":
    arguably.run()
