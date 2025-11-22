import asyncio
from collections.abc import Callable
from functools import partial
import json
from multiprocessing import cpu_count
from pathlib import Path
import sys

import arguably
from delphi.__main__ import non_redundant_hookpoints  # type: ignore
from delphi.__main__ import populate_cache as sae_populate_cache  # type: ignore
from delphi.clients import Offline  # type: ignore
from delphi.config import (  # type: ignore
    CacheConfig,
    ConstructorConfig,
    RunConfig,
    SamplerConfig,
)
from delphi.latents import LatentDataset, LatentRecord  # type: ignore
from delphi.latents.cache import (  # type: ignore
    InMemoryCache,
    LatentCache,
    generate_statistics_cache,
)
from delphi.log.result_analysis import log_results  # type: ignore
from delphi.pipeline import Pipe, Pipeline  # type: ignore
from delphi.scorers.classifier.intruder import IntruderScorer  # type: ignore
from delphi.scorers.scorer import ScorerResult  # type: ignore
from delphi.utils import load_tokenized_data  # type: ignore
from dictionary_learning.utils import load_dictionary
from loguru import logger
from nnterp import StandardizedTransformer
import orjson
import torch as th
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from core.device import get_backend
from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import (
    RouterLogitsPostprocessor,
    get_postprocessor,
)
from core.type import assert_type
from exp import OUTPUT_DIR
from exp.get_activations import ActivationKeys
from exp.kmeans import KMEANS_FILENAME


def dataset_postprocess(record: LatentRecord) -> LatentRecord:
    return record


# Saves the score to a file
def save_scorer_result_to_file(result: ScorerResult, score_dir: Path) -> None:
    safe_latent_name = str(result.record.latent).replace("/", "--")

    with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: th.Tensor | None,
) -> None:
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    and scores in the `scores_path` directory.
    """
    latent_dict: dict[str, th.Tensor] | None = (
        dict.fromkeys(hookpoints, latent_range) if latent_range is not None else None
    )

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )
    llm_client = Offline(
        run_cfg.explainer_model,
        max_memory=0.9,
        max_model_len=run_cfg.explainer_model_max_len,
        num_gpus=run_cfg.num_gpus,
        statistics=run_cfg.verbose,
    )

    intruder_scorer = IntruderScorer(
        llm_client,
        verbose=run_cfg.verbose,
        n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
        temperature=getattr(run_cfg, "temperature", 0.0),
        cot=getattr(run_cfg, "cot", False),
        type=getattr(run_cfg, "intruder_type", "default"),
        seed=run_cfg.seed,
    )

    pipeline = Pipeline(
        dataset,
        Pipe(dataset_postprocess),
        Pipe(intruder_scorer),
        Pipe(partial(save_scorer_result_to_file, score_dir=scores_path)),
    )

    await pipeline.run(run_cfg.pipeline_num_proc)


ACTIVATION_KEYS_TO_HOOKPOINT = {
    ActivationKeys.MLP_OUTPUT: "model.model.layers.{layer}.mlp",
    ActivationKeys.ROUTER_LOGITS: "model.model.layers.{layer}.mlp.gate",
    ActivationKeys.ATTN_OUTPUT: "model.model.layers.{layer}.self_attn",
    ActivationKeys.LAYER_OUTPUT: "model.model.layers.{layer}",
}


def load_hookpoints_and_saes(
    sae_base_path: Path,
    dtype: th.dtype,
) -> dict[str, Callable[[th.Tensor], th.Tensor]]:
    hookpoints_to_saes = {}

    for sae_dirpath in sae_base_path.iterdir():
        if not sae_dirpath.is_dir():
            continue

        config_path = sae_dirpath / "config.json"
        if not config_path.is_file():
            continue

        with open(config_path) as f:
            config = json.load(f)

        ae, _ = load_dictionary(str(sae_dirpath), device="cuda")
        ae = ae.to(dtype)
        trainer_config = config.get("trainer")
        if trainer_config is None:
            logger.trace(config)
            raise ValueError(
                f"Trainer config is not set in the config for SAE at {sae_dirpath}"
            )

        layer = trainer_config.get("layer")
        if layer is None:
            logger.trace(config)
            raise ValueError(f"Layer is not set in the config for SAE at {sae_dirpath}")

        layer = int(layer)

        # i stored the activation key in the submodule_name field...
        activation_key = trainer_config.get("submodule_name")
        if activation_key is None:
            logger.trace(config)
            raise ValueError(
                f"Submodule name is not set in the config for SAE at {sae_dirpath}"
            )

        hookpoint = ACTIVATION_KEYS_TO_HOOKPOINT[activation_key].format(layer=layer)

        assert hookpoint not in hookpoints_to_saes, (
            f"Hookpoint {hookpoint} already exists in {hookpoints_to_saes.keys()}"
        )
        hookpoints_to_saes[hookpoint] = ae.encode

    logger.debug(
        f"Loaded {len(hookpoints_to_saes)} hookpoints and SAEs: {hookpoints_to_saes.keys()}"
    )

    return hookpoints_to_saes


def load_hookpoints(
    root_dir: Path,
    dtype: th.dtype,
) -> tuple[dict[str, Callable[[th.Tensor], th.Tensor]], int | None]:
    """
    Loads the hookpoints from the config file.
    """
    sae_config_path = root_dir / "config.yaml"
    if sae_config_path.is_file():
        # this is a sae experiment, not paths
        return load_hookpoints_and_saes(root_dir, dtype=dtype), None

    paths_path = root_dir / KMEANS_FILENAME
    if not paths_path.is_file():
        raise ValueError(f"Paths file not found at {paths_path}")

    with open(paths_path, "rb") as f:
        data = th.load(f)

    centroid_sets: list[th.Tensor] = data["centroids"]
    top_k: int = data["top_k"]

    hookpoints_to_sparse_encode = {}
    for centroids_idx, centroids in enumerate(centroid_sets):
        path_projection = nn.Linear(
            centroids.shape[1],
            centroids.shape[0],
            bias=False,
            device="cuda",
            dtype=dtype,
        )
        path_projection.weight.data.copy_(centroids)
        hookpoints_to_sparse_encode[f"paths_{centroids_idx}"] = path_projection

    return hookpoints_to_sparse_encode, top_k


class LatentPathsCache(LatentCache):
    def __init__(
        self,
        model: StandardizedTransformer,
        hookpoint_to_sparse_encode: dict[str, Callable],
        batch_size: int,
        log_path: Path | None = None,
        postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
    ):
        """
        Initialize the LatentCache.

        Args:
            model: The model to cache latents for.
            hookpoint_to_sparse_encode: Dictionary of sparse encoding functions.
            batch_size: Size of batches for processing.
            log_path: Path to save logging output.
            postprocessor: Router logits postprocessing method to use.
        """
        self.model: StandardizedTransformer = model
        self.hookpoint_to_sparse_encode = hookpoint_to_sparse_encode
        self.batch_size = batch_size
        self.widths = {}
        self.cache = InMemoryCache(filters=None, batch_size=batch_size)

        self.log_path = log_path
        self.postprocessor_fn = get_postprocessor(postprocessor)

    def run(self, n_tokens: int, tokens: th.Tensor, top_k: int, dtype: th.dtype):
        """
        Run the latent caching process.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.
            top_k: Top k paths to cache.
            dtype: Data type to use for the sparse paths.
        """
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()
        for batch_idx, batch in tqdm(
            enumerate(token_batches),
            total=total_batches,
            desc="Caching latents",
        ):
            total_tokens += tokens_per_batch
            router_paths = []

            with self.model.trace(batch):
                for layer_idx in tqdm(
                    self.model.layers_with_routers,
                    desc=f"Batch {batch_idx}",
                    total=len(self.model.layers_with_routers),
                    leave=False,
                    position=1,
                ):
                    router_output = self.model.routers_output[layer_idx]

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

                    router_paths.append(logits)

            router_paths = th.stack(router_paths, dim=-2)  # (B, T, L, E)
            sparse_paths = self.postprocessor_fn(router_paths, top_k).to(dtype=dtype)
            del router_paths

            router_paths_BTP = sparse_paths.view(*batch.shape, -1)  # (B, T, L * E)

            for hookpoint, sparse_encode in self.hookpoint_to_sparse_encode.items():
                sae_latents = sparse_encode(router_paths_BTP)
                self.cache.add(sae_latents, batch, batch_idx, hookpoint)

                self.widths[hookpoint] = sae_latents.shape[2]

        logger.info(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()

    def generate_statistics_cache(self):
        """
        Print statistics (number of dead features, number of single token features)
        to the console.
        """
        assert len(self.widths) > 0, "Widths must be set before generating statistics"
        logger.info("Feature statistics:")

        # Token frequency
        for hookpoint in self.cache.latent_locations:
            width = self.widths[hookpoint]

            logger.info(f"# Hookpoint: {hookpoint}")
            logger.debug(f"# Width: {width}")

            generate_statistics_cache(
                self.cache.tokens[hookpoint],
                self.cache.latent_locations[hookpoint],
                self.cache.latent_activations[hookpoint],
                width,
                verbose=True,
            )


def load_and_filter_tokens(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    cache_cfg: CacheConfig,
    run_cfg: RunConfig,
) -> th.Tensor:
    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            logger.debug("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~(flattened_tokens == tokenizer.bos_token_id)
            masked_tokens = flattened_tokens[mask]

            num_non_bos_tokens = masked_tokens.shape[0]
            extra_tokens = num_non_bos_tokens % cache_cfg.cache_ctx_len

            if extra_tokens > 0:
                logger.debug(
                    f"Warning: {extra_tokens} extra tokens after BOS filtering, truncating to {num_non_bos_tokens - extra_tokens}"
                )
                truncated_tokens = masked_tokens[:-extra_tokens]
                tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)
            else:
                tokens = masked_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    return tokens


def populate_cache(
    run_cfg: RunConfig,
    model: StandardizedTransformer,
    hookpoint_to_sparse_encode: dict[str, Callable],
    root_dir: Path,
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    top_k: int,
    dtype: th.dtype,
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
) -> None:
    """
    Populates an on-disk cache in `latents_path` with latent activations.
    """
    sae_config_path = root_dir / "config.yaml"
    if sae_config_path.is_file():
        # this is a sae experiment, not paths
        logger.debug(
            f"Running SAE populate cache on {root_dir} with hookpoints {hookpoint_to_sparse_encode.keys()}"
        )

        return sae_populate_cache(
            run_cfg,
            model,
            hookpoint_to_sparse_encode,
            latents_path,
            tokenizer,
            transcode=False,
        )

    logger.debug(
        f"Running paths populate cache on {root_dir} with hookpoints {hookpoint_to_sparse_encode.keys()}"
    )

    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_and_filter_tokens(tokenizer, cache_cfg, run_cfg)

    cache = LatentPathsCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        log_path=log_path,
        postprocessor=postprocessor,
    )
    cache.run(cache_cfg.n_tokens, tokens, top_k=top_k, dtype=dtype)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


@arguably.command()
def eval_intruder(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
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
    explainer_model: str = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
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
    device_type: str = "cuda",
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Set GPU count dynamically based on device type
    if num_gpus is None:
        backend = get_backend(device_type)
        num_gpus = backend.device_count() if backend.is_available() else 0

    logger.info(f"Running with log level: {log_level}")

    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)
    dtype_torch = get_dtype(dtype)

    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    root_dir = Path(OUTPUT_DIR, experiment_dir)
    base_path = root_dir / "delphi"
    latents_path = base_path / "latents"
    scores_path = base_path / "scores"
    visualize_path = base_path / "visualize"

    th.manual_seed(seed)

    logger.debug(
        f"Loading model from {model_config.hf_name} with revision {model_ckpt}"
    )

    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        torch_dtype=model_dtype_torch,
        token=hf_token,
    )
    tokenizer = model.tokenizer

    logger.trace("Model and tokenizer initialized")

    hookpoint_to_sparse_encode, top_k = load_hookpoints(root_dir, dtype=dtype_torch)
    hookpoints = list(hookpoint_to_sparse_encode.keys())

    latent_range = th.arange(n_latents) if n_latents else None

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

    nrh = assert_type(
        non_redundant_hookpoints(
            hookpoint_to_sparse_encode, latents_path, overwrite=False
        ),
        dict,
    )
    if nrh:
        logger.info(f"Populating cache with {len(nrh)} hookpoints")
        if top_k is None:
            raise ValueError("top_k cannot be None when populating cache")
        populate_cache(
            run_cfg,
            model,
            nrh,
            root_dir,
            latents_path,
            tokenizer,
            top_k=top_k,
            dtype=dtype_torch,
            postprocessor=postprocessor,
        )
    else:
        logger.debug("No non-redundant hookpoints found, skipping cache population")

    del model, hookpoint_to_sparse_encode

    nrh = assert_type(
        non_redundant_hookpoints(hookpoints, scores_path, overwrite=False),
        list,
    )
    if nrh:
        logger.info(f"Processing cache with {len(nrh)} hookpoints")
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
        logger.debug("No non-redundant hookpoints found, skipping cache processing")

    if run_cfg.verbose:
        logger.debug("Logging results")
        log_results(scores_path, visualize_path, run_cfg.hookpoints, run_cfg.scorers)


if __name__ == "__main__":
    arguably.run()
