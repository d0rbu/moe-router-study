import asyncio
from collections.abc import Callable
from functools import partial
import gc
import json
from pathlib import Path
import queue
import sys

import arguably
from dictionary_learning.utils import load_dictionary
from loguru import logger
from nnterp import StandardizedTransformer
import numpy as np
import orjson
from safetensors.numpy import save_file
import torch as th
import torch.multiprocessing as mp
from torch.multiprocessing import cpu_count
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from core.device import get_backend
from core.dtype import get_dtype
from core.intruder import DiskCache
from core.model import get_model_config
from core.moe import (
    CentroidMetric,
    CentroidProjection,
    RouterLogitsPostprocessor,
    get_postprocessor,
)
from core.type import assert_type
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
    LatentCache,
    generate_statistics_cache,
)
from delphi.log.result_analysis import log_results  # type: ignore
from delphi.pipeline import Pipe, Pipeline  # type: ignore
from delphi.scorers.classifier.intruder import IntruderScorer  # type: ignore
from delphi.scorers.scorer import ScorerResult  # type: ignore
from delphi.utils import load_tokenized_data  # type: ignore
from exp import OUTPUT_DIR
from exp.get_activations import ActivationKeys
from exp.kmeans import KMEANS_FILENAME


@th.inference_mode()
def _gpu_worker(
    gpu_id: int,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    log_queue: mp.Queue,
    model_name: str,
    model_revision: str,
    model_dtype: th.dtype,
    kmeans_path: Path,
    centroid_dtype: th.dtype,
    metric: CentroidMetric,
    metric_p: float,
    postprocessor: RouterLogitsPostprocessor,
    top_k: int,
    hf_token: str,
    quantization_config: BitsAndBytesConfig | None,
):
    """Worker that processes batches on a single GPU."""
    device = f"cuda:{gpu_id}"
    log_queue.put(f"Worker {gpu_id}: Loading model on {device}")

    # Load model
    model = StandardizedTransformer(
        model_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        revision=model_revision,
        device_map={"": device},
        quantization_config=quantization_config,
        torch_dtype=model_dtype,
        token=hf_token,
    )

    # Load centroids
    with open(kmeans_path, "rb") as f:
        data = th.load(f, map_location=device, weights_only=False)
    centroid_sets: list[th.Tensor] = data["centroids"]
    del data

    hookpoint_to_sparse_encode = {}
    for i, centroids in enumerate(centroid_sets):
        hookpoint_to_sparse_encode[f"paths_{i}"] = CentroidProjection(
            centroids.to(device=device, dtype=centroid_dtype), metric=metric, p=metric_p
        )
    del centroid_sets

    postprocessor_fn = get_postprocessor(postprocessor)
    log_queue.put(f"Worker {gpu_id}: Ready")

    # Process batches
    while True:
        item = work_queue.get()
        if item is None:  # Shutdown signal
            break

        batch_idx, batch_tokens = item
        batch_tokens = batch_tokens.to(device)

        # Forward pass
        router_paths = []
        with model.trace(batch_tokens):
            for layer_idx in model.layers_with_routers:
                out = model.routers_output[layer_idx]
                router_paths.append(
                    out[0].save() if isinstance(out, tuple) else out.save()
                )

        router_paths = th.stack(router_paths, dim=-2)
        sparse_paths = postprocessor_fn(router_paths, top_k).to(dtype=centroid_dtype)
        del router_paths

        router_paths_flat = sparse_paths.view(*batch_tokens.shape, -1)
        del sparse_paths

        # Encode
        results = {}
        for hookpoint, encoder in hookpoint_to_sparse_encode.items():
            latents = encoder(router_paths_flat)
            results[hookpoint] = (latents.cpu(), latents.shape[2])
            del latents

        del router_paths_flat

        log_queue.put(f"Worker {gpu_id}: submitting results for batch {batch_idx}")

        result_queue.put((batch_idx, batch_tokens.cpu(), results))
        del batch_tokens, results

        gc.collect()
        th.cuda.empty_cache()

    log_queue.put(f"Worker {gpu_id}: Done")


GPU_LOG_FILE = "gpu_log.txt"


def _log_worker(log_queue: mp.Queue):
    """Worker that logs messages from the log queue."""
    with open(GPU_LOG_FILE, "w") as f:
        f.write("Logging started\n")
        while True:
            try:
                log = log_queue.get(timeout=60)
            except queue.Empty:
                f.write("No logs after 1 minute\n")
                continue

            if log is None:
                f.write("Received stop signal\n")
                break
            f.write(f"{log}\n")

        f.write("Logging finished\n")


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
    metric: CentroidMetric = CentroidMetric.DOT_PRODUCT,
    metric_p: float = 2.0,
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
        path_projection = CentroidProjection(
            centroids.to(device="cuda", dtype=dtype),
            metric=metric,
            p=metric_p,
        )
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
        buffer_flush_size: int = 131072,  # how many tokens before we flush to disk
        cache_dir: Path | None = None,
    ):
        """
        Initialize the LatentCache.

        Args:
            model: The model to cache latents for.
            hookpoint_to_sparse_encode: Dictionary of sparse encoding functions.
            batch_size: Size of batches for processing.
            log_path: Path to save logging output.
            postprocessor: Router logits postprocessing method to use.
            buffer_flush_size: Number of tokens before flushing to disk.
                Defaults to 131072.
            cache_dir: Directory to store intermediate cache files. If None,
                uses DiskCache.DEFAULT_CACHE_DIR.
        """
        self._init_cache(batch_size, buffer_flush_size, cache_dir, log_path)
        self.model: StandardizedTransformer = model
        self.hookpoint_to_sparse_encode = hookpoint_to_sparse_encode
        self.postprocessor_fn = get_postprocessor(postprocessor)

    def _init_cache(
        self,
        batch_size: int,
        buffer_flush_size: int,
        cache_dir: Path | None,
        log_path: Path | None,
    ):
        """Initialize cache-related attributes (shared by subclasses)."""
        self.batch_size = batch_size
        self.widths = {}
        self.cache = DiskCache(
            filters=None,
            batch_size=batch_size,
            buffer_flush_size=buffer_flush_size,
            cache_dir=cache_dir,
        )
        self.log_path = log_path

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

        # Check if we can resume from existing cache
        max_batch_idx = self.cache.get_max_batch_index()
        start_batch_idx = 0
        if max_batch_idx is not None:
            start_batch_idx = max_batch_idx + 1
            if start_batch_idx < len(token_batches):
                logger.info(
                    f"Resuming from batch {start_batch_idx} (batches 0-{max_batch_idx} already processed)"
                )
            else:
                logger.info(
                    f"All batches already processed (0-{max_batch_idx}), nothing to do"
                )
                self.cache.save()
                return

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()
        for batch_idx, batch in tqdm(
            enumerate(token_batches[start_batch_idx:], start=start_batch_idx),
            total=total_batches,
            desc="Caching latents",
            initial=start_batch_idx,
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

            gc.collect()
            th.cuda.empty_cache()

        logger.info(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()

    def _generate_split_indices(
        self, n_splits: int
    ) -> dict[str, list[tuple[th.Tensor, th.Tensor]]]:
        """
        Generate indices for splitting the latent space.

        Args:
            n_splits: Number of splits to generate.

        Returns:
            list[tuple[int, int]]: list of start and end indices for each split.
        """
        width_splits: dict[str, list[tuple[th.Tensor, th.Tensor]]] = {}

        for hookpoint in self.cache.hookpoints:
            width = self.widths[hookpoint]
            boundaries = th.linspace(0, width, steps=n_splits + 1).long()
            width_splits[hookpoint] = list(
                zip(boundaries[:-1], boundaries[1:] - 1, strict=True)
            )

        return width_splits

    @th.inference_mode()
    def save_splits(self, n_splits: int, save_dir: Path, save_tokens: bool = True):
        """
        Save the cached non-zero latent activations and locations in splits.

        Args:
            n_splits: Number of splits to generate.
            save_dir: Directory to save the splits.
            save_tokens: Whether to save the dataset tokens used to generate the cache.
            Defaults to True.
        """
        width_splits = self._generate_split_indices(n_splits)
        for hookpoint in self.cache.hookpoints:
            # Load all data for this hookpoint in a single disk read
            latent_locations, latent_activations, tokens = (
                self.cache.get_hookpoint_data(hookpoint)
            )
            tokens_np = tokens.numpy()
            split_indices = width_splits[hookpoint]

            latent_indices = latent_locations[:, 2]

            for start, end in split_indices:
                mask = (latent_indices >= start) & (latent_indices <= end)

                masked_activations = latent_activations[mask].half().numpy()

                masked_locations = latent_locations[mask].numpy()

                # Optimization to reduce the max value to enable a smaller dtype
                masked_locations[:, 2] = masked_locations[:, 2] - start.item()

                if (
                    masked_locations[:, 2].max() < 2**16
                    and masked_locations[:, 0].max() < 2**16
                ):
                    masked_locations = masked_locations.astype(np.uint16)
                else:
                    masked_locations = masked_locations.astype(np.uint32)
                    logger.warning(
                        "Increasing the number of splits might reduce the"
                        "memory usage of the cache."
                    )

                hookpoint_dir = save_dir / hookpoint
                hookpoint_dir.mkdir(parents=True, exist_ok=True)

                output_file = hookpoint_dir / f"{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens:
                    split_data["tokens"] = tokens_np

                save_file(split_data, output_file)

    def generate_statistics_cache(self):
        """
        Print statistics (number of dead features, number of single token features)
        to the console.
        """
        assert len(self.widths) > 0, "Widths must be set before generating statistics"
        logger.info("Feature statistics:")

        # Token frequency
        for hookpoint in self.cache.hookpoints:
            width = self.widths[hookpoint]

            logger.info(f"# Hookpoint: {hookpoint}")
            logger.debug(f"# Width: {width}")

            # Load all data for this hookpoint in a single disk read
            latent_locations, latent_activations, tokens = (
                self.cache.get_hookpoint_data(hookpoint)
            )

            generate_statistics_cache(
                tokens,
                latent_locations,
                latent_activations,
                width,
                verbose=True,
            )


class MultiGPULatentPathsCache(LatentPathsCache):
    """Subclass of LatentPathsCache that processes batches across multiple GPUs."""

    RESULT_QUEUE_MAX_SIZE = 10

    def __init__(
        self,
        batch_size: int,
        gpu_ids: list[int],
        model_name: str,
        model_revision: str,
        model_dtype: th.dtype,
        kmeans_path: Path,
        dtype: th.dtype,
        metric: CentroidMetric,
        metric_p: float,
        postprocessor: RouterLogitsPostprocessor,
        top_k: int,
        hf_token: str,
        quantization_config: BitsAndBytesConfig | None,
        log_path: Path | None = None,
        buffer_flush_size: int = 131072,
        cache_dir: Path | None = None,
    ):
        """
        Initialize the multi-GPU cache.

        Args:
            batch_size: Size of batches for processing.
            gpu_ids: List of GPU IDs to use for parallel processing.
            model_name: Name of the model to load.
            model_revision: Model revision/checkpoint.
            model_dtype: Data type for model weights.
            kmeans_path: Path to the kmeans centroids file.
            dtype: Data type for sparse encoding output.
            metric: Centroid distance metric.
            metric_p: P value for distance metric.
            postprocessor: Router logits postprocessing method.
            top_k: Top k paths to cache.
            hf_token: HuggingFace token for model access.
            quantization_config: Optional quantization config.
            log_path: Path to save logging output.
            buffer_flush_size: Number of tokens before flushing to disk.
            cache_dir: Directory to store intermediate cache files.
        """
        self._init_cache(batch_size, buffer_flush_size, cache_dir, log_path)

        # Store config for workers (they load their own models)
        self.gpu_ids = gpu_ids
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_dtype = model_dtype
        self.kmeans_path = kmeans_path
        self.dtype = dtype
        self.metric = metric
        self.metric_p = metric_p
        self.postprocessor = postprocessor
        self.top_k = top_k
        self.hf_token = hf_token
        self.quantization_config = quantization_config

    def run(self, n_tokens: int, tokens: th.Tensor):
        """Run caching using multiple GPUs in parallel."""
        token_batches = self.load_token_batches(n_tokens, tokens)
        total_batches = len(token_batches)

        # Check if we can resume from existing cache
        max_batch_idx = self.cache.get_max_batch_index()
        start_batch_idx = 0
        if max_batch_idx is not None:
            start_batch_idx = max_batch_idx + 1
            if start_batch_idx < len(token_batches):
                logger.info(
                    f"Resuming from batch {start_batch_idx} (batches 0-{max_batch_idx} already processed)"
                )
            else:
                logger.info(
                    f"All batches already processed (0-{max_batch_idx}), nothing to do"
                )
                self.cache.save()
                return

        batches_to_process = total_batches - start_batch_idx
        logger.info(
            f"Multi-GPU caching: {len(self.gpu_ids)} GPUs, {batches_to_process} batches to process (starting from batch {start_batch_idx})"
        )

        ctx = mp.get_context("spawn")
        work_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue(maxsize=self.RESULT_QUEUE_MAX_SIZE)
        log_queue: mp.Queue = ctx.Queue()

        # Fill work queue (skip already-processed batches)
        for batch_idx, batch_tokens in enumerate(
            token_batches[start_batch_idx:], start=start_batch_idx
        ):
            work_queue.put((batch_idx, batch_tokens))

        # Add shutdown signals
        for _ in self.gpu_ids:
            work_queue.put(None)

        # Spawn workers
        workers = [
            ctx.Process(
                target=_gpu_worker,
                name=f"gpu_worker_{gpu_id}",
                args=(
                    gpu_id,
                    work_queue,
                    result_queue,
                    log_queue,
                    self.model_name,
                    self.model_revision,
                    self.model_dtype,
                    self.kmeans_path,
                    self.dtype,
                    self.metric,
                    self.metric_p,
                    self.postprocessor,
                    self.top_k,
                    self.hf_token,
                    self.quantization_config,
                ),
                daemon=True,
            )
            for gpu_id in self.gpu_ids
        ]

        for worker in workers:
            logger.info(f"Starting worker {worker.name}")
            worker.start()

        # Start logging worker
        log_worker = ctx.Process(
            target=_log_worker,
            name="logging_worker",
            args=(log_queue,),
            daemon=True,
        )
        log_worker.start()

        # Collect results
        for _ in tqdm(
            range(batches_to_process),
            total=total_batches,
            desc="Caching (multi-GPU)",
            initial=start_batch_idx,
        ):
            batch_idx, batch_tokens, results = result_queue.get(timeout=300)

            logger.debug(f"Received results for batch {batch_idx}")

            for hookpoint, (latents, width) in results.items():
                self.cache.add(latents, batch_tokens, batch_idx, hookpoint)
                self.widths[hookpoint] = width

        # Wait for workers
        for w in workers:
            w.join(timeout=30)

        log_queue.put(None)
        log_worker.join()

        logger.info(
            f"Total tokens processed: {total_batches * token_batches[0].numel():,}"
        )
        logger.info(
            f"Tokens processed in this run: {batches_to_process * token_batches[0].numel():,}"
        )
        self.cache.save()


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


def populate_cache_multiprocess(
    run_cfg: RunConfig,
    root_dir: Path,
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    top_k: int,
    dtype: th.dtype,
    gpu_ids: list[int],
    model_name: str,
    model_revision: str,
    model_dtype: th.dtype,
    metric: CentroidMetric,
    metric_p: float,
    hf_token: str,
    quantization_config: BitsAndBytesConfig | None,
    postprocessor: RouterLogitsPostprocessor,
) -> None:
    """Populate cache using multiple GPUs."""
    kmeans_path = root_dir / KMEANS_FILENAME
    if not kmeans_path.is_file():
        raise ValueError(f"Kmeans file not found: {kmeans_path}")

    latents_path.mkdir(parents=True, exist_ok=True)
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_and_filter_tokens(tokenizer, cache_cfg, run_cfg)

    cache = MultiGPULatentPathsCache(
        batch_size=cache_cfg.batch_size,
        gpu_ids=gpu_ids,
        model_name=model_name,
        model_revision=model_revision,
        model_dtype=model_dtype,
        kmeans_path=kmeans_path,
        dtype=dtype,
        metric=metric,
        metric_p=metric_p,
        postprocessor=postprocessor,
        top_k=top_k,
        hf_token=hf_token,
        quantization_config=quantization_config,
        log_path=log_path,
    )

    cache.run(n_tokens=cache_cfg.n_tokens, tokens=tokens)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(n_splits=cache_cfg.n_splits, save_dir=latents_path)
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
    batchsize: int = 32,
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
    vllm_num_gpus: int = 1,
    cache_num_gpus: int = 0,
    cache_start_gpu: int = 0,
    verbose: bool = True,
    seed: int = 0,
    hf_token: str = "",
    log_level: str = "INFO",
    device_type: str = "cuda",
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
    metric: str = "dot_product",
    metric_p: float = 2.0,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    metric = CentroidMetric(metric)

    # Handle GPU configuration
    backend = get_backend(device_type)
    total_gpus = backend.device_count() if backend.is_available() else 0

    # Use num_gpus if provided (for backward compatibility), otherwise use vllm_num_gpus
    effective_vllm_gpus = num_gpus if num_gpus is not None else vllm_num_gpus

    # Calculate cache GPU IDs
    if cache_num_gpus <= 0:
        cache_num_gpus = total_gpus

    cache_gpu_ids = list(range(cache_start_gpu, cache_start_gpu + cache_num_gpus))
    cache_gpu_ids = [g for g in cache_gpu_ids if g < total_gpus]

    assert cache_gpu_ids, "No cache GPUs available"

    if total_gpus == 1:
        logger.warning("Only 1 GPU available. Caching and VLLM will share GPU 0.")
        cache_gpu_ids = [0]

    logger.info(f"Running with log level: {log_level}")
    logger.info(
        f"Device allocation: caching on GPUs {cache_gpu_ids}, "
        f"VLLM using {effective_vllm_gpus} GPU(s) starting from device 0"
    )

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

    use_multiprocess = len(cache_gpu_ids) > 1

    # Load tokenizer (always needed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_name, revision=str(model_ckpt), token=hf_token
    )

    hookpoint_to_sparse_encode, top_k = load_hookpoints(
        root_dir, dtype=dtype_torch, metric=metric, metric_p=metric_p
    )
    hookpoints = list(hookpoint_to_sparse_encode.keys())

    # Only load hookpoints if single-GPU mode
    if use_multiprocess:
        assert top_k is not None, "Multi-GPU not supported for SAE experiments"

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
        num_gpus=effective_vllm_gpus,
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
        if top_k is None:
            raise ValueError("top_k cannot be None when populating cache")

        if use_multiprocess:
            logger.info(
                f"Populating cache with {len(hookpoints)} hookpoints using {len(cache_gpu_ids)} GPUs"
            )
            populate_cache_multiprocess(
                run_cfg=run_cfg,
                root_dir=root_dir,
                latents_path=latents_path,
                tokenizer=tokenizer,
                top_k=top_k,
                dtype=dtype_torch,
                gpu_ids=cache_gpu_ids,
                model_name=model_config.hf_name,
                model_revision=str(model_ckpt),
                model_dtype=model_dtype_torch,
                metric=metric,
                metric_p=metric_p,
                hf_token=hf_token,
                quantization_config=quantization_config,
                postprocessor=postprocessor,
            )
        else:
            # Single GPU: load model and use original populate_cache
            cache_device = f"{device_type}:{cache_gpu_ids[0]}"
            logger.info(
                f"Populating cache with {len(nrh)} hookpoints on {cache_device}"
            )
            model = StandardizedTransformer(
                model_config.hf_name,
                check_attn_probs_with_trace=False,
                check_renaming=False,
                revision=str(model_ckpt),
                device_map={"": cache_device},
                quantization_config=quantization_config,
                torch_dtype=model_dtype_torch,
                token=hf_token,
            )
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
            del model
    else:
        logger.debug("No non-redundant hookpoints found, skipping cache population")

    # Clean up
    th.cuda.empty_cache()
    gc.collect()

    # Process cache and run intruder detection (VLLM will use device 0)
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
