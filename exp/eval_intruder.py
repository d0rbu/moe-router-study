from collections.abc import Callable
from functools import partial
import json
from multiprocessing import cpu_count
from pathlib import Path
import sys

import arguably
from dictionary_learning.utils import load_dictionary
from loguru import logger
from nnterp import StandardizedTransformer
import orjson
import torch as th
from transformers import (
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from core.type import assert_type
from delphi.__main__ import populate_cache as sae_populate_cache
from delphi.clients import Offline
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.latents import LatentCache, LatentDataset, LatentRecord
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline
from delphi.scorers.classifier.intruder import IntruderScorer
from delphi.scorers.scorer import ScorerResult
from delphi.sparse_coders.sparse_model import non_redundant_hookpoints
from delphi.utils import load_tokenized_data
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
    latent_dict = dict.fromkeys(hookpoints, latent_range) if latent_range else None

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
        temperature=run_cfg.temperature,
        cot=run_cfg.cot,
        type=run_cfg.intruder_type,
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
    ActivationKeys.MLP_OUTPUT: "mlp_outputs.{{layer}}",
    ActivationKeys.ROUTER_LOGITS: "routers_output.{{layer}}",
    ActivationKeys.ATTN_OUTPUT: "attentions_output.{{layer}}",
    ActivationKeys.LAYER_OUTPUT: "layers_output.{{layer}}",
}


def load_hookpoints_and_saes(
    sae_base_path: Path,
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

        ae, _ = load_dictionary(sae_dirpath, device="cuda")
        layer = config.get("layer")
        if layer is None:
            raise ValueError(f"Layer is not set in the config for SAE at {sae_dirpath}")

        layer = int(layer)

        # i stored the activation key in the submodule_name field...
        activation_key = config.get("submodule_name")
        if activation_key is None:
            raise ValueError(
                f"Submodule name is not set in the config for SAE at {sae_dirpath}"
            )

        hookpoint = ACTIVATION_KEYS_TO_HOOKPOINT[activation_key].format(layer=layer)

        hookpoints_to_saes[hookpoint] = ae.encode

    return hookpoints_to_saes


def load_hookpoints(
    root_dir: Path,
) -> dict[str, Callable[[th.Tensor], th.Tensor]]:
    """
    Loads the hookpoints from the config file.
    """
    path_config_path = root_dir / "config.yaml"
    if not path_config_path.is_file():
        # this is a sae experiment, not paths
        return load_hookpoints_and_saes(root_dir)

    paths_path = root_dir / KMEANS_FILENAME
    if not paths_path.is_file():
        raise ValueError(f"Paths file not found at {paths_path}")

    with open(paths_path, "rb") as f:
        data = th.load(f)

    return dict.fromkeys(data["hookpoints"], None)


def populate_cache(
    run_cfg: RunConfig,
    model: StandardizedTransformer,
    hookpoint_to_sparse_encode: dict[str, Callable],
    root_dir: Path,
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> None:
    """
    Populates an on-disk cache in `latents_path` with latent activations.
    """
    path_config_path = root_dir / "config.yaml"
    if not path_config_path.is_file():
        # this is a sae experiment, not paths
        return sae_populate_cache(
            run_cfg,
            model,
            hookpoint_to_sparse_encode,
            latents_path,
            tokenizer,
            transcode=False,
        )

    with open(path_config_path) as f:
        config = yaml.safe_load(f)

    paths_path = root_dir / KMEANS_FILENAME
    assert paths_path.is_file(), f"Paths file not found at {paths_path}"
    with open(paths_path) as f:
        paths = th.load(f)

    centroid_sets: list[th.Tensor] = paths["centroids"]
    top_k: int = paths["top_k"]

    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
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
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~(flattened_tokens == tokenizer.bos_token_id)
            masked_tokens = flattened_tokens[mask]

            num_non_bos_tokens = masked_tokens.shape[0]
            extra_tokens = num_non_bos_tokens % cache_cfg.cache_ctx_len

            if extra_tokens > 0:
                print(
                    f"Warning: {extra_tokens} extra tokens after BOS filtering, truncating to {num_non_bos_tokens - extra_tokens}"
                )
                truncated_tokens = masked_tokens[:-extra_tokens]
                tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)
            else:
                tokens = masked_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=False,
        log_path=log_path,
    )

    for centroid_set in centroid_sets:
        cache.run(cache_cfg.n_tokens, centroid_set, tokens)

    raise NotImplementedError("Implement populate_cache for paths")


@arguably.command()
def main(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "f32",
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
    num_gpus: int = th.cuda.device_count(),
    verbose: bool = True,
    seed: int = 0,
    hf_token: str = "",
    log_level: str = "INFO",
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    dtype = get_dtype(model_dtype)

    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    root_dir = Path(OUTPUT_DIR, experiment_dir)
    base_path = root_dir / "delphi"
    latents_path = base_path / "latents"
    scores_path = base_path / "scores"
    visualize_path = base_path / "visualize"

    th.manual_seed(seed)

    model = StandardizedTransformer(
        model_config.hf_name,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        torch_dtype=dtype,
        token=hf_token,
    )
    tokenizer = model.tokenizer

    hookpoint_to_sparse_encode = load_hookpoints(root_dir)
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
        sparse_model=root_dir,
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
        dict,
        non_redundant_hookpoints(
            hookpoint_to_sparse_encode, latents_path, overwrite=False
        ),
    )
    if nrh:
        populate_cache(
            run_cfg,
            model,
            nrh,
            root_dir,
            latents_path,
            tokenizer,
        )

    del model, hookpoint_to_sparse_encode

    nrh = assert_type(
        list,
        non_redundant_hookpoints(hookpoints, scores_path, overwrite=False),
    )
    if nrh:
        await process_cache(
            run_cfg,
            latents_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range,
        )

    if run_cfg.verbose:
        log_results(scores_path, visualize_path, run_cfg.hookpoints, run_cfg.scorers)
