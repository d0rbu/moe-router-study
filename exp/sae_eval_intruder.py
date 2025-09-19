from collections.abc import Callable
import json
from pathlib import Path
import sys

import arguably
from loguru import logger
import torch as th
from transformers import (
    AutoModel,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from core.dtype import get_dtype
from core.model import get_model_config
from core.type import assert_type
from delphi.__main__ import populate_cache
from delphi.config import CacheConfig, RunConfig
from delphi.log.result_analysis import log_results
from delphi.sparse_coders.sparse_model import non_redundant_hookpoints
from exp import OUTPUT_DIR


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: th.Tensor | None,
) -> None:
    raise NotImplementedError("Implement process_cache")


def load_hookpoints_and_saes(
    model: PreTrainedModel, sae_base_path: Path
) -> dict[str, Callable]:
    raise NotImplementedError("Implement load_hookpoints_and_saes")


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
    expansion_factor: int = 16,
    k: int = 160,
    layer: int = 7,
    architecture: str = "batchtopk",
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

    matching_sae_dirpaths = []
    for sae_dirpath in root_dir.iterdir():
        if not sae_dirpath.is_dir():
            continue

        config_path = sae_dirpath / "config.json"
        if not config_path.is_file():
            continue

        with open(config_path) as f:
            config = json.load(f)

        if config["expansion_factor"] != expansion_factor:
            continue

        if config["k"] != k:
            continue

        if config["layer"] != layer:
            continue

        if config["architecture"] != architecture:
            continue

        matching_sae_dirpaths.append(sae_dirpath)

    if not matching_sae_dirpaths:
        raise ValueError(
            "No matching SAE directory found for params:\n"
            f"expansion_factor {expansion_factor}\n"
            f"k {k}\n"
            f"layer {layer}\n"
            f"architecture {architecture}"
        )

    sae_base_path = matching_sae_dirpaths[0]

    if len(matching_sae_dirpaths) > 1:
        logger.warning(
            "Multiple matching SAE directories found for params:\n"
            f"expansion_factor {expansion_factor}\n"
            f"k {k}\n"
            f"layer {layer}\n"
            f"architecture {architecture}\n"
            f"Using the first matching SAE directory: {sae_base_path}"
        )

    base_path = sae_base_path / "delphi"
    latents_path = base_path / "latents"
    scores_path = base_path / "scores"
    visualize_path = base_path / "visualize"

    model = AutoModel.from_pretrained(
        model_config.hf_name,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        torch_dtype=dtype,
        token=hf_token,
    )
    tokenizer = model.tokenizer

    hookpoint_to_sparse_encode = load_hookpoints_and_saes(model, sae_base_path)
    hookpoints = list(hookpoint_to_sparse_encode.keys())

    latent_range = th.arange(n_latents) if n_latents else None

    run_cfg = RunConfig(
        max_latents=n_latents,
        cache_cfg=CacheConfig(
            cache_ctx_len=ctxlen,
            batch_size=batchsize,
            n_tokens=n_tokens,
        ),
    )

    raise NotImplementedError("Implement run_cfg")

    nrh = assert_type(
        dict,
        non_redundant_hookpoints(hookpoints, latents_path, overwrite=False),
    )
    if nrh:
        populate_cache(
            run_cfg,
            model,
            nrh,
            latents_path,
            tokenizer,
            transcode=False,
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
