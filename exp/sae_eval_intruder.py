from collections.abc import Callable
import json
from pathlib import Path
import sys

import arguably
from dictionary_learning.utils import load_dictionary
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
from transformers import (
    BitsAndBytesConfig,
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
from exp.get_activations import ActivationKeys


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: th.Tensor | None,
) -> None:
    raise NotImplementedError("Implement process_cache")


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

    hookpoint_to_sparse_encode = load_hookpoints_and_saes(root_dir)
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
