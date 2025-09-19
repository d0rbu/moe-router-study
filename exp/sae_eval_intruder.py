from pathlib import Path

import arguably
from transformers import (
    AutoModel,
    BitsAndBytesConfig,
)

from core.dtype import get_dtype
from core.model import get_model_config
from core.type import assert_type
from delphi.__main__ import populate_cache
from delphi.config import CacheConfig, RunConfig
from delphi.sparse_coders.sparse_model import non_redundant_hookpoints
from exp import OUTPUT_DIR


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
) -> None:
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    dtype = get_dtype(model_dtype)

    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    root_dir = Path(OUTPUT_DIR, experiment_dir)
    base_path = root_dir / "delphi"
    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"

    model = AutoModel.from_pretrained(
        model_config.hf_name,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        torch_dtype=dtype,
        token=hf_token,
    )

    raise NotImplementedError("Implement hookpoint_to_sparse_encode and run_cfg")

    run_cfg = RunConfig(
        max_latents=n_latents,
        cache_cfg=CacheConfig(
            cache_ctx_len=ctxlen,
            batch_size=batchsize,
            n_tokens=n_tokens,
        ),
    )

    nrh = assert_type(
        dict,
        non_redundant_hookpoints(hookpoint_to_sparse_encode, latents_path, False),
    )
    if nrh:
        populate_cache(
            run_cfg,
            model,
            nrh,
            latents_path,
            model.tokenizer,
        )
