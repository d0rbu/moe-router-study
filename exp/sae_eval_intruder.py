import os

import arguably
from sparsify.data import chunk_and_tokenize
from transformers import AutoModel, BitsAndBytesConfig

from core.data import smollm2_small
from core.dtype import get_dtype
from core.model import get_model_config
from delphi.config import RunConfig
from delphi.latents import LatentCache
from delphi.sparse_coders import load_hooks_sparse_coders
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
    num_tokens_to_evaluate_on: int = 10_000_000,
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

    model = AutoModel.from_pretrained(
        model_config.hf_name,
        revision=model_config.revision,
        device_map={"": "cuda"},
        quantization_config=quantization_config,
        torch_dtype=dtype,
        token=hf_token,
    )

    data = smollm2_small()
    tokens = chunk_and_tokenize(
        data, model.tokenizer, max_seq_len=ctxlen, text_key="raw_content"
    )["input_ids"]

    cache = LatentCache(model, submodule_dict, batch_size=batchsize)

    cache.run(n_tokens=num_tokens_to_evaluate_on, tokens=tokens)

    experiment_dir_path = os.path.join(OUTPUT_DIR, experiment_dir)
    sae_locations = os.listdir(experiment_dir_path)

    run_cfg = RunConfig(max_latents=n_latents)

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )
