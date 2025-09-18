import os

import arguably
from sparsify.data import chunk_and_tokenize
from transformers import AutoModel, AutoTokenizer

from core.data import smollm2_small
from core.dtype import get_dtype
from core.model import get_model_config
from delphi.__main__ import load_artifacts
from delphi.clients import Offline
from delphi.config import RunConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer, NoOpExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import IntruderScorer
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import assert_type, load_tokenized_data
from exp import MODEL_DIRNAME, OUTPUT_DIR


@arguably.command()
def main(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    model_dtype: str = "f32",
    ctxlen: int = 256,
    num_tokens_to_evaluate_on: int = 10_000_000,
    batchsize: int = 8,
    seed: int = 0,
) -> None:
    model_config = get_model_config(model_name)
    dtype = get_dtype(model_dtype)

    hf_name = model_config.hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)

    path = local_path if os.path.exists(local_path) else hf_name

    tokenizer = AutoTokenizer.from_pretrained(path)
    data = smollm2_small()
    tokens = chunk_and_tokenize(
        data, tokenizer, max_seq_len=ctxlen, text_key="raw_content"
    )["input_ids"]

    cache = LatentCache(model, submodule_dict, batch_size=batchsize)

    cache.run(n_tokens=num_tokens_to_evaluate_on, tokens=tokens)

    experiment_dir_path = os.path.join(OUTPUT_DIR, experiment_dir)
    sae_locations = os.listdir(experiment_dir_path)

    model = AutoModel.from_pretrained(
        model_config.hf_name
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    run_cfg = RunConfig()

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )
