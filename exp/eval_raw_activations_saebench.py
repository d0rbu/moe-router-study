"""
Main evaluation script for raw model activations.

This script runs both SAEBench evaluations (autointerp and sparse probing) on raw model
activations from specified layers and activation types. It orchestrates both evaluation
types in a single script.
"""

import os
import sys

import arguably
from dotenv import load_dotenv
from loguru import logger
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    output_folders as EVAL_DIRS,
)
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.sae_bench_utils import general_utils
from transformers import AutoConfig

from core.dtype import get_dtype
from core.model import get_model_config
from exp import OUTPUT_DIR
from exp.eval_raw_activations_autointerp import run_eval as run_autointerp_eval
from exp.eval_raw_activations_sparse_probing import run_eval as run_sparse_probing_eval
from exp.get_activations import ActivationKeys

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@arguably.command()
def eval_raw_activations(
    *,
    model_name: str = "olmoe-i",
    activation_key: ActivationKeys = ActivationKeys.LAYER_OUTPUT,
    layers: list[int] | None = None,
    batchsize: int = 512,
    dtype: str = "bfloat16",
    seed: int = 0,
    logs_path: str | None = None,
    log_level: str = "INFO",
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
) -> None:
    """
    Evaluate raw model activations using SAEBench evaluations.

    Args:
        model_name: Model name to evaluate
        activation_key: Type of activation (layer_output, attn_output, mlp_output)
        layers: List of layer indices to evaluate (if None, uses all layers)
        batchsize: Batch size for evaluation
        dtype: Data type for evaluation
        seed: Random seed
        logs_path: Path to save logs
        log_level: Logging level
        skip_autointerp: Skip autointerp evaluation
        skip_sparse_probing: Skip sparse probing evaluation
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running with log level: {log_level}")

    th_dtype = get_dtype(dtype)
    str_dtype = th_dtype.__str__().split(".")[-1]
    logger.trace(f"Using dtype: {str_dtype}")

    device = general_utils.setup_environment()
    logger.trace(f"Using device: {device}")

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

    assert not skip_autointerp or not skip_sparse_probing, (
        "Cannot skip both autointerp and sparse probing"
    )

    logger.info(
        f"Evaluating raw activations: {activation_key} from layers {layers_sorted} on {model_name}"
    )

    # Create experiment name
    layers_str = "_".join(map(str, layers_sorted))
    experiment_name = f"raw_{activation_key}_layers_{layers_str}_{model_name}"
    experiment_path = os.path.join(OUTPUT_DIR, experiment_name)
    artifacts_path = os.path.join(experiment_path, "artifacts")

    # Run autointerp
    if not skip_autointerp:
        autointerp_eval_dir = EVAL_DIRS["autointerp"]
        autointerp_eval_dir = os.path.join(OUTPUT_DIR, autointerp_eval_dir)
        logger.trace(f"Running autointerp evaluation in {autointerp_eval_dir}")
        run_autointerp_eval(
            config=AutoInterpEvalConfig(
                model_name=model_name,
                random_seed=seed,
                llm_batch_size=batchsize,
                llm_dtype=str_dtype,
            ),
            activation_key=activation_key,
            layers=layers_sorted,
            device=device,
            api_key=OPENAI_API_KEY,
            output_path=autointerp_eval_dir,
            force_rerun=False,
            save_logs_path=logs_path,
            artifacts_path=artifacts_path,
            log_level=log_level,
        )

        logger.info("Autointerp evaluation complete")

    # Run sparse probing
    if not skip_sparse_probing:
        sparse_probing_eval_dir = EVAL_DIRS["sparse_probing"]
        sparse_probing_eval_dir = os.path.join(OUTPUT_DIR, sparse_probing_eval_dir)
        logger.trace(f"Running sparse probing evaluation in {sparse_probing_eval_dir}")
        run_sparse_probing_eval(
            config=SparseProbingEvalConfig(
                model_name=model_name,
                random_seed=seed,
                llm_dtype=str_dtype,
            ),
            activation_key=activation_key,
            layers=layers_sorted,
            device=device,
            output_path=sparse_probing_eval_dir,
            force_rerun=False,
            clean_up_activations=False,
            save_activations=True,
            artifacts_path=artifacts_path,
            log_level=log_level,
        )

        logger.info("Sparse probing evaluation complete")

    logger.success("done :)")


if __name__ == "__main__":
    arguably.run()
