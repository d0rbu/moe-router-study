"""
SAEBench evaluation for raw model activations.

This script evaluates raw model activations using SAEBench tasks (autointerp and sparse probing).
Based on exp/path_eval_saebench.py, exp/autointerp_saebench.py, and exp/sparse_probing_saebench.py.
"""

import os
import sys

import arguably
from dotenv import load_dotenv
from loguru import logger
from nnterp import StandardizedTransformer
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    output_folders as EVAL_DIRS,
)
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.sae_bench_utils import general_utils
import torch as th

from core.dtype import get_dtype
from core.model import get_model_config
from exp import MODEL_DIRNAME, OUTPUT_DIR
from exp.autointerp_saebench import Paths
from exp.autointerp_saebench import run_eval as run_autointerp_eval
from exp.get_activations import ActivationKeys
from exp.sparse_probing_saebench import run_eval as run_sparse_probing_eval

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@arguably.command()
def eval_raw_activations_saebench(
    *,
    model_name: str = "olmoe-i",
    activation_key: ActivationKeys = ActivationKeys.LAYER_OUTPUT,
    layers: set[int] | None = None,
    batchsize: int = 512,
    dtype: str = "bfloat16",
    seed: int = 0,
    logs_path: str | None = None,
    log_level: str = "INFO",
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
) -> None:
    """
    Evaluate raw model activations on SAEBench tasks.

    This creates an identity matrix as the "paths" so that each activation dimension
    is treated as a separate feature, effectively evaluating the raw model outputs.

    Args:
        model_name: Model name to evaluate
        activation_key: Type of activation (ActivationKeys enum)
        layers: Set of layer indices to evaluate (default: all layers)
        batchsize: Batch size for evaluation
        dtype: Data type for tensors
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

    # Get model configuration and load model to determine activation dimensions
    model_config = get_model_config(model_name)
    logger.trace(f"Model config: {model_config}")

    # Load model to get hidden size and number of layers
    hf_name = model_config.hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)
    path = local_path if os.path.exists(local_path) else hf_name

    logger.info(f"Loading model from {path} to extract architecture info")
    model = StandardizedTransformer(
        path,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        device_map=device,
        torch_dtype=th_dtype,
    )

    # Set default layers if None or empty
    if layers is None or len(layers) == 0:
        num_layers = len(model.layers_with_routers)
        layers = set(range(num_layers))
    
    layers_sorted = sorted(layers)

    # Get the activation dimension (hidden size) from the model config
    activation_dim = model.model.config.hidden_size
    # Total dimension across all layers
    total_activation_dim = activation_dim * len(layers_sorted)
    
    # Clean up model to free memory
    del model
    th.cuda.empty_cache() if th.cuda.is_available() else None

    logger.info(
        f"Model: {model_name}, layers: {layers_sorted}, activation_key: {activation_key.value}, activation_dim per layer: {activation_dim}, total_dim: {total_activation_dim}"
    )

    # Create identity matrix for raw activations
    # This means each activation dimension is treated as a separate feature
    identity_matrix = th.eye(total_activation_dim, dtype=th_dtype, device=device)

    # We'll use top_k=1 to preserve sparsity (though raw activations aren't sparse)
    top_k = 1

    logger.info(f"Created identity matrix of shape {identity_matrix.shape}")

    # Create a Paths object for the identity mapping
    # Note: For raw activations, the Paths interface is a bit of a misnomer,
    # but we reuse it for compatibility with the evaluation framework
    hookpoint = f"raw_{activation_key.value}_layers_{'_'.join(map(str, layers_sorted))}"
    paths = Paths(
        data=identity_matrix,
        top_k=top_k,
        name=f"{hookpoint}_{model_name}",
        metadata={
            "type": "raw_activations",
            "model_name": model_name,
            "activation_key": activation_key.value,
            "layers": layers_sorted,
            "activation_dim": activation_dim,
            "total_activation_dim": total_activation_dim,
            "top_k": top_k,
        },
    )
    paths_set = [paths]

    logger.trace(f"Using paths set: len={len(paths_set)}")

    # Create output directory for raw activations results
    experiment_name = f"{hookpoint}_{model_name}"
    experiment_path = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    logger.trace(f"Using experiment path: {experiment_path}")

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
            selected_paths_set=paths_set,
            device=device,
            api_key=OPENAI_API_KEY,
            output_path=autointerp_eval_dir,
            force_rerun=False,
            save_logs_path=logs_path,
            artifacts_path=os.path.join(experiment_path, "artifacts"),
            log_level=log_level,
        )

        logger.info("Autointerp evaluation complete")

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
            selected_paths_set=paths_set,
            device=device,
            output_path=sparse_probing_eval_dir,
            force_rerun=False,
            clean_up_activations=False,
            save_activations=True,
            artifacts_path=os.path.join(experiment_path, "artifacts"),
            log_level=log_level,
        )

        logger.info("Sparse probing evaluation complete")

    logger.success("done :)")


if __name__ == "__main__":
    arguably.run()