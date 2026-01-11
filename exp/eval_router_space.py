"""
Evaluation script for raw router space (without k-means clustering).

This script creates a synthetic experiment directory with identity matrix centroids
and runs eval_all_paths on it to evaluate the raw MoE router output space.
"""

import os
import sys

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor
from exp import OUTPUT_DIR
from exp.eval_all_paths import eval_all_paths
from exp.kmeans import KMEANS_FILENAME, KMEANS_TYPE, METADATA_FILENAME


@arguably.command()
def eval_router_space(
    *,
    model_name: str = "olmoe-i",
    run_saebench: bool = True,
    skip_autointerp: bool = False,
    run_intruder: bool = True,
    saebench_batchsize: int = 64,
    intruder_batchsize: int = 32,
    intruder_explainer_model: str = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    intruder_n_tokens: int = 10_000_000,
    intruder_vllm_num_gpus: int = 3,
    intruder_cache_device_idx: int = 1,
    dtype: str = "bf16",
    seed: int = 0,
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.IDENTITY,
    log_level: str = "INFO",
) -> None:
    """
    Evaluate raw router space by creating synthetic experiment with identity centroids.

    Args:
        model_name: Model name to evaluate on
        run_saebench: Whether to run SAEBench evaluation
        skip_autointerp: Whether to skip autointerp evaluation
        run_intruder: Whether to run intruder evaluation
        saebench_batchsize: Batch size for SAEBench evaluation
        intruder_batchsize: Batch size for intruder evaluation
        intruder_n_tokens: Number of tokens for intruder evaluation
        intruder_vllm_num_gpus: Number of GPUs for VLLM (default: 1, uses device 0)
        intruder_cache_device_idx: Device index for caching model (default: 1, reserves 0 for VLLM)
        dtype: Data type for evaluation
        seed: Random seed
        log_level: Logging level
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    th_dtype = get_dtype(dtype)
    logger.info(f"Evaluating raw router space for model: {model_name}")

    # Load model to get architecture info
    model_config = get_model_config(model_name)
    hf_name = model_config.hf_name

    logger.info(f"Loading model {hf_name} to extract architecture info")
    model = StandardizedTransformer(
        hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        device_map="cuda",
        torch_dtype=th_dtype,
    )

    # Get architecture info
    num_layers = len(model.layers_with_routers)
    model_config = model.config
    num_experts = model_config.num_experts

    assert isinstance(num_experts, int), (
        f"num_experts must be an integer, got {type(num_experts)}"
    )

    activation_dim = num_layers * num_experts
    top_k = model_config.num_experts_per_tok

    assert isinstance(top_k, int), f"top_k must be an integer, got {type(top_k)}"

    logger.info(
        f"Architecture: {num_layers} layers, {num_experts} experts, top_k={top_k}"
    )

    # Clean up model
    del model
    th.cuda.empty_cache() if th.cuda.is_available() else None

    # Create synthetic experiment directory
    experiment_name = f"router_space_{model_name}"
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create identity matrix as centroids (one centroid set with identity matrix)
    identity_centroids = th.eye(activation_dim, dtype=th_dtype)
    centroids_data = {
        "centroids": [identity_centroids],  # List with single identity matrix
        "top_k": top_k,
        "losses": th.tensor([0.0]),  # No clustering loss for identity
    }

    # Save centroids file
    centroids_path = os.path.join(experiment_dir, KMEANS_FILENAME)
    th.save(centroids_data, centroids_path)
    logger.info(f"Created synthetic centroids file: {centroids_path}")

    # Create metadata file
    metadata = {
        "type": KMEANS_TYPE,
        "model_name": model_name,
        "activation_dim": activation_dim,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "k": (activation_dim,),
        "kmeans_experiment_name": experiment_name,
        "postprocessor": postprocessor.value,
    }
    metadata_path = os.path.join(experiment_dir, METADATA_FILENAME)
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)
    logger.info(f"Created metadata file: {metadata_path}")

    # Run eval_all_paths on synthetic experiment
    logger.info("Running eval_all_paths on synthetic router space experiment")
    eval_all_paths(
        experiment_dir=experiment_name,
        model_name=model_name,
        run_saebench=run_saebench,
        skip_autointerp=skip_autointerp,
        skip_sparse_probing=True,
        run_intruder=run_intruder,
        saebench_batchsize=saebench_batchsize,
        intruder_batchsize=intruder_batchsize,
        intruder_explainer_model=intruder_explainer_model,
        intruder_n_tokens=intruder_n_tokens,
        intruder_vllm_num_gpus=intruder_vllm_num_gpus,
        intruder_cache_start_gpu=intruder_cache_device_idx,
        dtype=dtype,
        seed=seed,
        log_level=log_level,
    )

    logger.success(f"🎉 Router space evaluation complete for {model_name}!")


if __name__ == "__main__":
    arguably.run()
