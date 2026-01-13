"""
Comprehensive evaluation script for k-means path experiments.

This script:
1. Evaluates a k-means experiment using both path_eval_saebench and eval_intruder
2. Much simpler than eval_all_saes since k-means stores multiple centroids in a single file
"""

from multiprocessing import cpu_count
import sys
import traceback

import arguably
from loguru import logger

from core.device import get_backend
from core.moe import CentroidMetric, RouterLogitsPostprocessor
from exp.eval_intruder import eval_intruder
from exp.path_eval_saebench import path_eval_saebench


def run_path_eval_saebench(
    experiment_name: str,
    model_name: str,
    batchsize: int,
    dtype: str,
    seed: int,
    log_level: str = "INFO",
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
    metric: CentroidMetric = CentroidMetric.DOT_PRODUCT,
    metric_p: float = 2.0,
    num_autointerp_latents: int = 1000,
) -> bool:
    """Run SAEBench evaluation on a k-means path experiment."""
    logger.info(f"Running SAEBench evaluation on {experiment_name}")

    try:
        path_eval_saebench(
            experiment_dir=experiment_name,
            model_name=model_name,
            batchsize=batchsize,
            dtype=dtype,
            seed=seed,
            logs_path=None,
            log_level=log_level,
            skip_autointerp=skip_autointerp,
            skip_sparse_probing=skip_sparse_probing,
            metric=metric,
            metric_p=metric_p,
            num_autointerp_latents=num_autointerp_latents,
        )
        logger.debug(f"✅ SAEBench evaluation completed for {experiment_name}")
        return True
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.exception(
            f"❌ Unexpected error in SAEBench evaluation for {experiment_name}:\n{traceback_str}{exception_str}"
        )
        return False


def run_intruder_eval(
    experiment_name: str,
    model_name: str,
    model_step_ckpt: int | None,
    model_dtype: str,
    dtype: str,
    ctxlen: int,
    load_in_8bit: bool,
    n_tokens: int,
    batchsize: int,
    n_latents: int,
    example_ctx_len: int,
    min_examples: int,
    num_non_activating: int,
    num_examples: int,
    n_quantiles: int,
    explainer_model: str,
    explainer_model_max_len: int,
    explainer_provider: str,
    explainer: str,
    n_splits: int,
    filter_bos: bool,
    pipeline_num_proc: int,
    num_gpus: int,
    vllm_num_gpus: int,
    cache_num_gpus: int,
    cache_start_gpu: int,
    verbose: bool,
    seed: int,
    hf_token: str,
    log_level: str = "INFO",
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
    metric: CentroidMetric = CentroidMetric.DOT_PRODUCT,
    metric_p: float = 2.0,
) -> bool:
    """Run intruder evaluation on a k-means path experiment."""
    logger.info(f"Running intruder evaluation on {experiment_name}")

    try:
        eval_intruder(
            experiment_dir=experiment_name,
            model_name=model_name,
            model_step_ckpt=model_step_ckpt,
            model_dtype=model_dtype,
            dtype=dtype,
            ctxlen=ctxlen,
            load_in_8bit=load_in_8bit,
            n_tokens=n_tokens,
            batchsize=batchsize,
            n_latents=n_latents,
            example_ctx_len=example_ctx_len,
            min_examples=min_examples,
            num_non_activating=num_non_activating,
            num_examples=num_examples,
            n_quantiles=n_quantiles,
            explainer_model=explainer_model,
            explainer_model_max_len=explainer_model_max_len,
            explainer_provider=explainer_provider,
            explainer=explainer,
            n_splits=n_splits,
            filter_bos=filter_bos,
            pipeline_num_proc=pipeline_num_proc,
            num_gpus=num_gpus,
            vllm_num_gpus=vllm_num_gpus,
            cache_num_gpus=cache_num_gpus,
            cache_start_gpu=cache_start_gpu,
            verbose=verbose,
            seed=seed,
            hf_token=hf_token,
            log_level=log_level,
            postprocessor=postprocessor,
            metric=metric,
            metric_p=metric_p,
        )
        logger.debug(f"✅ Intruder evaluation completed for {experiment_name}")
        return True
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(
            f"❌ Unexpected error in intruder evaluation for {experiment_name}:\n{traceback_str}{exception_str}"
        )
        return False


@arguably.command()
def eval_all_paths(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    run_saebench: bool = True,
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
    run_intruder: bool = True,
    saebench_batchsize: int = 64,
    num_autointerp_latents: int = 1000,
    intruder_model_step_ckpt: int | None = None,
    intruder_model_dtype: str = "bf16",
    intruder_ctxlen: int = 256,
    intruder_load_in_8bit: bool = False,
    intruder_n_tokens: int = 10_000_000,
    intruder_batchsize: int = 32,
    intruder_n_latents: int = 1000,
    intruder_example_ctx_len: int = 32,
    intruder_min_examples: int = 200,
    intruder_num_non_activating: int = 50,
    intruder_num_examples: int = 50,
    intruder_n_quantiles: int = 10,
    intruder_explainer_model: str = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    intruder_explainer_model_max_len: int = 5120,
    intruder_explainer_provider: str = "offline",
    intruder_explainer: str = "default",
    intruder_n_splits: int = 5,
    intruder_filter_bos: bool = False,
    intruder_pipeline_num_proc: int = cpu_count() // 2,
    intruder_num_gpus: int | None = None,
    intruder_vllm_num_gpus: int = 1,
    intruder_cache_num_gpus: int = 0,
    intruder_cache_start_gpu: int = 0,
    intruder_verbose: bool = True,
    intruder_hf_token: str = "",
    dtype: str = "bf16",
    seed: int = 0,
    log_level: str = "INFO",
    device_type: str = "cuda",
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
    metric: str = "dot_product",
    metric_p: float = 2.0,
) -> None:
    """
    Evaluate a k-means path experiment using both SAEBench and intruder evaluations.

    Args:
        experiment_dir: Name of the k-means experiment directory (e.g., "kmeans_2024-01-01_00-00-00")
        model_name: Model name to evaluate on
        run_saebench: Whether to run SAEBench evaluation
        run_intruder: Whether to run intruder evaluation
        saebench_batchsize: Batch size for SAEBench evaluation
        intruder_model_step_ckpt: Model checkpoint step for intruder evaluation
        intruder_ctxlen: Context length for intruder evaluation
        intruder_load_in_8bit: Whether to load model in 8-bit for intruder evaluation
        intruder_n_tokens: Number of tokens for intruder evaluation
        intruder_batchsize: Batch size for intruder evaluation
        intruder_n_latents: Number of latents for intruder evaluation
        intruder_example_ctx_len: Example context length for intruder evaluation
        intruder_min_examples: Minimum examples for intruder evaluation
        intruder_num_non_activating: Number of non-activating examples for intruder evaluation
        intruder_num_examples: Number of examples for intruder evaluation
        intruder_n_quantiles: Number of quantiles for intruder evaluation
        intruder_explainer_model: Explainer model for intruder evaluation
        intruder_explainer_model_max_len: Max length for explainer model
        intruder_explainer_provider: Explainer provider for intruder evaluation
        intruder_explainer: Explainer type for intruder evaluation
        intruder_n_splits: Number of splits for intruder evaluation
        intruder_filter_bos: Whether to filter BOS tokens for intruder evaluation
        intruder_pipeline_num_proc: Number of processes for intruder evaluation pipeline
        intruder_num_gpus: Number of GPUs for intruder evaluation (deprecated, use intruder_vllm_num_gpus)
        intruder_vllm_num_gpus: Number of GPUs for VLLM (default: 1, uses device 0)
        intruder_cache_num_gpus: Number of GPUs for caching (default: 0 means all GPUs)
        intruder_cache_start_gpu: Starting GPU index for caching (default: 0)
        intruder_verbose: Whether to use verbose output for intruder evaluation
        intruder_hf_token: HuggingFace token for intruder evaluation
        dtype: Data type for evaluation
        seed: Random seed
        log_level: Logging level
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    metric = CentroidMetric(metric)

    # Set GPU count dynamically based on device type
    if intruder_num_gpus is None:
        backend = get_backend(device_type)
        intruder_num_gpus = backend.device_count() if backend.is_available() else 0

    logger.info(f"Running evaluation for k-means experiment: {experiment_dir}")
    logger.info(
        f"Configuration: run_saebench={run_saebench}, run_intruder={run_intruder}, skip_autointerp={skip_autointerp}, skip_sparse_probing={skip_sparse_probing}"
    )

    success = True

    if run_saebench:
        logger.info("Starting SAEBench evaluation...")
        saebench_success = run_path_eval_saebench(
            experiment_dir,
            model_name,
            saebench_batchsize,
            dtype,
            seed,
            log_level,
            skip_autointerp,
            skip_sparse_probing,
            metric,
            metric_p,
            num_autointerp_latents,
        )
        if not saebench_success:
            logger.error("SAEBench evaluation failed!")
            success = False
        else:
            logger.success("✅ SAEBench evaluation completed successfully")
    else:
        assert not skip_autointerp and not skip_sparse_probing, (
            "Cannot skip autointerp or sparse probing if SAEBench is not run"
        )

    if run_intruder:
        logger.info("Starting intruder evaluation...")
        intruder_success = run_intruder_eval(
            experiment_dir,
            model_name,
            intruder_model_step_ckpt,
            intruder_model_dtype,
            dtype,
            intruder_ctxlen,
            intruder_load_in_8bit,
            intruder_n_tokens,
            intruder_batchsize,
            intruder_n_latents,
            intruder_example_ctx_len,
            intruder_min_examples,
            intruder_num_non_activating,
            intruder_num_examples,
            intruder_n_quantiles,
            intruder_explainer_model,
            intruder_explainer_model_max_len,
            intruder_explainer_provider,
            intruder_explainer,
            intruder_n_splits,
            intruder_filter_bos,
            intruder_pipeline_num_proc,
            intruder_num_gpus,
            intruder_vllm_num_gpus,
            intruder_cache_num_gpus,
            intruder_cache_start_gpu,
            intruder_verbose,
            seed,
            intruder_hf_token,
            log_level,
            postprocessor,
            metric,
            metric_p,
        )
        if not intruder_success:
            logger.error("Intruder evaluation failed!")
            success = False
        else:
            logger.success("✅ Intruder evaluation completed successfully")

    if success:
        logger.success("🎉 All evaluations completed successfully!")
    else:
        logger.error("❌ Some evaluations failed")
        sys.exit(1)


if __name__ == "__main__":
    arguably.run()
