"""
Main evaluation script for raw model activations.

This script orchestrates both SAEBench and intruder evaluations for raw model activations.
"""

import sys
import traceback

import arguably
from loguru import logger

from core.device import DeviceType, get_backend
from exp.eval_raw_activations_intruder import eval_raw_activations_intruder
from exp.eval_raw_activations_saebench import eval_raw_activations_saebench
from exp.get_activations import ActivationKeys


def run_raw_activations_eval_saebench(
    model_name: str,
    activation_key: ActivationKeys,
    layers: set[int] | None,
    batchsize: int,
    dtype: str,
    seed: int,
    log_level: str = "INFO",
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
) -> bool:
    """Run SAEBench evaluation on raw activations."""
    logger.info(
        f"Running SAEBench evaluation on raw activations for {model_name} {activation_key.value} layers {sorted(layers) if layers else 'all'}"
    )

    try:
        eval_raw_activations_saebench(
            model_name=model_name,
            activation_key=activation_key,
            layers=layers,
            batchsize=batchsize,
            dtype=dtype,
            seed=seed,
            logs_path=None,
            log_level=log_level,
            skip_autointerp=skip_autointerp,
            skip_sparse_probing=skip_sparse_probing,
        )
        logger.debug(f"✅ SAEBench evaluation completed for raw activations")
        return True
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.exception(
            f"❌ Unexpected error in SAEBench evaluation for raw activations:\n{traceback_str}{exception_str}"
        )
        return False


def run_raw_activations_eval_intruder(
    model_name: str,
    activation_key: ActivationKeys,
    layers: set[int] | None,
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
    filter_bos: bool,
    pipeline_num_proc: int,
    num_gpus: int,
    verbose: bool,
    seed: int,
    hf_token: str,
    log_level: str = "INFO",
    device_type: DeviceType = "cuda",
) -> bool:
    """Run intruder evaluation on raw activations."""
    logger.info(
        f"Running intruder evaluation on raw activations for {model_name} {activation_key.value} layers {sorted(layers) if layers else 'all'}"
    )

    try:
        eval_raw_activations_intruder(
            model_name=model_name,
            activation_key=activation_key,
            layers=layers,
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
            filter_bos=filter_bos,
            pipeline_num_proc=pipeline_num_proc,
            num_gpus=num_gpus,
            verbose=verbose,
            seed=seed,
            hf_token=hf_token,
            log_level=log_level,
            device_type=device_type,
        )
        logger.debug(f"✅ Intruder evaluation completed for raw activations")
        return True
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(
            f"❌ Unexpected error in intruder evaluation for raw activations:\n{traceback_str}{exception_str}"
        )
        return False


@arguably.command()
def eval_raw_activations(
    *,
    model_name: str = "olmoe-i",
    activation_key: ActivationKeys = ActivationKeys.LAYER_OUTPUT,
    layers: set[int] | None = None,
    run_saebench: bool = True,
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
    run_intruder: bool = True,
    saebench_batchsize: int = 512,
    intruder_model_step_ckpt: int | None = None,
    intruder_model_dtype: str = "bf16",
    intruder_ctxlen: int = 256,
    intruder_load_in_8bit: bool = False,
    intruder_n_tokens: int = 10_000_000,
    intruder_batchsize: int = 8,
    intruder_n_latents: int = 1000,
    intruder_example_ctx_len: int = 32,
    intruder_min_examples: int = 200,
    intruder_num_non_activating: int = 50,
    intruder_num_examples: int = 50,
    intruder_n_quantiles: int = 10,
    intruder_explainer_model: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    intruder_explainer_model_max_len: int = 5120,
    intruder_explainer_provider: str = "offline",
    intruder_explainer: str = "default",
    intruder_filter_bos: bool = False,
    intruder_pipeline_num_proc: int = 8,
    intruder_num_gpus: int | None = None,
    intruder_verbose: bool = True,
    intruder_hf_token: str = "",
    dtype: str = "bf16",
    seed: int = 0,
    log_level: str = "INFO",
    device_type: DeviceType = "cuda",
) -> None:
    """
    Evaluate raw model activations (without SAE encoding).

    This evaluates the raw activations from the model's residual stream by using
    an identity function instead of SAE encoding. Each activation dimension is
    treated as a separate feature for evaluation.

    Args:
        model_name: Model name to evaluate on
        activation_key: Type of activation (ActivationKeys enum)
        layers: Set of layer indices to evaluate (default: all layers)
        run_saebench: Whether to run SAEBench evaluation
        skip_autointerp: Whether to skip autointerp evaluation
        skip_sparse_probing: Whether to skip sparse probing evaluation
        run_intruder: Whether to run intruder evaluation
        saebench_batchsize: Batch size for SAEBench evaluation
        intruder_*: Parameters for intruder evaluation
        dtype: Data type for evaluation
        seed: Random seed
        log_level: Logging level
        device_type: Device type (cuda, xpu, etc.)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Set GPU count dynamically based on device type
    if intruder_num_gpus is None:
        backend = get_backend(device_type)
        intruder_num_gpus = backend.device_count() if backend.is_available() else 0

    logger.info(
        f"Running evaluation for raw activations on model: {model_name}, activation_key: {activation_key.value}, layers: {sorted(layers) if layers else 'all'}"
    )
    logger.info(
        f"Configuration: run_saebench={run_saebench}, run_intruder={run_intruder}, skip_autointerp={skip_autointerp}, skip_sparse_probing={skip_sparse_probing}"
    )

    success = True

    if run_saebench:
        logger.info("Starting SAEBench evaluation...")
        saebench_success = run_raw_activations_eval_saebench(
            model_name,
            activation_key,
            layers,
            saebench_batchsize,
            dtype,
            seed,
            log_level,
            skip_autointerp,
            skip_sparse_probing,
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
        intruder_success = run_raw_activations_eval_intruder(
            model_name,
            activation_key,
            layers,
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
            intruder_filter_bos,
            intruder_pipeline_num_proc,
            intruder_num_gpus,
            intruder_verbose,
            seed,
            intruder_hf_token,
            log_level,
            device_type,
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