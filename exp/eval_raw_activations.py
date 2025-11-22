"""
Unified evaluation script for raw model activations.

This script runs all three evaluations (intruder, autointerp, and sparse probing) on raw model
activations from specified layers and activation types. It orchestrates all evaluation
types in a single script.
"""

from multiprocessing import cpu_count
import os
import subprocess
import sys
import traceback

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
from exp.eval_raw_activations_saebench import run_eval as run_autointerp_eval
from exp.eval_raw_activations_sparse_probing import run_eval as run_sparse_probing_eval
from exp.get_activations import ActivationKeys

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def run_intruder_eval(
    activation_key: ActivationKeys,
    layers: list[int],
    model_name: str,
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
    num_gpus: int | None,
    verbose: bool,
    seed: int,
    hf_token: str,
    log_level: str,
) -> bool:
    """Run intruder evaluation on raw activations."""
    layers_str = "_".join(map(str, sorted(layers)))
    logger.info(f"Running intruder evaluation on raw_{activation_key}_layers_{layers_str}")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "exp.eval_raw_activations_intruder",
        "--model-name",
        model_name,
        "--activation-key",
        str(activation_key),
        "--model-dtype",
        model_dtype,
        "--dtype",
        dtype,
        "--ctxlen",
        str(ctxlen),
        "--n-tokens",
        str(n_tokens),
        "--batchsize",
        str(batchsize),
        "--n-latents",
        str(n_latents),
        "--example-ctx-len",
        str(example_ctx_len),
        "--min-examples",
        str(min_examples),
        "--num-non-activating",
        str(num_non_activating),
        "--num-examples",
        str(num_examples),
        "--n-quantiles",
        str(n_quantiles),
        "--explainer-model",
        explainer_model,
        "--explainer-model-max-len",
        str(explainer_model_max_len),
        "--explainer-provider",
        explainer_provider,
        "--explainer",
        explainer,
        "--pipeline-num-proc",
        str(pipeline_num_proc),
        "--seed",
        str(seed),
        "--log-level",
        log_level,
    ]

    # Add layers
    for layer in layers:
        cmd.extend(["--layers", str(layer)])

    # Add optional flags
    if load_in_8bit:
        cmd.append("--load-in-8bit")
    if filter_bos:
        cmd.append("--filter-bos")
    if verbose:
        cmd.append("--verbose")
    if num_gpus is not None:
        cmd.extend(["--num-gpus", str(num_gpus)])
    if hf_token:
        cmd.extend(["--hf-token", hf_token])

    logger.debug(f"🔧 Running command: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=7200,  # 2 hour timeout
        )
        logger.debug("✅ Intruder evaluation completed")
        return True
    except subprocess.TimeoutExpired:
        logger.error("❌ Intruder evaluation timed out (2 hour limit)")
        return False
    except subprocess.CalledProcessError as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        logger.error("❌ Intruder evaluation failed")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Return code: {exception.returncode}")
        return False
    except Exception as exception:
        traceback_lines = traceback.format_tb(exception.__traceback__)
        traceback_str = "".join(traceback_lines)
        exception_str = str(exception)
        logger.error(
            f"❌ Unexpected error in intruder evaluation:\n{traceback_str}{exception_str}"
        )
        return False


@arguably.command()
def eval_raw_activations(
    *,
    model_name: str = "olmoe-i",
    activation_key: ActivationKeys = ActivationKeys.LAYER_OUTPUT,
    layers: list[int] | None = None,
    model_dtype: str = "bf16",
    dtype: str = "bf16",
    ctxlen: int = 256,
    load_in_8bit: bool = False,
    n_tokens: int = 10_000_000,
    batchsize: int = 512,
    n_latents: int = 1000,
    example_ctx_len: int = 32,
    min_examples: int = 200,
    num_non_activating: int = 50,
    num_examples: int = 50,
    n_quantiles: int = 10,
    explainer_model: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    explainer_model_max_len: int = 5120,
    explainer_provider: str = "offline",
    explainer: str = "default",
    filter_bos: bool = False,
    pipeline_num_proc: int = cpu_count() // 2,
    num_gpus: int | None = None,
    verbose: bool = True,
    seed: int = 0,
    hf_token: str = "",
    logs_path: str | None = None,
    log_level: str = "INFO",
    skip_intruder: bool = False,
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
) -> None:
    """
    Evaluate raw model activations using all three evaluation types.

    This unified script runs:
    1. Intruder detection evaluation
    2. SAEBench autointerp evaluation
    3. SAEBench sparse probing evaluation

    Args:
        model_name: Model name to evaluate
        activation_key: Type of activation (layer_output, attn_output, mlp_output)
        layers: List of layer indices to evaluate (if None, uses all layers)
        model_dtype: Model data type
        dtype: Activation data type
        ctxlen: Context length for caching
        load_in_8bit: Load model in 8-bit
        n_tokens: Number of tokens to process
        batchsize: Batch size for processing
        n_latents: Number of latents to evaluate
        example_ctx_len: Context length for examples
        min_examples: Minimum examples per latent
        num_non_activating: Number of non-activating examples
        num_examples: Number of examples to show
        n_quantiles: Number of quantiles for sampling
        explainer_model: Model for explanations
        explainer_model_max_len: Max length for explainer
        explainer_provider: Provider for explainer
        explainer: Type of explainer
        filter_bos: Filter BOS tokens
        pipeline_num_proc: Number of pipeline processes
        num_gpus: Number of GPUs
        verbose: Verbose output
        seed: Random seed
        hf_token: HuggingFace token
        logs_path: Path to save logs
        log_level: Logging level
        skip_intruder: Skip intruder evaluation
        skip_autointerp: Skip autointerp evaluation
        skip_sparse_probing: Skip sparse probing evaluation
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running with log level: {log_level}")

    # Ensure at least one evaluation is enabled
    assert not (skip_intruder and skip_autointerp and skip_sparse_probing), (
        "Cannot skip all evaluations"
    )

    th_dtype = get_dtype(dtype)
    str_dtype = th_dtype.__str__().split(".")[-1]
    logger.trace(f"Using dtype: {str_dtype}")

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

    logger.info(
        f"Evaluating raw activations: {activation_key} from layers {layers_sorted} on {model_name}"
    )

    # Create experiment name
    layers_str = "_".join(map(str, layers_sorted))
    experiment_name = f"raw_{activation_key}_layers_{layers_str}_{model_name}"
    experiment_path = os.path.join(OUTPUT_DIR, experiment_name)
    artifacts_path = os.path.join(experiment_path, "artifacts")

    # Track success of each evaluation
    results = {
        "intruder": None,
        "autointerp": None,
        "sparse_probing": None,
    }

    # Run intruder evaluation
    if not skip_intruder:
        logger.info("=" * 80)
        logger.info("Running intruder evaluation...")
        logger.info("=" * 80)
        intruder_success = run_intruder_eval(
            activation_key=activation_key,
            layers=layers_sorted,
            model_name=model_name,
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
        )
        results["intruder"] = intruder_success
        if intruder_success:
            logger.info("✅ Intruder evaluation complete")
        else:
            logger.error("❌ Intruder evaluation failed")

    # Run autointerp evaluation
    if not skip_autointerp:
        logger.info("=" * 80)
        logger.info("Running autointerp evaluation...")
        logger.info("=" * 80)
        try:
            device = general_utils.setup_environment()
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
            results["autointerp"] = True
            logger.info("✅ Autointerp evaluation complete")
        except Exception as e:
            logger.error(f"❌ Autointerp evaluation failed: {e}")
            traceback.print_exc()
            results["autointerp"] = False

    # Run sparse probing evaluation
    if not skip_sparse_probing:
        logger.info("=" * 80)
        logger.info("Running sparse probing evaluation...")
        logger.info("=" * 80)
        try:
            device = general_utils.setup_environment()
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
            results["sparse_probing"] = True
            logger.info("✅ Sparse probing evaluation complete")
        except Exception as e:
            logger.error(f"❌ Sparse probing evaluation failed: {e}")
            traceback.print_exc()
            results["sparse_probing"] = False

    # Print summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    for eval_name, success in results.items():
        if success is None:
            logger.info(f"{eval_name}: SKIPPED")
        elif success:
            logger.info(f"{eval_name}: ✅ SUCCESS")
        else:
            logger.info(f"{eval_name}: ❌ FAILED")
    logger.info("=" * 80)

    # Check if any evaluation failed
    failed = [name for name, success in results.items() if success is False]
    if failed:
        logger.error(f"The following evaluations failed: {', '.join(failed)}")
        sys.exit(1)

    logger.success("🎉 All evaluations complete!")


if __name__ == "__main__":
    arguably.run()
