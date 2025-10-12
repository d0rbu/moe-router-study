"""
Comprehensive evaluation script for k-means path experiments.

This script:
1. Evaluates a k-means experiment using both path_eval_saebench and eval_intruder
2. Much simpler than eval_all_saes since k-means stores multiple centroids in a single file
"""

import subprocess
import sys

import arguably
from loguru import logger


def run_path_eval_saebench(
    experiment_name: str,
    model_name: str,
    batchsize: int,
    dtype: str,
    seed: int,
) -> bool:
    """Run SAEBench evaluation on a k-means path experiment."""
    logger.info(f"Running SAEBench evaluation on {experiment_name}")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "exp.path_eval_saebench",
        "--experiment-dir",
        experiment_name,
        "--model-name",
        model_name,
        "--batchsize",
        str(batchsize),
        "--dtype",
        dtype,
        "--seed",
        str(seed),
    ]

    logger.debug(f"🔧 Running command: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=3600,  # 1 hour timeout
        )
        logger.debug(f"✅ SAEBench evaluation completed for {experiment_name}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(
            f"❌ SAEBench evaluation timed out for {experiment_name} (1 hour limit)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ SAEBench evaluation failed for {experiment_name}")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        logger.error(
            f"❌ Unexpected error in SAEBench evaluation for {experiment_name}: {e}"
        )
        return False


def run_intruder_eval(
    experiment_name: str,
    model_name: str,
    model_dtype: str,
    n_tokens: int,
    batchsize: int,
    n_latents: int,
    seed: int,
) -> bool:
    """Run intruder evaluation on a k-means path experiment."""
    logger.info(f"Running intruder evaluation on {experiment_name}")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "exp.eval_intruder",
        "--experiment-dir",
        experiment_name,
        "--model-name",
        model_name,
        "--model-dtype",
        model_dtype,
        "--n-tokens",
        str(n_tokens),
        "--batchsize",
        str(batchsize),
        "--n-latents",
        str(n_latents),
        "--seed",
        str(seed),
    ]

    logger.debug(f"🔧 Running command: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=3600,  # 1 hour timeout
        )
        logger.debug(f"✅ Intruder evaluation completed for {experiment_name}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(
            f"❌ Intruder evaluation timed out for {experiment_name} (1 hour limit)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Intruder evaluation failed for {experiment_name}")
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        logger.error(
            f"❌ Unexpected error in intruder evaluation for {experiment_name}: {e}"
        )
        return False


@arguably.command()
def eval_all_paths(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    run_saebench: bool = True,
    run_intruder: bool = True,
    saebench_batchsize: int = 512,
    intruder_n_tokens: int = 10_000_000,
    intruder_batchsize: int = 8,
    intruder_n_latents: int = 1000,
    dtype: str = "bfloat16",
    seed: int = 0,
    log_level: str = "INFO",
) -> None:
    """
    Evaluate a k-means path experiment using both SAEBench and intruder evaluations.

    Args:
        experiment_dir: Name of the k-means experiment directory (e.g., "kmeans_2024-01-01_00-00-00")
        model_name: Model name to evaluate on
        run_saebench: Whether to run SAEBench evaluation
        run_intruder: Whether to run intruder evaluation
        saebench_batchsize: Batch size for SAEBench evaluation
        intruder_n_tokens: Number of tokens for intruder evaluation
        intruder_batchsize: Batch size for intruder evaluation
        intruder_n_latents: Number of latents for intruder evaluation
        dtype: Data type for evaluation
        seed: Random seed
        log_level: Logging level
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info(f"Running evaluation for k-means experiment: {experiment_dir}")
    logger.info(
        f"Configuration: run_saebench={run_saebench}, run_intruder={run_intruder}"
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
        )
        if not saebench_success:
            logger.error("SAEBench evaluation failed!")
            success = False
        else:
            logger.success("✅ SAEBench evaluation completed successfully")

    if run_intruder:
        logger.info("Starting intruder evaluation...")
        intruder_success = run_intruder_eval(
            experiment_dir,
            model_name,
            dtype,
            intruder_n_tokens,
            intruder_batchsize,
            intruder_n_latents,
            seed,
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
