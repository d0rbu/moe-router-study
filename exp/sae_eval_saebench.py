import os
import sys

import arguably
from dotenv import load_dotenv
from loguru import logger
from sae_bench.custom_saes import run_all_evals_dictionary_learning_saes
from sae_bench.sae_bench_utils import general_utils

from core.dtype import get_dtype
from exp import OUTPUT_DIR

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@arguably.command()
def main(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    eval_types: list[str] | None = None,
    batchsize: int = 512,
    dtype: str = "bfloat16",
    seed: int = 0,
    log_level: str = "INFO",
    num_autointerp_latents: int = 1000,
) -> None:
    """
    Evaluate the SAEs on the given model.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running with log level: {log_level}")

    if not eval_types:
        eval_types = [
            "absorption",
            "autointerp",
            "core",
            "scr",
            "tpp",
            "sparse_probing",
            "unlearning",
        ]

    th_dtype = get_dtype(dtype)
    str_dtype = th_dtype.__str__().split(".")[-1]

    device = general_utils.setup_environment()

    experiment_dir_path = os.path.join(OUTPUT_DIR, experiment_dir)
    sae_locations = os.listdir(experiment_dir_path)

    logger.info(f"Evaluating SAEs in {experiment_dir}")
    logger.debug(f"SAE locations: {sae_locations}")
    logger.debug(f"Experiment directory path: {experiment_dir_path}")
    logger.debug(f"Eval types: {eval_types}")

    run_all_evals_dictionary_learning_saes.run_evals(
        repo_id=experiment_dir,
        model_name=model_name,
        sae_locations=sae_locations,
        llm_batch_size=batchsize,
        llm_dtype=str_dtype,
        device=device,
        eval_types=eval_types,
        random_seed=seed,
        download_location=OUTPUT_DIR,
        api_key=OPENAI_API_KEY,
        num_autointerp_latents=num_autointerp_latents,
    )


if __name__ == "__main__":
    arguably.run()
