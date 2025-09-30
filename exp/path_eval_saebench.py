import os

import arguably
from dotenv import load_dotenv
from loguru import logger
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    output_folders as EVAL_DIRS,
)
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.sae_bench_utils import general_utils
import torch as th
import yaml

from core.dtype import get_dtype
from exp import OUTPUT_DIR
from exp.autointerp_saebench import Paths
from exp.autointerp_saebench import run_eval as run_autointerp_eval
from exp.kmeans import KMEANS_FILENAME, KMEANS_TYPE
from exp.sparse_probing_saebench import run_eval as run_sparse_probing_eval

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@arguably.command()
def main(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    batchsize: int = 512,
    dtype: str = "bfloat16",
    seed: int = 0,
    logs_path: str | None = None,
    log_level: str = "INFO",
) -> None:
    """
    Evaluate the paths on the given model.
    """

    th_dtype = get_dtype(dtype)
    str_dtype = th_dtype.__str__().split(".")[-1]

    device = general_utils.setup_environment()

    experiment_path = os.path.join(OUTPUT_DIR, experiment_dir)

    config_path = os.path.join(experiment_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["type"] == KMEANS_TYPE, (
        f"Experiment is not a kmeans experiment, type={config['type']}"
    )
    assert config["model_name"] == model_name, (
        f"Model name mismatch: {model_name} != {config['model_name']}"
    )

    paths_set = []
    with open(os.path.join(experiment_path, KMEANS_FILENAME)) as f:
        kmeans_data = th.load(f)

        # list of tensors of shape (num_centroids, num_layers * num_experts)
        centroid_sets = kmeans_data["centroids"].to(dtype=th_dtype, device=device)
        top_k = kmeans_data["top_k"]
        losses = kmeans_data["losses"].tolist()

        paths = Paths(
            data=centroid_sets,
            top_k=top_k,
            name=f"paths_{centroid_sets.shape[0]}",
            metadata={
                "num_paths": centroid_sets.shape[0],
                "top_k": top_k,
                "losses": losses,
            },
        )
        paths_set.append(paths)

    # run autointerp
    autointerp_eval_dir = EVAL_DIRS["autointerp"]
    autointerp_eval_dir = os.path.join(OUTPUT_DIR, autointerp_eval_dir)
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

    logger.info("Autointerp evaluation complete, running sparse probing")

    sparse_probing_eval_dir = EVAL_DIRS["sparse_probing"]
    sparse_probing_eval_dir = os.path.join(OUTPUT_DIR, sparse_probing_eval_dir)
    run_sparse_probing_eval(
        config=SparseProbingEvalConfig(
            model_name=model_name,
            random_seed=seed,
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
