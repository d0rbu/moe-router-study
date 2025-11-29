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
import torch as th
import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor
from exp import OUTPUT_DIR
from exp.autointerp_saebench import Paths
from exp.autointerp_saebench import run_eval as run_autointerp_eval
from exp.kmeans import KMEANS_FILENAME, KMEANS_TYPE
from exp.sparse_probing_saebench import run_eval as run_sparse_probing_eval

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@arguably.command()
def path_eval_saebench(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    batchsize: int = 512,
    dtype: str = "bfloat16",
    seed: int = 0,
    logs_path: str | None = None,
    log_level: str = "INFO",
    skip_autointerp: bool = False,
    skip_sparse_probing: bool = False,
) -> None:
    """
    Evaluate the paths on the given model.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running with log level: {log_level}")

    th_dtype = get_dtype(dtype)
    str_dtype = th_dtype.__str__().split(".")[-1]
    logger.trace(f"Using dtype: {str_dtype}")

    device = general_utils.setup_environment()
    logger.trace(f"Using device: {device}")

    experiment_path = os.path.join(OUTPUT_DIR, experiment_dir)
    logger.trace(f"Using experiment path: {experiment_path}")

    config_path = os.path.join(experiment_path, "metadata.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    else:
        logger.trace(f"Using config path: {config_path}")

    kmeans_data_path = os.path.join(experiment_path, KMEANS_FILENAME)
    if not os.path.exists(kmeans_data_path):
        raise FileNotFoundError(f"Kmeans data file not found at {kmeans_data_path}")
    else:
        logger.trace(f"Using kmeans data file: {kmeans_data_path}")

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    postprocessor = RouterLogitsPostprocessor(
        config.get("postprocessor", RouterLogitsPostprocessor.MASKS)
    )

    assert config["type"] == KMEANS_TYPE, (
        f"Experiment is not a kmeans experiment, type={config['type']}"
    )
    assert config["model_name"] == model_name, (
        f"Model name mismatch: {model_name} != {config['model_name']}"
    )
    assert not skip_autointerp or not skip_sparse_probing, (
        "Cannot skip both autointerp and sparse probing"
    )
    logger.trace(f"Using config: {config}")

    model_config = get_model_config(model_name)
    hf_name = model_config.hf_name
    logger.debug(f"Using model hf_name: {hf_name}")

    paths_set = []
    kmeans_data = th.load(kmeans_data_path)

    # list of tensors of shape (num_centroids, num_layers * num_experts)
    centroid_sets = [
        centroids.to(dtype=th_dtype, device=device)
        for centroids in kmeans_data["centroids"]
    ]
    top_k = kmeans_data["top_k"]
    losses = kmeans_data["losses"].tolist()

    # Create a Paths object for each centroid set
    for i, centroids in enumerate(centroid_sets):
        paths = Paths(
            data=centroids,
            top_k=top_k,
            name=f"paths_{centroids.shape[0]}_set_{i}",
            postprocessor=postprocessor,
            metadata={
                "num_paths": centroids.shape[0],
                "top_k": top_k,
                "losses": losses,
                "centroid_set_index": i,
            },
        )
        paths_set.append(paths)
        logger.trace(
            f"Added paths to paths set: len={len(paths.data)} top_k={top_k} name={paths.name} metadata={paths.metadata}"
        )

    logger.trace(f"Using paths set: len={len(paths_set)}")

    # run autointerp
    if not skip_autointerp:
        autointerp_eval_dir = EVAL_DIRS["autointerp"]
        autointerp_eval_dir = os.path.join(OUTPUT_DIR, autointerp_eval_dir)
        logger.trace(f"Running autointerp evaluation in {autointerp_eval_dir}")
        run_autointerp_eval(
            config=AutoInterpEvalConfig(
                model_name=hf_name,
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
                model_name=hf_name,
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

    logger.success("done :)")


if __name__ == "__main__":
    arguably.run()
