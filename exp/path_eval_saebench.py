import os

import arguably
from dotenv import load_dotenv
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    output_folders as EVAL_DIRS,
)
from sae_bench.evals.autointerp import main as autointerp
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.sae_bench_utils import general_utils
import yaml

from core.dtype import get_dtype
from exp import OUTPUT_DIR
from exp.kmeans import KMEANS_TYPE

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@arguably.command()
def main(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    batchsize: int = 512,
    dtype: str = "float32",
    seed: int = 0,
    logs_path: str | None = None,
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

    # TODO: wrap paths in SAE objects and pass here
    selected_saes = None

    # run autointerp
    autointerp_eval_dir = EVAL_DIRS["autointerp"]
    autointerp_eval_dir = os.path.join(OUTPUT_DIR, autointerp_eval_dir)
    autointerp.run_eval(
        config=AutoInterpEvalConfig(
            model_name=model_name,
            random_seed=seed,
            llm_batch_size=batchsize,
            llm_dtype=str_dtype,
        ),
        selected_saes=selected_saes,
        device=device,
        api_key=OPENAI_API_KEY,
        output_path=autointerp_eval_dir,
        force_rerun=False,
        save_logs_path=logs_path,
        artifacts_path=os.path.join(experiment_path, "artifacts"),
    )

    # TODO: run sparse probing
