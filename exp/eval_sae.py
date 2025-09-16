import glob
import os
import re

import arguably
from sae_bench.custom_saes import run_all_evals_dictionary_learning_saes
from sae_bench.sae_bench_utils import general_utils
from sae_lens.toolkit.pretrained_saes_directory import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from exp import OUTPUT_DIR
from exp.kmeans import KMEANS_TYPE, METADATA_FILENAME


@arguably.command()
def main(
    *,
    model_name: str = "olmoe-i",
    files: str = rf"{OUTPUT_DIR}/olmoe-i_lmsys.+ae\.pt$",
    eval_types: list[str] | None = None,
    batchsize: int = 512,
    dtype: str = "float32",
    save_activations: bool = False,
    seed: int = 0,
) -> None:
    """
    Evaluate the SAEs on the given model.
    """
    if eval_types is None:
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

    model_config = get_model_config(model_name)

    device = general_utils.setup_environment()

    files_regex = re.compile(files)

    ae_filepaths = [
        filepath
        for filepath in glob.glob(
            os.path.join(OUTPUT_DIR, "**", "*.pt"), recursive=True
        )
        if files_regex.match(filepath)
    ]

    for ae_filepath in ae_filepaths:
        kmeans_metadata_path = os.path.join(ae_filepath, METADATA_FILENAME)
        if not os.path.exists(kmeans_metadata_path):
            continue

        with open(kmeans_metadata_path) as f:
            metadata = yaml.safe_load(f)

        if metadata.get("type") == KMEANS_TYPE:
            raise ValueError(
                f"KMeans metadata found in {ae_filepath}, please evaluate with eval_kmeans.py"
            )

    ae_dirs = [os.path.dirname(ae_filepath) for ae_filepath in ae_filepaths]

    run_all_evals_dictionary_learning_saes.run_evals(
        repo_id=model_config.hf_name,
        model_name=model_name,
        eval_types=eval_types,
        batchsize=batchsize,
        dtype=th_dtype,
        save_activations=save_activations,
        seed=seed,
    )
