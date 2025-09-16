import arguably
import sae_bench.sae_bench_utils.general_utils as general_utils

from core.dtype import get_dtype
from core.model import get_model_config

RANDOM_SEED = 0


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    eval_types: list[str] | None = None,
    llm_batchsize: int = 512,
    dtype: str = "float32",
    save_activations: bool = False,
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

    dtype = get_dtype(dtype)
    str_dtype = dtype.__str__().split(".")[-1]

    model_config = get_model_config(model_name)

    device = general_utils.setup_environment()
