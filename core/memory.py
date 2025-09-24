import gc

import torch as th


def clear_memory() -> None:
    if th.cuda.is_available():
        th.cuda.empty_cache()
    gc.collect()
