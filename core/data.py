from collections.abc import Callable
from typing import Any, cast

from datasets import IterableDataset, load_dataset
from tqdm import tqdm


def fineweb_10bt_text() -> Any:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    return cast(Any, fineweb["text"])


DATASETS: dict[str, Callable[[], Any]] = {
    "fw": fineweb_10bt_text,
}


if __name__ == "__main__":
    for dataset_name, dataset_fn in tqdm(
        DATASETS.items(), desc="Loading datasets", total=len(DATASETS)
    ):
        dataset = dataset_fn()
        log_every = 1000

        for sample_idx, sample in tqdm(
            enumerate(dataset), desc=f"Loading {dataset_name}"
        ):
            if sample_idx % log_every == 0:
                print(f"Sample {sample_idx} from {dataset_name}: {sample}")
