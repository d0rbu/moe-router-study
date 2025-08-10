from collections.abc import Callable
from typing import cast

from datasets import IterableColumn, load_dataset
from tqdm import tqdm


def fineweb_10bt_text() -> IterableColumn:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    return cast("IterableColumn", fineweb["text"])


def toy_text() -> IterableColumn:
    """Tiny, in-repo dataset for tests and quick runs."""
    # Return a simple in-memory sequence; avoids constructing datasets.IterableDataset
    # which requires an internal ex_iterable argument.
    samples = [
        "Tiny sample 1",
        "Tiny sample 2",
        "Tiny sample 3",
        "Tiny sample 4",
    ]
    # Quote the type for runtime-only typing context to satisfy Ruff TC006
    return cast("IterableColumn", samples)


DATASETS: dict[str, Callable[[], IterableColumn]] = {
    "fw": fineweb_10bt_text,
    "toy": toy_text,
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
