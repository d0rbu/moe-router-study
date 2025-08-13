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
    samples = [
        "Tiny sample 1",
        "Tiny sample 2",
        "Tiny sample 3",
        "Tiny sample 4",
    ]
    # Cast for typing; tests only iterate over the values
    return cast("IterableColumn", samples)


def test_dataset_text() -> IterableColumn:
    """Dataset specifically for testing purposes.

    This function is designed to be easily mocked in tests.
    It tries to load a dataset from the Hub, but falls back to toy_text if that fails.
    """
    try:
        # Use the load_dataset function directly so it can be mocked in tests
        ds = load_dataset("toy", split="train", streaming=True)
        return cast("IterableColumn", ds["text"])
    except Exception:
        # Fall back to toy_text for non-test environments
        return toy_text()


DATASETS: dict[str, Callable[[], IterableColumn]] = {
    "fw": fineweb_10bt_text,
    # Use test_dataset_text instead of patched_toy_text to avoid monkeypatching
    "toy": test_dataset_text,
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
