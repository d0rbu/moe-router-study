from collections.abc import Callable
from typing import cast

from datasets import IterableColumn, IterableDataset, load_dataset
from tqdm import tqdm


def fineweb_10bt_text() -> IterableColumn:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    return cast("IterableColumn", fineweb["text"])


def toy_text() -> IterableColumn:
    """Tiny, in-repo dataset for tests and quick runs."""
    from datasets.iterable_dataset import ExamplesIterable

    def toy_generator():
        yield from [
            {"text": "Tiny sample 1"},
            {"text": "Tiny sample 2"},
            {"text": "Tiny sample 3"},
            {"text": "Tiny sample 4"},
        ]

    ex_iterable = ExamplesIterable(toy_generator, {})
    dataset = IterableDataset(ex_iterable=ex_iterable)
    return dataset["text"]


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
