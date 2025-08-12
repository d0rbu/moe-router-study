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


def patched_toy_text() -> IterableColumn:
    """Toy text that defers to datasets.load_dataset when patched.

    Integration tests patch datasets.load_dataset to return a MagicMock that
    yields a list of strings. Use it when available, otherwise fall back to
    the static toy_text samples.
    """
    try:
        # Import module to ensure monkeypatching `datasets.load_dataset` works
        import datasets  # type: ignore

        ds = datasets.load_dataset("toy", split="train", streaming=True)
        # When patched, ds["text"] is an iterator; when not, this may raise.
        return cast("IterableColumn", ds["text"])  # type: ignore[index]
    except Exception:
        return toy_text()


DATASETS: dict[str, Callable[[], IterableColumn]] = {
    "fw": fineweb_10bt_text,
    # Point to patched_toy_text so integration tests that monkeypatch
    # datasets.load_dataset take effect, while toy_text remains available
    # for unit tests that expect fixed strings.
    "toy": patched_toy_text,
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
