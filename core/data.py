from datasets import IterableColumn, IterableDataset, load_dataset
from typing import Callable, Generator, cast
from tqdm import tqdm


def fineweb_10bt_text() -> IterableColumn:
    fineweb = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    
    return cast(IterableColumn, fineweb["text"])


DATASETS: dict[str, Callable[[], IterableColumn]] = {
    "fw": fineweb_10bt_text,
}


if __name__ == "__main__":
    for dataset_name, dataset_fn in tqdm(DATASETS.items(), desc="Loading datasets", total=len(DATASETS)):
        dataset = dataset_fn()

        for sample_idx, sample in tqdm(enumerate(dataset), desc=f"Loading {dataset_name}"):
            pass

        print(f"Sample {sample_idx} from {dataset_name}: {sample}")
