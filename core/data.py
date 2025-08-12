from collections.abc import Callable
from typing import Any, cast

from datasets import IterableColumn, load_dataset
from tqdm import tqdm


def fineweb_10bt_text() -> IterableColumn:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    return cast("IterableColumn", fineweb["text"])


def lmsys_chat_1m_text() -> IterableColumn:
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    def _format_conversation(row: dict[str, Any]) -> str:
        conversation = row.get("conversation") or []
        parts: list[str] = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _iter():
        for row in ds:
            # Type narrowing for static checkers
            if not isinstance(row, dict):
                continue
            if cast(dict[str, Any], row).get("redacted", False):
                continue
            yield _format_conversation(cast(dict[str, Any], row))

    # Cast the plain iterator of strings to IterableColumn for compatibility with callers
    return cast("IterableColumn", _iter())


DATASETS: dict[str, Callable[[], IterableColumn]] = {
    "fw": fineweb_10bt_text,
    # Register the new LMSYS dataset under a short key
    "lmsys": lmsys_chat_1m_text,
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
