from collections.abc import Callable
import os
from typing import Any, cast

from datasets import Dataset, IterableColumn, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from exp import DATASET_DIRNAME


def fineweb_10bt_text(_tokenizer: PreTrainedTokenizer) -> IterableColumn:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    return cast("IterableColumn", fineweb["text"])


def toy_text(_tokenizer: PreTrainedTokenizer) -> IterableColumn:
    """Tiny, in-repo dataset for tests and quick runs."""
    samples = [
        "Tiny sample 1",
        "Tiny sample 2",
        "Tiny sample 3",
        "Tiny sample 4",
    ]
    # Cast for typing; tests only iterate over the values
    return cast("IterableColumn", samples)


def test_dataset_text(_tokenizer: PreTrainedTokenizer) -> IterableColumn:
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
        return toy_text(_tokenizer)


def lmsys_chat_1m_text(
    tokenizer: PreTrainedTokenizer,
    start_idx: int = 0,
    stop_idx: int = 0,
    streaming: bool = False,
) -> IterableColumn:
    """Stream and format conversations from the LMSYS Chat-1M dataset.

    Each conversation is formatted as a plain text transcript with "role: content" format,
    with each message on a new line. Redacted conversations are skipped.

    Returns:
        IterableColumn: Stream of formatted conversation texts
    """
    hf_name = "lmsys/lmsys-chat-1m"
    local_path = os.path.join(os.path.abspath(DATASET_DIRNAME), hf_name)
    path = local_path if os.path.exists(local_path) else hf_name

    ds = load_dataset(path, split="train", streaming=streaming)

    if streaming:
        assert start_idx == 0 and stop_idx == 0, (
            "Streaming mode does not support start_idx and stop_idx"
        )
    else:
        ds = cast("Dataset", ds)
        assert start_idx >= 0 and stop_idx >= 0, (
            "Non-streaming mode requires start_idx and stop_idx to be non-negative"
        )
        assert start_idx < stop_idx, "start_idx must be less than stop_idx"
        assert start_idx < len(ds), (
            "start_idx must be less than the length of the dataset"
        )
        assert stop_idx <= len(ds), (
            "stop_idx must be less than or equal to the length of the dataset"
        )

        if stop_idx == 0:
            stop_idx = len(ds)

    def _format_conversation(conversation: list[dict[str, Any]]) -> str:
        chat = tokenizer.apply_chat_template(conversation, tokenize=False)

        if not isinstance(chat, str):
            raise ValueError(f"Expected chat to be a string, got {type(chat)}")

        return chat

    def _iter():
        conversations = ds["conversation"]

        if streaming:
            iterator = tqdm(conversations, desc="Formatting conversations")
        else:
            conversations = cast("Dataset", conversations)
            iterator = tqdm(
                conversations[start_idx:stop_idx],
                desc="Formatting conversations",
                total=stop_idx - start_idx,
            )

        for conversation in iterator:
            yield _format_conversation(conversation)

    # Cast the plain iterator of strings to IterableColumn for compatibility with callers
    return cast("IterableColumn", _iter())


DATASETS: dict[str, Callable[[PreTrainedTokenizer], IterableColumn]] = {
    "fw": fineweb_10bt_text,
    # Use test_dataset_text instead of patched_toy_text to avoid monkeypatching
    "toy": test_dataset_text,
    # Register the new LMSYS dataset under a short key
    "lmsys": lmsys_chat_1m_text,
}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    for dataset_name, dataset_fn in tqdm(
        DATASETS.items(), desc="Loading datasets", total=len(DATASETS)
    ):
        dataset = dataset_fn(tokenizer)
        log_every = 1000

        for sample_idx, sample in tqdm(
            enumerate(dataset), desc=f"Loading {dataset_name}"
        ):
            if sample_idx % log_every == 0:
                print(f"Sample {sample_idx} from {dataset_name}: {sample}")
