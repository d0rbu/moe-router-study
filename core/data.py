from collections.abc import Callable, Iterable
import os
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.type import assert_type
from exp import DATASET_DIRNAME


def fineweb_10bt_text(
    _tokenizer: PreTrainedTokenizer | None = None,
) -> Iterable[str]:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    # Handle both real IterableDataset and test mocks (dict)
    if isinstance(fineweb, dict):
        return fineweb["text"]
    else:
        return (sample["text"] for sample in fineweb)


def toy_text(
    _tokenizer: PreTrainedTokenizer | None = None,
) -> Iterable[str]:
    """Tiny, in-repo dataset for tests and quick runs."""
    samples = [
        "Tiny sample 1",
        "Tiny sample 2",
        "Tiny sample 3",
        "Tiny sample 4",
    ]

    return samples


def smollm2_small(
    _tokenizer: PreTrainedTokenizer | None = None,
) -> Iterable[str]:
    smollm2_135m_10b = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train[:1%]")

    # Handle both real Dataset and test mocks (dict)
    if isinstance(smollm2_135m_10b, dict):
        return smollm2_135m_10b["text"]
    else:
        dataset = assert_type(smollm2_135m_10b, Dataset)
        return dataset["text"]


def lmsys_chat_1m_text(
    tokenizer: PreTrainedTokenizer,
    start_idx: int = 0,
    stop_idx: int = 0,
    streaming: bool = True,
) -> Iterable[str]:
    """Stream and format conversations from the LMSYS Chat-1M dataset.

    Each conversation is formatted as a plain text transcript with "role: content" format,
    with each message on a new line. Redacted conversations are skipped.

    Returns:
        Iterable[str]: Stream of formatted conversation texts
    """
    hf_name = "lmsys/lmsys-chat-1m"
    local_path = os.path.join(os.path.abspath(DATASET_DIRNAME), hf_name)
    path = local_path if os.path.exists(local_path) else hf_name

    print(f"Loading dataset from {path}")
    ds = load_dataset(path, split="train", streaming=streaming)

    if streaming:
        assert start_idx == 0 and stop_idx == 0, (
            "Streaming mode does not support start_idx and stop_idx"
        )
    else:
        dataset = assert_type(ds, Dataset)

        if stop_idx == 0:
            stop_idx = len(dataset)

        assert start_idx >= 0 and stop_idx >= 0, (
            "Non-streaming mode requires start_idx and stop_idx to be non-negative"
        )
        assert start_idx < stop_idx, "start_idx must be less than stop_idx"
        assert start_idx < len(dataset), (
            "start_idx must be less than the length of the dataset"
        )
        assert stop_idx <= len(dataset), (
            "stop_idx must be less than or equal to the length of the dataset"
        )

    def _format_conversation(conversation: list[dict[str, Any]]) -> str:
        chat = tokenizer.apply_chat_template(conversation, tokenize=False)

        if not isinstance(chat, str):
            raise ValueError(f"Expected chat to be a string, got {type(chat)}")

        return chat

    def _iter():
        if streaming:
            # Handle objects that support subscripting (dict, mocks, etc.)
            if hasattr(ds, "__getitem__"):
                conversations = ds["conversation"]  # type: ignore[index]
            else:
                conversations = (sample["conversation"] for sample in ds)
            iterator = tqdm(conversations, desc="Formatting conversations")
        else:
            subset_ds = dataset.select(range(start_idx, stop_idx))
            conversations = subset_ds["conversation"]
            iterator = tqdm(
                conversations,
                desc="Formatting conversations",
                total=stop_idx - start_idx,
            )

        for conversation in iterator:
            yield _format_conversation(conversation)

    return _iter()


DATASETS: dict[str, Callable[[PreTrainedTokenizer], Iterable[str]]] = {
    "fw": fineweb_10bt_text,
    "toy": toy_text,
    "lmsys": lmsys_chat_1m_text,
    "smol": smollm2_small,
}


def get_dataset_fn(
    dataset_name: str,
) -> Callable[[PreTrainedTokenizer], Iterable[str]]:
    dataset_fn = DATASETS.get(dataset_name)
    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset_fn


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
