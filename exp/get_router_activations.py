import argparse
from collections.abc import Callable, Iterator
import json
import os

from datasets import load_dataset
import torch as th
import torch.nn.functional as F
from tqdm import tqdm

from core.data import batched
from core.model import Model, load_model
from core.utils import get_device


def get_router_activations(
    model: Model,
    input_ids: th.Tensor,
    attention_mask: th.Tensor | None = None,
    return_logits: bool = False,
) -> tuple[th.Tensor, th.Tensor]:
    """
    Get the router activations for a batch of inputs.

    Args:
        model: The model to use.
        input_ids: The input ids.
        attention_mask: The attention mask.
        return_logits: Whether to return the router logits.

    Returns:
        A tuple of (activated_experts, router_logits).
    """
    device = get_device()

    if attention_mask is None:
        attention_mask = th.ones_like(input_ids)

    # move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # get the router activations
    with th.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
        )

    router_logits = outputs.router_logits

    # get the activated experts
    activated_experts = []
    for layer_idx in range(len(router_logits)):
        layer_router_logits = router_logits[layer_idx]
        batch_size, seq_len, num_experts = layer_router_logits.shape

        # reshape to (batch_size * seq_len, num_experts)
        layer_router_logits = layer_router_logits.reshape(-1, num_experts)

        # get the top-k experts
        top_k_experts = th.topk(
            layer_router_logits, k=model.model.config.num_experts_per_tok, dim=-1
        ).indices

        # reshape back to (batch_size, seq_len, num_experts_per_tok)
        top_k_experts = top_k_experts.reshape(
            batch_size, seq_len, model.model.config.num_experts_per_tok
        )

        # only keep the last token
        top_k_experts = top_k_experts[:, -1, :]

        # convert to one-hot
        one_hot = F.one_hot(top_k_experts, num_classes=num_experts).sum(dim=1)

        activated_experts.append(one_hot)

    # stack the activated experts
    activated_experts = th.stack(activated_experts, dim=1)

    if return_logits:
        return activated_experts, router_logits

    return activated_experts, None


def get_top_k_experts(
    model: Model,
    input_ids: th.Tensor,
    attention_mask: th.Tensor | None = None,
    k: int = 5,
) -> list[list[list[int]]]:
    """
    Get the top-k experts for a batch of inputs.

    Args:
        model: The model to use.
        input_ids: The input ids.
        attention_mask: The attention mask.
        k: The number of experts to return.

    Returns:
        A list of lists of lists of expert indices.
    """
    device = get_device()

    if attention_mask is None:
        attention_mask = th.ones_like(input_ids)

    # move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # get the router activations
    with th.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
        )

    router_logits = outputs.router_logits

    # get the top-k experts
    top_k_experts = []
    for batch_idx in range(input_ids.shape[0]):
        batch_top_k_experts = []
        for layer_idx in range(len(router_logits)):
            layer_router_logits = router_logits[layer_idx][batch_idx]
            seq_len, num_experts = layer_router_logits.shape

            # only keep the last token
            layer_router_logits = layer_router_logits[-1]

            # get the top-k experts
            layer_top_k_experts = th.topk(layer_router_logits, k=k, dim=-1).indices

            # convert to list
            layer_top_k_experts = layer_top_k_experts.tolist()

            batch_top_k_experts.append(layer_top_k_experts)

        top_k_experts.append(batch_top_k_experts)

    return top_k_experts


def save_activations(
    model_name: str,
    dataset_fn: Callable,
    batch_size: int = 32,
    tokens_per_file: int = 1_000_000,
    output_dir: str | None = None,
) -> None:
    """
    Save the router activations for a dataset.

    Args:
        model_name: The name of the model to use.
        dataset_fn: A function that returns a dataset.
        batch_size: The batch size to use.
        tokens_per_file: The number of tokens to save per file.
        output_dir: The directory to save the activations to.
    """
    model = load_model(model_name)

    if output_dir is None:
        output_dir = f"activations/{model_name}"

    os.makedirs(output_dir, exist_ok=True)

    file_idx = 0
    token_count = 0
    activated_experts_list = []
    top_k_list = []

    pbar = tqdm(total=tokens_per_file, desc="Filling up file")

    for _batch_idx, batch in enumerate(
        tqdm(
            batched(dataset_fn(model.tokenizer), batch_size),
            desc="Processing batches",
        )
    ):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # get the router activations
        activated_experts, _ = get_router_activations(model, input_ids, attention_mask)
        top_k = get_top_k_experts(model, input_ids, attention_mask)

        # add to the list
        activated_experts_list.append(activated_experts)
        top_k_list.extend(top_k)

        # update the token count
        token_count += input_ids.shape[0]
        pbar.update(input_ids.shape[0])

        # save the activations if we have enough tokens
        if token_count >= tokens_per_file:
            # concatenate the activated experts
            activated_experts = th.cat(activated_experts_list, dim=0)

            # save the activations
            with open(f"{output_dir}/activated_experts_{file_idx}.pt", "wb") as f:
                th.save(activated_experts, f)

            # save the top-k experts
            with open(f"{output_dir}/top_k_{file_idx}.json", "w") as f:
                json.dump(top_k_list, f)

            # reset the lists
            activated_experts_list = []
            top_k_list = []

            # update the file index
            file_idx += 1
            token_count = 0

            # reset the progress bar
            pbar.reset()

    # save any remaining activations
    if activated_experts_list:
        # concatenate the activated experts
        activated_experts = th.cat(activated_experts_list, dim=0)

        # save the activations
        with open(f"{output_dir}/activated_experts_{file_idx}.pt", "wb") as f:
            th.save(activated_experts, f)

        # save the top-k experts
        with open(f"{output_dir}/top_k_{file_idx}.json", "w") as f:
            json.dump(top_k_list, f)


def c4_dataset(tokenizer) -> Iterator[dict[str, th.Tensor]]:
    """
    Get the C4 dataset.

    Args:
        tokenizer: The tokenizer to use.

    Returns:
        An iterator over the dataset.
    """
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    for item in dataset:
        # tokenize the text
        tokenized = tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        yield tokenized


def pile_dataset(tokenizer) -> Iterator[dict[str, th.Tensor]]:
    """
    Get the Pile dataset.

    Args:
        tokenizer: The tokenizer to use.

    Returns:
        An iterator over the dataset.
    """
    dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
    for item in dataset:
        # tokenize the text
        tokenized = tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        yield tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tokens-per-file", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "c4":
        dataset_fn = c4_dataset
    elif args.dataset == "pile":
        dataset_fn = pile_dataset
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    save_activations(
        args.model,
        dataset_fn,
        batch_size=args.batch_size,
        tokens_per_file=args.tokens_per_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
