import gc
from itertools import batched
import os

import arguably
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.data import DATASETS
from core.device_map import CUSTOM_DEVICES
from core.model import MODELS
from exp import OUTPUT_DIR, ROUTER_LOGITS_DIR


def save_router_logits(
    router_logit_collection: list[th.Tensor],
    tokenized_batch_collection: list[list[str]],
    top_k: int,
    file_idx: int,
) -> None:
    router_logits = th.cat(router_logit_collection, dim=0)
    output: dict[str, th.Tensor] = {
        "topk": top_k,
        "router_logits": router_logits,
        "tokens": tokenized_batch_collection,
    }
    output_path = os.path.join(ROUTER_LOGITS_DIR, f"{file_idx}.pt")
    th.save(output, output_path)

    # Explicitly clean up large tensors
    del router_logits
    del output

    # Force garbage collection
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()


def process_batch(
    batch: list[str],
    model: StandardizedTransformer,
    router_layers: list[int],
) -> tuple[th.Tensor, list[list[str]]]:
    """
    Process a single batch of data to extract router logits.

    Args:
        batch: A batch of text data
        model: The transformer model
        router_layers: List of layer indices with routers

    Returns:
        tuple: (router_logits, tokenized_batch)
    """

    encoded_batch = model.tokenizer(batch, padding=True, return_tensors="pt")
    tokenized_batch = [model.tokenizer.tokenize(text) for text in batch]

    router_logits = []

    with model.trace(batch) as tracer:
        for layer in router_layers:
            padding_mask: th.Tensor = encoded_batch.attention_mask.bool().view(
                -1
            )  # (batch_size * seq_len)

            # Get router logits and immediately detach and move to CPU
            router_output = model.routers_output[layer]

            match router_output:
                case (router_scores, _router_indices):
                    logits = router_scores.cpu()[padding_mask].save()
                case tuple():
                    raise ValueError(
                        f"Found tuple of length {len(router_output)} for router output at layer {layer}"
                    )
                case router_scores:
                    logits = router_scores.cpu()[padding_mask].save()

            router_logits.append(logits.clone().detach())

        # Explicitly stop the tracer to clean up resources
        tracer.stop()

    # Stack the logits and create a new tensor to break references
    router_logits_tensor = th.stack(router_logits, dim=1).clone().detach()

    # Clean up memory explicitly
    del encoded_batch
    del router_logits
    del tracer

    # Force garbage collection to clean up any lingering references
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()

    return router_logits_tensor, tokenized_batch


@arguably.command()
def get_router_activations(
    model_name: str = "gpt",
    dataset: str = "lmsys",
    *_args,
    batch_size: int = 4,
    device: str = "cpu",
    tokens_per_file: int = 10_000,
    resume: bool = False,
) -> None:
    model_config = MODELS.get(model_name, None)

    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    dataset_fn = DATASETS.get(dataset, None)

    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset} not found")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROUTER_LOGITS_DIR, exist_ok=True)

    device_map = CUSTOM_DEVICES.get(device, lambda: device)()

    model = StandardizedTransformer(
        model_config.hf_name, check_attn_probs_with_trace=False, device_map=device_map
    )
    router_layers: list[int] = model.layers_with_routers
    top_k: int = model.router_probabilities.get_top_k()

    start_file_idx = 0
    if resume:
        for file in os.listdir(ROUTER_LOGITS_DIR):
            if file.endswith(".pt"):
                start_file_idx = max(start_file_idx, int(file.split(".")[0]))

    with th.inference_mode():
        router_logit_collection: list[th.Tensor] = []
        tokenized_batch_collection: list[list[str]] = []
        router_logit_collection_size: int = 0
        router_logit_collection_idx: int = 0

        pbar = tqdm(total=tokens_per_file, desc="Filling up file")

        for _batch_idx, batch in enumerate(
            tqdm(
                batched(dataset_fn(model.tokenizer), batch_size),
                desc="Processing batches",
            )
        ):
            # Process batch in isolated function to ensure variable cleanup
            router_logits, tokenized_batch = process_batch(batch, model, router_layers)

            # Store weak references to avoid circular references
            router_logit_collection.append(router_logits)
            tokenized_batch_collection.extend(tokenized_batch)
            router_logit_collection_size += router_logits.shape[0]
            pbar.update(router_logits.shape[0])

            # save the router probabilities to a file
            if router_logit_collection_size >= tokens_per_file:
                save_router_logits(
                    router_logit_collection,
                    tokenized_batch_collection,
                    top_k,
                    router_logit_collection_idx,
                )
                router_logit_collection_idx += 1
                router_logit_collection_size = 0
                router_logit_collection = []
                tokenized_batch_collection = []
                pbar.reset()

            # clean up memory
            del router_logits
            del tokenized_batch
            gc.collect()
            th.cuda.empty_cache()

        # save the remaining router probabilities to a file
        if router_logit_collection_size > 0:
            save_router_logits(
                router_logit_collection,
                tokenized_batch_collection,
                top_k,
                router_logit_collection_idx,
            )

        # Final cleanup
        del router_logit_collection
        del tokenized_batch_collection
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()


if __name__ == "__main__":
    arguably.run()
