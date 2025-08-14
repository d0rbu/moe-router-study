from __future__ import annotations

import gc
from itertools import batched
import os
from typing import TYPE_CHECKING

import arguably
from nnterp import StandardizedTransformer
import psutil
import torch as th
from tqdm import tqdm

from core.data import DATASETS
from core.device_map import CUSTOM_DEVICES
from core.model import MODELS
from exp import OUTPUT_DIR, ROUTER_LOGITS_DIR

if TYPE_CHECKING:
    from datasets import IterableDataset
    from transformers import PreTrainedTokenizer


def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # MB

    gpu_mem_allocated = 0
    gpu_mem_reserved = 0
    if th.cuda.is_available():
        gpu_mem_allocated = th.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_mem_reserved = th.cuda.memory_reserved() / (1024 * 1024)  # MB

    print(
        f"{prefix} Memory - CPU: {cpu_mem:.2f} MB, "
        f"GPU allocated: {gpu_mem_allocated:.2f} MB, "
        f"GPU reserved: {gpu_mem_reserved:.2f} MB"
    )


def save_router_logits(
    router_logit_collection: list[th.Tensor],
    tokenized_batch_collection: list[list[str]],
    top_k: int,
    file_idx: int,
) -> None:
    """Save router logits to disk and clean up memory."""
    # Stack tensors only when saving to avoid keeping references
    router_logits = th.cat(router_logit_collection, dim=0)

    # Create a new dictionary to avoid reference issues
    output = {
        "topk": top_k,
        "router_logits": router_logits,
        "tokens": tokenized_batch_collection,
    }

    # Save to disk
    output_path = os.path.join(ROUTER_LOGITS_DIR, f"{file_idx}.pt")
    th.save(output, output_path)

    # Explicitly clean up large tensors
    del router_logits
    del output

    # Force garbage collection
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()


def get_dataset_iterator(dataset_fn, tokenizer: PreTrainedTokenizer) -> IterableDataset:
    """Get a fresh dataset iterator to avoid memory accumulation."""
    return dataset_fn(tokenizer)


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
    # Use inference mode to prevent autograd graph buildup
    with th.inference_mode():
        # Tokenize the batch
        encoded_batch = model.tokenizer(batch, padding=True, return_tensors="pt")
        tokenized_batch = [model.tokenizer.tokenize(text) for text in batch]

        router_logits = []

        # Use the tracer to get router outputs
        with model.trace(batch) as tracer:
            for layer in router_layers:
                padding_mask: th.Tensor = encoded_batch.attention_mask.bool().view(
                    -1
                )  # (batch_size * seq_len)

                # Get router logits and immediately detach and move to CPU
                logits = model.routers_output[layer].cpu()[padding_mask].save()
                # Create a copy to break reference to the original tensor
                logits_copy = logits.clone().detach()
                router_logits.append(logits_copy)

                # Delete the original tensor to free memory
                del logits

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


def create_model(model_config, device_map):
    """Create a fresh model instance to avoid memory accumulation."""
    model = StandardizedTransformer(
        model_config.hf_name, check_attn_probs_with_trace=False, device_map=device_map
    )
    return model


@arguably.command()
def get_router_activations(
    model_name: str = "olmoe-i",
    dataset: str = "lmsys",
    batch_size: int = 4,
    device: str = "cpu",
    tokens_per_file: int = 10_000,
    reload_model_every: int = 10,  # Reload model every N batches
    memory_logging: bool = True,  # Enable memory logging
) -> None:
    """
    Extract router activations from a MoE model.

    Args:
        model_name: Name of the model to use
        dataset: Name of the dataset to use
        batch_size: Number of examples per batch
        device: Device to run on (cpu, cuda, mlp_gpu, attn_gpu)
        tokens_per_file: Number of tokens to save per file
        reload_model_every: Reload model every N batches to prevent memory leaks
        memory_logging: Enable memory usage logging
    """
    model_config = MODELS.get(model_name, None)

    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    dataset_fn = DATASETS.get(dataset, None)

    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset} not found")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROUTER_LOGITS_DIR, exist_ok=True)

    device_map = CUSTOM_DEVICES.get(device, lambda: device)()

    # Create initial model
    model = create_model(model_config, device_map)
    router_layers: list[int] = model.layers_with_routers
    top_k: int = model.router_probabilities.get_top_k()

    # Log initial memory state
    if memory_logging:
        log_memory_usage("Initial")

    with th.inference_mode():
        router_logit_collection: list[th.Tensor] = []
        tokenized_batch_collection: list[list[str]] = []
        router_logit_collection_size: int = 0
        router_logit_collection_idx: int = 0

        pbar = tqdm(total=tokens_per_file, desc="Filling up file")

        # Get a fresh dataset iterator
        dataset_iter = get_dataset_iterator(dataset_fn, model.tokenizer)

        # Process batches
        batch_count = 0
        for batch_count, batch in enumerate(batched(dataset_iter, batch_size)):
            if memory_logging:
                log_memory_usage(f"Before batch {batch_count}")

            # Reload model periodically to clear any accumulated state
            if batch_count > 0 and batch_count % reload_model_every == 0:
                print(f"Reloading model (batch {batch_count})")
                # Delete old model and create a new one
                del model
                gc.collect()
                if th.cuda.is_available():
                    th.cuda.empty_cache()

                model = create_model(model_config, device_map)
                router_layers = model.layers_with_routers

                if memory_logging:
                    log_memory_usage(f"After model reload (batch {batch_count})")

            # Process batch in isolated function to ensure variable cleanup
            router_logits, tokenized_batch = process_batch(batch, model, router_layers)

            # Store tensors
            router_logit_collection.append(router_logits)
            tokenized_batch_collection.extend(tokenized_batch)
            router_logit_collection_size += router_logits.shape[0]
            pbar.update(router_logits.shape[0])

            # Save the router probabilities to a file when we've collected enough
            if router_logit_collection_size >= tokens_per_file:
                if memory_logging:
                    log_memory_usage(
                        f"Before saving file {router_logit_collection_idx}"
                    )

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

                # Get a fresh dataset iterator to avoid memory accumulation
                del dataset_iter
                gc.collect()
                dataset_iter = get_dataset_iterator(dataset_fn, model.tokenizer)

                if memory_logging:
                    log_memory_usage(
                        f"After saving file {router_logit_collection_idx - 1}"
                    )

            # Clean up memory
            del router_logits
            del tokenized_batch
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()

            if memory_logging:
                log_memory_usage(f"After batch {batch_count}")

        # Save any remaining router probabilities to a file
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
        del model
        del dataset_iter
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()


if __name__ == "__main__":
    arguably.run()
