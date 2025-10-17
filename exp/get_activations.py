import asyncio
from collections import deque
from collections.abc import Sized
from enum import StrEnum
import gc
from itertools import pairwise
import math
import os
import queue
import re
import sys
import time
from typing import Any
import warnings

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import trackio as wandb
import yaml

from core.data import get_dataset_fn
from core.device import DeviceType, assert_device_type, get_backend
from core.dtype import get_dtype
from core.model import get_model_config
from core.type import assert_type
from exp import ACTIVATION_DIRNAME, MODEL_DIRNAME, OUTPUT_DIR
from exp.training import get_experiment_name

# Constants
CONFIG_FILENAME = "config.yaml"

# within-node parallelism constants
MAIN_QUEUE_MAXSIZE = 16
GPU_QUEUE_MAXSIZE = 8
OUTPUT_QUEUE_MAXSIZE = 3


def save_config(config: dict, experiment_dir: str) -> None:
    """Save experiment configuration to a YAML file."""
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


CONFIG_KEYS_TO_VERIFY = {
    "model_name",
    "dataset_name",
    "tokens_per_file",
}


def verify_config(config: dict, experiment_dir: str) -> None:
    """Verify that the current configuration matches the saved one."""
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        saved_config = yaml.safe_load(f)

    # Check for mismatches
    mismatches = {}
    for key in CONFIG_KEYS_TO_VERIFY:
        current_value = config.get(key)
        saved_value = saved_config.get(key)
        if current_value != saved_value and current_value is not None:
            mismatches[key] = (saved_value, current_value)

    if mismatches:
        mismatch_str = "\n".join(
            f"  - {key}: saved={saved} vs current={current}"
            for key, (saved, current) in mismatches.items()
        )
        raise ValueError(
            f"Configuration mismatch with existing experiment:\n{mismatch_str}"
        )


class ActivationKeys(StrEnum):
    ATTN_OUTPUT = "attn_output"
    ROUTER_LOGITS = "router_logits"
    MLP_OUTPUT = "mlp_output"
    LAYER_OUTPUT = "layer_output"


ACTIVATION_KEYS = frozenset(str(activation_key) for activation_key in ActivationKeys)


def process_batch(
    encoded_batch: dict,
    batch_idx: int,
    model: StandardizedTransformer,
    rank: int,
    minibatch_size: int,
    router_layers: set[int],
    layers_to_store: set[int],
    activations_to_store: frozenset[str] = ACTIVATION_KEYS,
    device_type: DeviceType = "cuda",
) -> dict[str, th.Tensor]:
    """Process a batch of texts through the model and extract router logits.

    Args:
        encoded_batch: Encoded batch from tokenizer with padding.
        batch_idx: Index of the batch.
        model: Model to process the batch.
        rank: Rank of the device.
        minibatch_size: Size of the minibatch to process on each device.
        router_layers: Set of router layer indices to extract.
        layers_to_store: Set of layer indices to store.
        stored_activations: Activations to score
        device_type: Device type ("cuda" or "xpu", defaults to "cuda")

    Returns:
        Dictionary of activations. These are lists of lists of tensors, so they need to be cleaned up by the caller.
    """
    backend = get_backend(device_type)

    logger.debug(
        f"Processing batch {batch_idx} with activations to store: {activations_to_store} layers to store: {layers_to_store}"
    )

    batch_size = encoded_batch["input_ids"].shape[0]

    if minibatch_size <= 0:
        minibatch_size = batch_size
        logger.warning(f"Minibatch size is 0, using batch size {batch_size}")
    else:
        minibatch_size = min(minibatch_size, batch_size)

    num_minibatches = math.ceil(batch_size / minibatch_size)
    num_layers = len(assert_type(model.layers, Sized))

    # Extract activations
    activations = {
        activation_key: [[] for _ in range(num_layers)]
        for activation_key in activations_to_store
    }

    for minibatch_idx in tqdm(
        range(num_minibatches),
        desc=f"Batch {batch_idx}",
        total=num_minibatches,
        leave=False,
        position=rank * 2,
    ):
        backend.empty_cache()
        gc.collect()

        minibatch_start = minibatch_idx * minibatch_size
        minibatch_end = min(minibatch_start + minibatch_size, batch_size)
        encoded_minibatch = {
            k: v[minibatch_start:minibatch_end].to(model.device)
            for k, v in encoded_batch.items()
        }

        minibatch_token_count = encoded_minibatch["attention_mask"].sum().item()
        logger.debug(
            f"Batch {batch_idx} minibatch {minibatch_idx} has {minibatch_token_count} tokens"
        )

        # Use trace context manager to capture router outputs
        with model.trace(encoded_minibatch):
            # Get attention mask to filter out padding tokens
            attention_mask = encoded_minibatch["attention_mask"]
            padding_mask = attention_mask.cpu().bool()
            padding_mask_flat = padding_mask.flatten()

            # Extract activations for each layer
            for layer_idx in tqdm(
                range(num_layers),
                desc=f"Batch {batch_idx} minibatch {minibatch_idx}",
                total=num_layers,
                leave=False,
                position=rank * 2 + 1,
            ):
                if (
                    ActivationKeys.ATTN_OUTPUT in activations_to_store
                    and layer_idx in layers_to_store
                ):
                    attn_output = model.attentions_output[layer_idx]
                    flattened_attn_output = attn_output.cpu()[padding_mask].save()
                    activations[str(ActivationKeys.ATTN_OUTPUT)][layer_idx].append(
                        flattened_attn_output.clone().detach()
                    )

                if (
                    ActivationKeys.ROUTER_LOGITS in activations_to_store
                    and layer_idx in router_layers
                ):
                    router_output = model.routers_output[layer_idx]

                    # Handle different router output formats
                    if isinstance(router_output, tuple):
                        if len(router_output) == 2:
                            router_scores, _router_indices = router_output
                        else:
                            raise ValueError(
                                f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                            )
                    else:
                        router_scores = router_output
                    logits = router_scores.cpu()[padding_mask_flat].save()

                    activations[str(ActivationKeys.ROUTER_LOGITS)][layer_idx].append(
                        logits.clone().detach()
                    )

                if (
                    ActivationKeys.MLP_OUTPUT in activations_to_store
                    and layer_idx in layers_to_store
                ):
                    mlp_output = model.mlps_output[layer_idx]
                    flattened_mlp_output = mlp_output.cpu()[padding_mask].save()
                    activations[str(ActivationKeys.MLP_OUTPUT)][layer_idx].append(
                        flattened_mlp_output.clone().detach()
                    )

                if (
                    ActivationKeys.LAYER_OUTPUT in activations_to_store
                    and layer_idx in layers_to_store
                ):
                    layer_output = model.layers_output[layer_idx]
                    flattened_layer_output = layer_output.cpu()[padding_mask].save()
                    activations[str(ActivationKeys.LAYER_OUTPUT)][layer_idx].append(
                        flattened_layer_output.clone().detach()
                    )

    # Return unstacked activations for disk worker to handle stacking
    return activations


def tokenizer_worker(
    model_name: str,
    dataset_name: str,
    context_length: int,
    tokens_per_file: int,
    main_queue: mp.Queue,
    stop_event: Any,  # mp.Event is not properly typed
    log_queue: mp.Queue,
    rank: int,
    world_size: int,
    completed_batches: set[int] | None = None,
    num_tokens: int = 1_000_000_000,  # 1B tokens
    log_level: str = "INFO",
) -> None:
    """Worker process for tokenizing text data."""

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Get model config and tokenizer
    model_config = get_model_config(model_name)

    # Import here to avoid circular imports
    from transformers import AutoTokenizer

    hf_name = model_config.hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)

    path = local_path if os.path.exists(local_path) else hf_name

    tokenizer = AutoTokenizer.from_pretrained(path)

    logger.info(f"Using tokenizer from {path}")
    # Get dataset function
    dataset_fn = get_dataset_fn(dataset_name)

    # Create dataset iterator
    dataset_iter = dataset_fn(tokenizer)

    # Initialize buffer and statistics
    buffer: deque[tuple[str, list[str]]] = deque()
    buffer_token_count = 0
    total_tokens = 0
    batch_idx = -1
    start_time = time.time()

    assert num_tokens > 0, "Total number of tokens to process must be greater than 0"

    if completed_batches is None:
        completed_batches = set()

    batch_skip_progress_bar = tqdm(
        desc="Skipping batches",
        leave=False,
    )

    logger.info(f"Starting tokenizer worker for {dataset_name}")
    # Process dataset
    try:
        for text_idx, text in enumerate(dataset_iter):
            if stop_event.is_set():
                break

            # Tokenize text
            tokens = tokenizer.tokenize(text)
            count = len(tokens)

            if text_idx % 1000 == 0:
                logger.debug(f"Tokenized text {text_idx} into {count} tokens")

            # Add to buffer
            buffer.append((text, tokens))
            buffer_token_count += count
            total_tokens += count

            # Check if we have enough tokens to create a batch for processing
            if buffer_token_count >= tokens_per_file or total_tokens >= num_tokens:
                current_buffer_token_count = buffer_token_count
                buffer_token_count = 0
                batch_idx += 1

                # skip if this batch is already completed or if this batch is not for this rank
                if batch_idx in completed_batches or batch_idx % world_size != rank:
                    logger.debug(f"Skipping batch {batch_idx}")
                    batch_skip_progress_bar.update(1)
                    log_queue.put({"tokenizer/skipping_batches": batch_idx})

                    buffer.clear()

                    continue

                # Create batch from all items in buffer
                logger.debug(
                    f"Creating batch {batch_idx} from {len(buffer)} items with {current_buffer_token_count} tokens"
                )
                batch = [buffer.popleft() for _ in range(len(buffer))]
                batch_texts, batch_tokens = tuple(zip(*batch, strict=False))

                # Create encoded batch
                logger.debug("Creating encoded batch")
                encoded_batch = tokenizer(
                    batch_texts,
                    padding=True,
                    return_tensors="pt",
                    max_length=context_length,
                    truncation=True,
                )
                tokens_sum = sum(len(tokens) for tokens in batch_tokens)

                # Put in queue
                logger.debug(f"Putting batch {batch_idx} in queue")
                main_queue.put(
                    (batch_idx, encoded_batch, batch_tokens, tokens_sum), block=True
                )

                # Log statistics
                elapsed = time.time() - start_time

                assert elapsed > 0, "Elapsed time is 0, this should never happen"

                logger.debug(f"Putting statistics in queue for batch {batch_idx}")
                log_queue.put(
                    {
                        "tokenizer/batch_idx": batch_idx,
                        "tokenizer/queue_size": main_queue.qsize(),
                        "tokenizer/tokens_in_batch": tokens_sum,
                        "tokenizer/total_tokens": total_tokens,
                        "tokenizer/progress": total_tokens / num_tokens,
                        "tokenizer/sequences_in_batch": len(batch_texts),
                        "tokenizer/buffer_size": len(buffer),
                        "tokenizer/tokens_per_second": total_tokens / elapsed,
                    }
                )
    finally:
        main_queue.put(None)


def multiplexer_worker(
    main_queue: mp.Queue,
    gpu_queues: list[mp.Queue],
    output_queue: mp.Queue,
    stop_event: Any,  # mp.Event is not properly typed
    gpu_busy: list[bool],
    log_queue: mp.Queue,
    log_level: str = "INFO",
) -> None:
    """Worker process that distributes batches to GPU queues based on load."""

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Initialize statistics
    total_batches = 0
    total_tokens = 0
    start_time = time.time()
    gpu_batch_counts = [0] * len(gpu_queues)  # Track batches sent to each GPU

    logger.info("Starting multiplexer worker")
    try:
        while not stop_event.is_set():
            # Get batch from main queue
            item = main_queue.get(block=True)
            if item is None:
                break

            batch_idx, encoded_batch, batch_tokens, tokens_count = item
            total_batches += 1
            total_tokens += tokens_count

            # Find GPU with smallest queue, considering if they're busy
            logger.debug("Finding GPU with smallest queue")
            queue_sizes = []
            for i, q in enumerate(gpu_queues):
                size = q.qsize()
                # Add penalty if GPU is busy
                size += gpu_busy[i]
                queue_sizes.append(size)

            min_queue_idx = th.argmin(th.tensor(queue_sizes)).item()

            # Put batch in the selected GPU queue
            logger.debug(f"Putting batch {batch_idx} in GPU {min_queue_idx} queue")
            gpu_queues[min_queue_idx].put(
                (batch_idx, encoded_batch, batch_tokens, tokens_count), block=True
            )

            # Update batch count for this GPU
            logger.debug(f"Updating batch count for GPU {min_queue_idx}")
            gpu_batch_counts[min_queue_idx] += 1

            # Log statistics
            elapsed = time.time() - start_time
            logger.debug(f"Putting statistics in queue for batch {batch_idx}")
            log_queue.put(
                {
                    "multiplexer/batch_idx": batch_idx,
                    "multiplexer/main_queue_size": main_queue.qsize(),
                    "multiplexer/total_batches": total_batches,
                    "multiplexer/total_tokens": total_tokens,
                    "multiplexer/tokens_per_second": total_tokens / elapsed
                    if elapsed > 0
                    else 0,
                    "multiplexer/elapsed_time": elapsed,
                    **{
                        f"multiplexer/gpu_{i}_queue_size": q.qsize()
                        for i, q in enumerate(gpu_queues)
                    },
                    **{
                        f"multiplexer/gpu_{i}_batch_count": count
                        for i, count in enumerate(gpu_batch_counts)
                    },
                }
            )

    finally:
        # Signal all GPU queues that we're done
        for q in gpu_queues:
            q.put(None)

        # Signal disk worker that we're done
        output_queue.put(None)


def gpu_worker(
    rank: int,
    device_ids: list[int],
    gpu_queue: mp.Queue,
    model_name: str,
    minibatch_size: int,
    experiment_name: str,
    gpu_available: bool,
    stop_event: Any,  # mp.Event is not properly typed
    gpu_busy: list[bool],  # Reference to multiplexer's gpu_busy list
    log_queue: mp.Queue,
    output_queue: mp.Queue,
    activations_to_store: frozenset[str] = ACTIVATION_KEYS,
    layers_to_store: set[int] | None = None,
    dtype: th.dtype = th.bfloat16,
    log_level: str = "INFO",
    device_type: DeviceType = "cuda",
) -> None:
    """Worker process for processing batches on a specific device."""
    logger.debug(
        f"Processing batch with activations to store: {activations_to_store} layers to store: {layers_to_store}"
    )

    extra_activations_to_store = activations_to_store - ACTIVATION_KEYS
    if extra_activations_to_store:
        raise ValueError(
            f"Unexpected activations to store: {extra_activations_to_store}"
        )

    current_activations_to_store = activations_to_store & ACTIVATION_KEYS

    if len(current_activations_to_store) == 0:
        raise ValueError(
            f"No activations to store. Available activations: {ACTIVATION_KEYS}"
        )

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Get model config
    model_config = get_model_config(model_name)

    hf_name = model_config.hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)
    path = local_path if os.path.exists(local_path) else hf_name

    logger.info(f"Using model from {path}")
    # Initialize model
    model = StandardizedTransformer(
        path,
        check_attn_probs_with_trace=False,
        device_map="cpu" if not gpu_available else "auto",
        torch_dtype=dtype,
    )
    logger.debug("Model initialized")
    layers_with_routers = set(model.layers_with_routers)
    top_k = model.router_probabilities.get_top_k()
    num_layers = len(assert_type(model.layers, Sized))

    if layers_to_store is None:
        # take the middle 20% of layers by default
        # this does NOT apply to router logits; those are stored at all layers
        num_layers_to_store = math.ceil(num_layers * 0.2)
        mid_layer = num_layers // 2
        start_layer_to_store = mid_layer - (num_layers_to_store // 2)
        layers_to_store = set(
            range(start_layer_to_store, start_layer_to_store + num_layers_to_store)
        )

    assert all(layer_idx < num_layers for layer_idx in layers_to_store), (
        f"Layers to store out of bounds for model with {num_layers} layers: {layers_to_store}"
    )

    # Create output directory
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    activations_dir = os.path.join(experiment_dir, ACTIVATION_DIRNAME)
    os.makedirs(activations_dir, exist_ok=True)

    # Initialize statistics
    total_batches = 0
    total_tokens = 0
    processing_times: list[float] = []
    start_time = time.time()

    if gpu_available:
        logger.info(
            f"Starting GPU worker {rank} on {', '.join(f'cuda:{device_id}' for device_id in device_ids)}"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(device_id) for device_id in device_ids
        )
    else:
        logger.info(f"Starting CPU worker {rank}")

    # Process batches
    while not stop_event.is_set():
        # Signal that we're ready for a new batch
        gpu_busy[rank] = False

        # Get batch from queue
        item = gpu_queue.get(block=True)

        logger.debug(f"Rank {rank} picked up batch from queue")

        # Signal that we're busy processing
        gpu_busy[rank] = True

        # Check if we're done
        if item is None:
            break

        # Process batch
        batch_idx, encoded_batch, batch_tokens, tokens_count = item

        # Move tensors to device
        if gpu_available:
            encoded_batch = {k: v.to(device_ids[0]) for k, v in encoded_batch.items()}

        # Process batch and get router logits
        logger.debug(f"Rank {rank} processing batch {batch_idx}")
        batch_start = time.time()
        with th.inference_mode():
            activations_raw = process_batch(
                encoded_batch,
                batch_idx,
                model,
                rank,
                minibatch_size,
                layers_with_routers,
                layers_to_store,
                activations_to_store=current_activations_to_store,
                device_type=device_type,
            )

        logger.debug(f"Rank {rank} processed batch {batch_idx}")

        # Put results in output queue for disk worker
        output = {
            "batch_idx": batch_idx,
            "topk": top_k,
            "tokens": batch_tokens,
            "layers": sorted(layers_to_store),
            "router_layers": sorted(layers_with_routers),
            "activations_raw": activations_raw,
        }
        logger.debug(f"Rank {rank} putting batch {batch_idx} in output queue")
        output_queue.put(output, block=True)

        # Update statistics
        batch_time = time.time() - batch_start
        processing_times.append(batch_time)
        total_batches += 1
        total_tokens += tokens_count
        elapsed = time.time() - start_time

        # Log statistics
        logger.debug(f"Rank {rank} logging statistics for batch {batch_idx}")
        log_queue.put(
            {
                f"gpu_{rank}/batch_idx": batch_idx,
                f"gpu_{rank}/queue_size": gpu_queue.qsize(),
                f"gpu_{rank}/batch_time": batch_time,
                f"gpu_{rank}/batch_tokens": tokens_count,
                f"gpu_{rank}/tokens_per_second": tokens_count / batch_time
                if batch_time > 0
                else 0,
                f"gpu_{rank}/total_batches": total_batches,
                f"gpu_{rank}/total_tokens": total_tokens,
                f"gpu_{rank}/avg_batch_time": sum(processing_times)
                / len(processing_times),
                f"gpu_{rank}/elapsed_time": elapsed,
            }
        )


async def stack_list_of_list_of_tensors(data: list[list[th.Tensor]]) -> th.Tensor:
    awaitable_concatenated_tensors = [
        asyncio.to_thread(th.cat, layer_activations, dim=0)
        for layer_activations in data
        if len(layer_activations) > 0
    ]
    concatenated_tensors = await asyncio.gather(*awaitable_concatenated_tensors)

    return th.stack(concatenated_tensors, dim=1)


async def disk_worker_async(
    output_queue: mp.Queue,
    experiment_name: str,
    stop_event: Any,  # mp.Event is not properly typed
    log_queue: mp.Queue,
    log_level: str = "INFO",
) -> None:
    """Worker process for saving activations to disk."""
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Create output directory
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    activations_dir = os.path.join(experiment_dir, ACTIVATION_DIRNAME)
    os.makedirs(activations_dir, exist_ok=True)

    # Initialize statistics
    total_batches = 0
    start_time = time.time()

    logger.info("Starting disk worker")

    while not stop_event.is_set():
        output = output_queue.get(block=True)

        # Check if we're done
        if output is None:
            break

        batch_idx = output.pop("batch_idx")
        activations_raw = output.pop("activations_raw")

        logger.debug(f"Disk worker processing batch {batch_idx}")

        # Stack activations asynchronously
        stacked_activation_keys = list(activations_raw.keys())
        stacked_activation_awaitables = [
            stack_list_of_list_of_tensors(activations_raw[activation_key])
            for activation_key in stacked_activation_keys
        ]
        stacked_activation_tensors = await asyncio.gather(
            *stacked_activation_awaitables
        )
        stacked_activations = dict(
            zip(stacked_activation_keys, stacked_activation_tensors, strict=True)
        )

        # Save results
        output_path = os.path.join(activations_dir, f"{batch_idx}.pt")
        output.update(stacked_activations)

        logger.debug(f"Disk worker saving batch {batch_idx} to {output_path}")
        th.save(output, output_path)

        # Update statistics
        total_batches += 1
        elapsed = time.time() - start_time

        # Log statistics
        log_queue.put(
            {
                "disk/batch_idx": batch_idx,
                "disk/total_batches": total_batches,
                "disk/elapsed_time": elapsed,
            }
        )


def disk_worker(
    output_queue: mp.Queue,
    experiment_name: str,
    stop_event: Any,  # mp.Event is not properly typed
    log_queue: mp.Queue,
    log_level: str = "INFO",
) -> None:
    """Wraps disk_worker_async which uses asyncio to parallelize activation stacking."""
    asyncio.run(
        disk_worker_async(
            output_queue, experiment_name, stop_event, log_queue, log_level
        )
    )


def find_completed_batches(experiment_dir: str) -> set[int]:
    """Find all completed batch indices in the experiment directory.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        A set of completed batch indices.
    """
    activations_dir = os.path.join(experiment_dir, ACTIVATION_DIRNAME)
    if not os.path.exists(activations_dir):
        return set()

    completed_batches = set()
    for filename in os.listdir(activations_dir):
        if filename.endswith(".pt"):
            # Extract batch index from filename (format: {batch_idx}.pt)
            batch_idx = int(filename.split(".")[0])
            completed_batches.add(batch_idx)

    return completed_batches


CUDA_VISIBLE_DEVICES_REGEX = re.compile(r"^(([0-9]+,)+[0-9]+|[0-9]*)$")


@arguably.command()
def get_router_activations(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    context_length: int = 2048,
    minibatch_size: int = 2,
    gpus_per_worker: int = 2,
    cuda_devices: str = "",
    tokens_per_file: int = 5_000,
    num_tokens: int = 1_000_000_000,  # 1B tokens
    resume: bool = True,
    name: str | None = None,
    activations_to_store: list[str] | None = None,
    layers_to_store: list[int] | None = None,
    dtype: str = "bf16",
    log_level: str = "INFO",
    device_type: DeviceType = "cuda",
) -> None:
    """
    Extract router activations from a model using multiple devices.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        context_length: Context length for processing
        minibatch_size: Batch size for processing on each device
        gpus_per_worker: Number of devices to shard the model across
        cuda_devices: Comma-separated list of device indices to use. If empty, defaults to CUDA_VISIBLE_DEVICES environment variable or CPU if it's not set.
        tokens_per_file: Target number of tokens per output file
        num_tokens: Number of tokens to process
        resume: Whether to resume from a previous run
        name: Custom name for the experiment
        device_type: Device type ("cuda" or "xpu", defaults to "cuda")
    """
    print(f"Running with log level: {log_level}")

    # Validate device type
    assert_device_type(device_type)
    backend = get_backend(device_type)

    torch_dtype = get_dtype(dtype)

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.debug(f"Running with log level: {log_level}")

    cuda_devices_raw = cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # make sure device list is of the form "0,1,...,n"
    if not CUDA_VISIBLE_DEVICES_REGEX.match(cuda_devices_raw):
        raise ValueError(
            f"Device list must be of the form '0,1,...,n' or empty: \"{cuda_devices_raw}\""
        )

    if cuda_devices_raw == "":
        cuda_device_ids = list(range(backend.device_count()))
    else:
        cuda_device_ids = [int(i) for i in cuda_devices_raw.split(",")]

    num_gpus = len(cuda_device_ids)
    if num_gpus == 0:
        logger.warning(f"No {device_type} devices found")

    num_gpu_workers = num_gpus // gpus_per_worker

    if num_gpu_workers > 0:
        device_ids = cuda_device_ids
        num_workers = num_gpu_workers
        gpu_available = True
    else:
        device_ids = [0]
        num_workers = 1
        gpu_available = False

    worker_device_map = (
        {
            worker_idx: device_ids[gpu_start_idx:gpu_end_idx]
            for worker_idx, (gpu_start_idx, gpu_end_idx) in enumerate(
                pairwise(range(0, num_gpus + 1, gpus_per_worker))
            )
        }
        if gpu_available
        else {worker_idx: [0] for worker_idx in range(num_workers)}
    )

    extra_gpus = num_gpus % gpus_per_worker
    if extra_gpus > 0:
        logger.warning(
            f"{extra_gpus} extra GPUs will be ignored due to gpus_per_worker={gpus_per_worker}"
        )

    if not activations_to_store:
        activations_to_store = [
            str(ActivationKeys.ROUTER_LOGITS),
            str(ActivationKeys.MLP_OUTPUT),
            str(ActivationKeys.LAYER_OUTPUT),
        ]

    layers_to_store_set: set[int] | None
    if not layers_to_store:
        layers_to_store_set = None
    elif isinstance(layers_to_store, list):
        layers_to_store_set = set(layers_to_store)

    if not gpu_available:
        logger.info("Using CPU only")
    else:
        logger.info(f"Using {num_gpus} {device_type} devices: {device_ids}")

    # Create experiment configuration
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "context_length": context_length,
        "num_tokens": num_tokens,
        "tokens_per_file": tokens_per_file,
        "num_gpus": num_gpus,
        "gpu_available": gpu_available,
        "device_ids": device_ids,
        "gpus_per_worker": gpus_per_worker,
        "worker_device_map": worker_device_map,
        "dtype": dtype,
    }

    # Generate experiment name if not provided
    if name is None:
        name = get_experiment_name(
            model_name=model_name,
            dataset_name=dataset_name,
            context_length=context_length,
            tokens_per_file=tokens_per_file,
        )

    # Create experiment directories
    experiment_dir = os.path.join(OUTPUT_DIR, name)
    activations_dir = os.path.join(experiment_dir, ACTIVATION_DIRNAME)

    os.makedirs(activations_dir, exist_ok=True)

    # Verify configuration against existing one (if any)
    verify_config(config, experiment_dir)

    # Save configuration
    save_config(config, experiment_dir)

    logger.info(f"Experiment name: {name}")

    # Initialize WandB
    wandb.init(project="router_activations", resume="allow", config=config)

    # Find completed batches if resuming
    completed_batches = set()
    if resume:
        completed_batches = find_completed_batches(experiment_dir)
        if completed_batches:
            max_batch_idx = max(completed_batches) if completed_batches else 0
            logger.info(
                f"Found {len(completed_batches)} completed batches, max batch idx: {max_batch_idx}"
            )
            logger.info("Will skip completed batches and process missing ones")
            wandb.log(
                {
                    "resume/max_batch_idx": max_batch_idx,
                    "resume/completed_count": len(completed_batches),
                }
            )

    # Initialize multiprocessing resources
    mp.set_start_method("spawn", force=True)

    # Create queues and events
    main_queue = mp.Queue(
        maxsize=MAIN_QUEUE_MAXSIZE
    )  # Buffer between tokenizer and multiplexer
    gpu_queues = [mp.Queue(maxsize=GPU_QUEUE_MAXSIZE) for _ in range(num_workers)]
    output_queue = mp.Queue(
        maxsize=OUTPUT_QUEUE_MAXSIZE
    )  # For sending processed batches to disk worker
    log_queue = mp.Queue()  # For sending logs back to main process
    stop_event = mp.Event()  # For signaling processes to stop

    # Create shared state for GPU busy status
    manager = mp.Manager()
    gpu_busy = manager.list([False] * num_workers)

    # Create and start processes
    processes = []

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Start tokenizer worker
    logger.info("Starting tokenizer worker")
    tokenizer_proc = mp.Process(
        target=tokenizer_worker,
        args=(
            model_name,
            dataset_name,
            context_length,
            tokens_per_file,
            main_queue,
            stop_event,
            log_queue,
            rank,
            world_size,
            completed_batches,
            num_tokens,
            log_level,
        ),
    )
    tokenizer_proc.start()
    processes.append(tokenizer_proc)

    # Start multiplexer worker
    logger.info("Starting multiplexer worker")
    multiplexer_proc = mp.Process(
        target=multiplexer_worker,
        args=(
            main_queue,
            gpu_queues,
            output_queue,
            stop_event,
            gpu_busy,
            log_queue,
            log_level,
        ),
    )
    multiplexer_proc.start()
    processes.append(multiplexer_proc)

    # Start disk worker
    logger.info("Starting disk worker")
    disk_proc = mp.Process(
        target=disk_worker,
        args=(output_queue, name, stop_event, log_queue, log_level),
    )
    disk_proc.start()
    processes.append(disk_proc)

    # Start device workers
    for rank, device_ids in worker_device_map.items():
        if not gpu_available:
            logger.info(f"Starting CPU worker {rank}")
        else:
            logger.info(
                f"Starting {device_type} worker {rank} on {', '.join(f'{device_type}:{device_id}' for device_id in device_ids)}"
            )

        logger.debug(f"Storing activations: {activations_to_store}")
        logger.debug(f"Storing layers: {layers_to_store_set}")
        gpu_proc = mp.Process(
            target=gpu_worker,
            args=(
                rank,
                device_ids,
                gpu_queues[rank],
                model_name,
                minibatch_size,
                name,
                gpu_available,
                stop_event,
                gpu_busy,
                log_queue,
                output_queue,
                set(activations_to_store),
                layers_to_store_set,
                torch_dtype,
                log_level,
                device_type,
            ),
        )
        gpu_proc.start()
        processes.append(gpu_proc)

    try:
        processes_are_running = True
        while processes_are_running:
            try:
                log = log_queue.get(block=True, timeout=10.0)
                wandb.log(log)
            except queue.Empty:
                warnings.warn(
                    "No logs received from log queue after 10 seconds", stacklevel=2
                )

            processes_are_running = any(proc.is_alive() for proc in processes)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, stopping all processes...")
        stop_event.set()

        # Wait for processes to terminate
        for proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


if __name__ == "__main__":
    arguably.run()
