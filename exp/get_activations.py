from collections import deque
import gc
import math
import os
import queue
import sys
import time
from typing import Any
import warnings

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import torch.multiprocessing as mp
from tqdm import tqdm
import trackio as wandb
import yaml

from core.data import DATASETS
from core.model import MODELS
from core.slurm import SlurmEnv, get_slurm_env
from exp import ACTIVATION_DIRNAME, MODEL_DIRNAME, OUTPUT_DIR

# Constants
CONFIG_FILENAME = "config.yaml"

# within-node parallelism constants
MAIN_QUEUE_MAXSIZE = 10
GPU_QUEUE_MAXSIZE = 2


def get_experiment_name(model_name: str, dataset_name: str, **kwargs) -> str:
    """Generate a unique experiment name based on configuration parameters."""
    base_name = f"{model_name}_{dataset_name}"

    # Track which keys are being filtered out
    ignored_keys = {"device", "resume"}
    filtered_keys = set()

    # Add any additional parameters that might affect the experiment
    param_items = []
    for k, v in sorted(kwargs.items()):
        if k in ignored_keys or k.startswith("_"):
            filtered_keys.add(k)
            continue
        param_items.append(f"{k}={v}")

    # Warn about filtered keys
    if filtered_keys:
        warnings.warn(
            f"The following keys were excluded from the experiment name: {filtered_keys}",
            stacklevel=2,
        )

    param_str = "_".join(param_items)

    if param_str:
        base_name = f"{base_name}_{param_str}"

    return base_name


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


ACTIVATION_KEYS = frozenset(
    {
        "attn_output",
        "router_logits",
        "mlp_output",
        "layer_output",
    }
)


def process_batch(
    encoded_batch: dict,
    batch_idx: int,
    model: StandardizedTransformer,
    gpu_minibatch_size: int,
    router_layers: set[int],
    activations_to_store: set[str] = ACTIVATION_KEYS,
) -> dict[str, th.Tensor]:
    """Process a batch of texts through the model and extract router logits.

    Args:
        encoded_batch: Encoded batch from tokenizer with padding.
        batch_idx: Index of the batch.
        model: Model to process the batch.
        gpu_minibatch_size: Size of the minibatch to process on each GPU.
        router_layers: Set of router layer indices to extract.
        stored_activations: Activations to score

    Returns:
        Dictionary of activations.
    """
    logger.debug(f"Processing batch {batch_idx} with activations to store: {activations_to_store}")

    batch_size = encoded_batch["input_ids"].shape[0]

    if gpu_minibatch_size <= 0:
        gpu_minibatch_size = batch_size
    else:
        gpu_minibatch_size = min(gpu_minibatch_size, batch_size)

    num_minibatches = math.ceil(batch_size / gpu_minibatch_size)
    num_layers = len(model.layers)

    # Extract activations
    activations = {
        activation_key: [[] for _ in range(num_layers)]
        for activation_key in activations_to_store
    }

    for minibatch_idx in tqdm(
        range(num_minibatches), desc=f"Batch {batch_idx}", total=num_minibatches, leave=False,
    ):
        th.cuda.empty_cache()
        gc.collect()

        minibatch_start = minibatch_idx * gpu_minibatch_size
        minibatch_end = min(minibatch_start + gpu_minibatch_size, batch_size)
        encoded_minibatch = {
            k: v[minibatch_start:minibatch_end].to(model.device)
            for k, v in encoded_batch.items()
        }

        minibatch_token_count = encoded_minibatch["attention_mask"].sum().item()
        logger.debug(f"Batch {batch_idx} minibatch {minibatch_idx} has {minibatch_token_count} tokens")

        # Use trace context manager to capture router outputs
        with model.trace(encoded_minibatch):
            # Get attention mask to filter out padding tokens
            attention_mask = encoded_minibatch["attention_mask"]
            padding_mask = attention_mask.cpu().bool().flatten()

            # Extract activations for each layer
            for layer_idx, _layer in tqdm(
                enumerate(model.layers),
                desc=f"Batch {batch_idx} minibatch {minibatch_idx}",
                total=len(model.layers),
                leave=False,
            ):
                if "attn_output" in activations_to_store:
                    attn_output = model.attentions_output[layer_idx]
                    activations["attn_output"][layer_idx].append(
                        attn_output.output.cpu().clone().detach()
                    )

                if (
                    "router_logits" in activations_to_store
                    and layer_idx in router_layers
                ):
                    router_output = model.routers_output[layer_idx]

                    # Handle different router output formats
                    match router_output:
                        case (router_scores, _router_indices):
                            logits = router_scores.cpu()[padding_mask].save()
                        case tuple():
                            raise ValueError(
                                f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                            )
                        case router_scores:
                            logits = router_scores.cpu()[padding_mask].save()

                    activations["router_logits"][layer_idx].append(
                        logits.cpu().clone().detach()
                    )

                if "mlp_output" in activations_to_store:
                    mlp_output = model.mlps_output[layer_idx]
                    activations["mlp_output"][layer_idx].append(
                        mlp_output.cpu().clone().detach()
                    )

                if "layer_output" in activations_to_store:
                    layer_output = model.layers_output[layer_idx]
                    activations["layer_output"][layer_idx].append(
                        layer_output.cpu().clone().detach()
                    )

    # Stack logits across minibatches and layers
    activations = {
        activation_key: [
            th.cat(activation_minibatches, dim=0)
            for activation_minibatches in activations_by_layer
        ]
        for activation_key, activations_by_layer in activations.items()
    }
    activations = {
        activation_key: th.stack(activations_by_layer, dim=1)
        for activation_key, activations_by_layer in activations.items()
    }

    return activations


def tokenizer_worker(
    model_name: str,
    dataset_name: str,
    context_length: int,
    tokens_per_file: int,
    main_queue: mp.Queue,
    stop_event: Any,  # mp.Event is not properly typed
    log_queue: mp.Queue,
    slurm_env: SlurmEnv,
    resume_from_batch: int = 0,
    num_tokens: int = 1_000_000_000,  # 1B tokens
    log_level: str = "INFO",
) -> None:
    """Worker process for tokenizing text data."""

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Get model config and tokenizer
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    # Import here to avoid circular imports
    from transformers import AutoTokenizer

    hf_name = MODELS[model_name].hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)

    path = local_path if os.path.exists(local_path) else hf_name

    tokenizer = AutoTokenizer.from_pretrained(path)

    logger.info(f"Using tokenizer from {path}")
    # Get dataset function
    dataset_fn = DATASETS.get(dataset_name)
    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Create dataset iterator
    dataset_iter = dataset_fn(tokenizer)

    # Initialize buffer and statistics
    buffer: deque[tuple[str, list[str]]] = deque()
    buffer_token_count = 0
    total_tokens = 0
    batch_idx = -1
    start_time = time.time()

    assert num_tokens > 0, "Total number of tokens to process must be greater than 0"

    slurm_rank = slurm_env.world_rank
    slurm_world_size = slurm_env.world_size

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

                if batch_idx < resume_from_batch or batch_idx % slurm_world_size != slurm_rank:
                    logger.debug(f"Skipping batch {batch_idx}")
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
            item = main_queue.get()
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


def gpu_worker(
    rank: int,
    device_ids: list[int],
    gpu_queue: mp.Queue,
    model_name: str,
    gpu_minibatch_size: int,
    experiment_name: str,
    cpu_only: bool,
    stop_event: Any,  # mp.Event is not properly typed
    gpu_busy: list[bool],  # Reference to multiplexer's gpu_busy list
    log_queue: mp.Queue,
    activations_to_store: set[str] = ACTIVATION_KEYS,
    log_level: str = "INFO",
) -> None:
    """Worker process for processing batches on a specific GPU."""
    logger.debug(f"Processing batch with activations to store: {activations_to_store}")

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
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    hf_name = MODELS[model_name].hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)
    path = local_path if os.path.exists(local_path) else hf_name

    logger.info(f"Using model from {path}")
    # Initialize model
    model = StandardizedTransformer(
        path,
        check_attn_probs_with_trace=False,
        device_map="auto",
    )
    logger.debug("Model initialized")
    layers_with_routers = model.layers_with_routers
    top_k = model.router_probabilities.get_top_k()

    # Create output directory
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    activations_dir = os.path.join(experiment_dir, ACTIVATION_DIRNAME)
    os.makedirs(activations_dir, exist_ok=True)

    # Initialize statistics
    total_batches = 0
    total_tokens = 0
    processing_times: list[float] = []
    start_time = time.time()

    logger.info(
        f"Starting GPU worker {rank} on {', '.join(f'cuda:{device_id}' for device_id in device_ids)}"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(device_id) for device_id in device_ids
    )

    # Process batches
    while not stop_event.is_set():
        # Signal that we're ready for a new batch
        gpu_busy[rank] = False

        # Get batch from queue
        item = gpu_queue.get()

        logger.debug(f"Rank {rank} picked up batch from queue")

        # Signal that we're busy processing
        gpu_busy[rank] = True

        # Check if we're done
        if item is None:
            break

        # Process batch
        batch_idx, encoded_batch, batch_tokens, tokens_count = item

        # Move tensors to device
        if not cpu_only:
            encoded_batch = {
                k: v.to(device_ids[0]) for k, v in encoded_batch.items()
            }

        # Process batch and get router logits
        logger.debug(f"Rank {rank} processing batch {batch_idx}")
        batch_start = time.time()
        with th.inference_mode():
            activations = process_batch(
                encoded_batch,
                batch_idx,
                model,
                gpu_minibatch_size,
                layers_with_routers,
                activations_to_store=current_activations_to_store,
            )

        logger.debug(f"Rank {rank} processed batch {batch_idx}")

        # Save results
        output_path = os.path.join(activations_dir, f"{batch_idx}.pt")
        output = {
            "topk": top_k,
            "tokens": batch_tokens,
            **activations,
        }
        logger.debug(f"Rank {rank} saving results to {output_path}")
        th.save(output, output_path)

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


@arguably.command()
def get_router_activations(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    context_length: int = 2048,
    gpu_minibatch_size: int = 2,
    gpus_per_worker: int = 2,
    cuda_devices: str = "",
    tokens_per_file: int = 20_000,
    num_tokens: int = 1_000_000_000,  # 1B tokens
    resume: bool = True,
    name: str | None = None,
    activations_to_store: list[str] | None = None,
    log_level: str = "INFO",
) -> None:
    """
    Extract router activations from a model using multiple GPUs.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        context_length: Context length for processing
        gpu_minibatch_size: Batch size for processing on each GPU
        gpus_per_worker: Number of GPUs to shard the model across
        cuda_devices: Comma-separated list of CUDA devices to use. If empty, defaults to CUDA_VISIBLE_DEVICES environment variable or CPU if it's not set.
        tokens_per_file: Target number of tokens per output file
        num_tokens: Number of tokens to process
        resume: Whether to resume from a previous run
        name: Custom name for the experiment
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Detect SLURM environment
    slurm_env = get_slurm_env()
    if slurm_env.is_slurm:
        logger.info(f"Running in SLURM environment: rank {slurm_env.world_rank}/{slurm_env.world_size}")
        logger.info(f"Node: {slurm_env.node_rank}/{slurm_env.num_nodes}, Local rank: {slurm_env.local_rank}")
    else:
        logger.info("Running in local environment (not SLURM)")

    cuda_devices_raw = cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cpu_only = cuda_devices_raw == ""
    device_ids = [int(i) for i in cuda_devices_raw.split(",")] if not cpu_only else [0]

    world_size = len(device_ids)
    if world_size == 0:
        raise ValueError(f"Unable to parse CUDA devices: {device_ids}")

    if not activations_to_store:
        activations_to_store = ["router_logits", "mlp_output", "layer_output"]

    if cpu_only:
        logger.info("Using CPU only")
    else:
        logger.info(f"Using {world_size} GPUs: {device_ids}")

    num_workers = world_size // gpus_per_worker
    worker_gpu_map = (
        {
            i: device_ids[i * gpus_per_worker : (i + 1) * gpus_per_worker]
            for i in range(num_workers)
        }
        if not cpu_only
        else {i: [0] for i in range(num_workers)}
    )

    extra_gpus = world_size % gpus_per_worker
    if extra_gpus > 0:
        logger.warning(
            f"{extra_gpus} extra GPUs will be ignored due to gpus_per_worker={gpus_per_worker}"
        )

    # Create experiment configuration
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "context_length": context_length,
        "num_tokens": num_tokens,
        "tokens_per_file": tokens_per_file,
        "world_size": world_size,
        "cpu_only": cpu_only,
        "device_ids": device_ids,
        "gpus_per_worker": gpus_per_worker,
        "worker_gpu_map": worker_gpu_map,
        "slurm_env": slurm_env,
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(activations_dir, exist_ok=True)

    # Verify configuration against existing one (if any)
    verify_config(config, experiment_dir)

    # Save configuration
    save_config(config, experiment_dir)

    logger.info(f"Experiment name: {name}")

    # Initialize WandB
    if slurm_env.global_rank == 0:
        wandb.init(project="router_activations", resume="allow", config=config)

    # Find completed batches if resuming
    resume_batch_idx = 0
    if resume:
        completed_batches = find_completed_batches(experiment_dir)
        if completed_batches:
            max_batch_idx = max(completed_batches) if completed_batches else 0
            resume_batch_idx = max_batch_idx + 1
            logger.info(f"Resuming from batch {resume_batch_idx}")
            if slurm_env.global_rank == 0:
                wandb.log({"resume/batch_idx": resume_batch_idx})

    # Initialize multiprocessing resources
    mp.set_start_method("spawn", force=True)

    # Create queues and events
    main_queue = mp.Queue(maxsize=MAIN_QUEUE_MAXSIZE)  # Buffer between tokenizer and multiplexer
    gpu_queues = [mp.Queue(maxsize=GPU_QUEUE_MAXSIZE) for _ in range(num_workers)]
    log_queue = mp.Queue()  # For sending logs back to main process
    stop_event = mp.Event()  # For signaling processes to stop

    # Create shared state for GPU busy status
    manager = mp.Manager()
    gpu_busy = manager.list([False] * world_size)

    # Create and start processes
    processes = []

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
            slurm_env,
            resume_batch_idx,
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
        args=(main_queue, gpu_queues, stop_event, gpu_busy, log_queue, log_level),
    )
    multiplexer_proc.start()
    processes.append(multiplexer_proc)

    # Start GPU workers
    for rank, device_ids in worker_gpu_map.items():
        if cpu_only:
            logger.info(f"Starting CPU worker {rank}")
        else:
            logger.info(
                f"Starting GPU worker {rank} on {', '.join(f'cuda:{device_id}' for device_id in device_ids)}"
            )

        logger.debug(f"Storing activations: {activations_to_store}")
        gpu_proc = mp.Process(
            target=gpu_worker,
            args=(
                rank,
                device_ids,
                gpu_queues[rank],
                model_name,
                gpu_minibatch_size,
                name,
                cpu_only,
                stop_event,
                gpu_busy,
                log_queue,
                set(activations_to_store),
                log_level,
            ),
        )
        gpu_proc.start()
        processes.append(gpu_proc)

    try:
        processes_are_running = True
        while processes_are_running:
            try:
                log = log_queue.get(block=True, timeout=10.0)
                if slurm_env.global_rank == 0:
                    wandb.log(log)
                else:
                    pass
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
