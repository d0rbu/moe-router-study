from collections import deque
import os
import time
from typing import Any, TypeVar
import warnings

import arguably
from nnterp import StandardizedTransformer
import torch as th
import torch.multiprocessing as mp
import trackio as wandb
import yaml

from core.data import DATASETS
from core.model import MODELS
from exp import OUTPUT_DIR

# Constants
ROUTER_LOGITS_DIRNAME = "router_logits"
CONFIG_FILENAME = "config.yaml"

# Type definitions
T = TypeVar("T")


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
        if current_value != saved_value:
            mismatches[key] = (saved_value, current_value)

    if mismatches:
        mismatch_str = "\n".join(
            f"  - {key}: saved={saved} vs current={current}"
            for key, (saved, current) in mismatches.items()
        )
        raise ValueError(
            f"Configuration mismatch with existing experiment:\n{mismatch_str}"
        )


def process_batch(
    encoded_batch: dict,
    tokenized_batch: list[list[str]],
    model: StandardizedTransformer,
    router_layers: list[int],
) -> tuple[th.Tensor, list[list[str]]]:
    """Process a batch of texts through the model and extract router logits.

    Args:
        encoded_batch: Encoded batch from tokenizer with padding.
        tokenized_batch: List of tokenized sequences (as strings).
        model: Model to process the batch.
        router_layers: List of router layer indices to extract.

    Returns:
        Tuple of (router_logits, tokenized_batch).
    """
    # Move tensors to model device
    encoded_batch = {k: v.to(model.device) for k, v in encoded_batch.items()}

    # Extract router logits
    router_logits = []

    # Use trace context manager to capture router outputs
    with model.trace(encoded_batch):
        # Get attention mask to filter out padding tokens
        attention_mask = encoded_batch["attention_mask"]
        padding_mask = attention_mask.bool().flatten()

        # Extract router logits for each layer
        for layer_idx in router_layers:
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

            router_logits.append(logits.clone().detach())

    # Stack router logits across layers
    router_logits_tensor = th.stack(router_logits, dim=1)

    return router_logits_tensor, tokenized_batch


def tokenizer_worker(
    model_name: str,
    dataset_name: str,
    tokens_per_file: int,
    main_queue: mp.Queue,
    stop_event: Any,  # mp.Event is not properly typed
    wandb_run_id: str,
    resume_from_batch: int = 0,
) -> None:
    """Worker process for tokenizing text data."""
    # Initialize wandb
    wandb.init(
        project="router_activations",
        # Use id parameter as that's what wandb expects
        id=wandb_run_id,  # type: ignore
        resume="allow",
        name=f"tokenizer-{dataset_name}-{model_name}",
    )

    # Get model config and tokenizer
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    # Import here to avoid circular imports
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_name)

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

    # Process dataset
    try:
        for text in dataset_iter:
            if stop_event.is_set():
                break

            # Tokenize text
            tokens = tokenizer.tokenize(text)
            count = len(tokens)

            # Add to buffer
            buffer.append((text, tokens))
            buffer_token_count += count
            total_tokens += count

            # Check if we have enough tokens to create a batch for processing
            if buffer_token_count >= tokens_per_file:
                buffer_token_count = 0
                batch_idx += 1

                if batch_idx < resume_from_batch:
                    wandb.log({"tokenizer/skipping_batches": batch_idx})
                    continue

                # Create batch from all items in buffer
                batch = [buffer.popleft() for _ in range(len(buffer))]
                batch_texts, batch_tokens = tuple(zip(*batch, strict=False))

                # Create encoded batch
                encoded_batch = tokenizer(
                    batch_texts, padding=True, return_tensors="pt"
                )
                tokens_sum = sum(len(tokens) for tokens in batch_tokens)

                # Put in queue
                main_queue.put(
                    (batch_idx, encoded_batch, batch_tokens, tokens_sum), block=True
                )

                # Log statistics
                elapsed = time.time() - start_time
                wandb.log(
                    {
                        "tokenizer/batch_idx": batch_idx,
                        "tokenizer/queue_size": main_queue.qsize(),
                        "tokenizer/tokens_in_batch": tokens_sum,
                        "tokenizer/total_tokens": total_tokens,
                        "tokenizer/sequences_in_batch": len(batch_texts),
                        "tokenizer/buffer_size": len(buffer),
                        "tokenizer/tokens_per_second": total_tokens / elapsed
                        if elapsed > 0
                        else 0,
                    }
                )

    except Exception as e:
        wandb.log({"tokenizer/error": str(e)})
        print(f"Tokenizer worker error: {e}")
    finally:
        # If there are remaining items in the buffer, send them as a final batch
        if buffer and not stop_event.is_set():
            batch = [buffer.popleft() for _ in range(len(buffer))]
            batch_texts, batch_tokens = tuple(zip(*batch, strict=False))

            # Create encoded batch
            encoded_batch = tokenizer(batch_texts, padding=True, return_tensors="pt")
            tokens_sum = sum(len(tokens) for tokens in batch_tokens)

            # Put in queue
            main_queue.put(
                (batch_idx, encoded_batch, batch_tokens, tokens_sum), block=True
            )

            elapsed = time.time() - start_time
            wandb.log(
                {
                    "tokenizer/batch_idx": batch_idx,
                    "tokenizer/queue_size": main_queue.qsize(),
                    "tokenizer/tokens_in_batch": tokens_sum,
                    "tokenizer/total_tokens": total_tokens,
                    "tokenizer/sequences_in_batch": len(batch_texts),
                    "tokenizer/buffer_size": 0,
                    "tokenizer/tokens_per_second": total_tokens / elapsed
                    if elapsed > 0
                    else 0,
                    "tokenizer/finished": True,
                }
            )

        # Signal that we're done
        main_queue.put(None)
        wandb.finish()


def multiplexer_worker(
    main_queue: mp.Queue,
    gpu_queues: list[mp.Queue],
    stop_event: Any,  # mp.Event is not properly typed
    wandb_run_id: str,
    gpu_busy: list[bool],
) -> None:
    """Worker process that distributes batches to GPU queues based on load."""
    # Initialize wandb
    wandb.init(
        project="router_activations",
        # Use id parameter as that's what wandb expects
        id=wandb_run_id,  # type: ignore
        resume="allow",
        name="multiplexer",
    )

    # Initialize statistics
    total_batches = 0
    total_tokens = 0
    start_time = time.time()
    gpu_batch_counts = [0] * len(gpu_queues)  # Track batches sent to each GPU

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
            queue_sizes = []
            for i, q in enumerate(gpu_queues):
                size = q.qsize()
                # Add penalty if GPU is busy
                size += gpu_busy[i]
                queue_sizes.append(size)

            # Use torch for argmin
            min_queue_idx = th.argmin(th.tensor(queue_sizes)).item()

            # Put batch in the selected GPU queue
            gpu_queues[min_queue_idx].put(
                (batch_idx, encoded_batch, batch_tokens, tokens_count), block=True
            )

            # Update batch count for this GPU
            gpu_batch_counts[min_queue_idx] += 1

            # Log statistics
            elapsed = time.time() - start_time
            wandb.log(
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

    except Exception as e:
        wandb.log({"multiplexer/error": str(e)})
        print(f"Multiplexer worker error: {e}")
    finally:
        # Signal all GPU queues that we're done
        for q in gpu_queues:
            q.put(None)
        wandb.finish()


def gpu_worker(
    rank: int,
    device_id: int,
    gpu_queue: mp.Queue,
    model_name: str,
    experiment_name: str,
    cpu_only: bool,
    stop_event: Any,  # mp.Event is not properly typed
    wandb_run_id: str,
    gpu_busy: list[bool],  # Reference to multiplexer's gpu_busy list
) -> None:
    """Worker process for processing batches on a specific GPU."""
    try:
        # Initialize wandb
        wandb.init(
            project="router_activations",
            # Use id parameter as that's what wandb expects
            id=wandb_run_id,  # type: ignore
            resume="allow",
            name=f"gpu-{device_id}" if not cpu_only else "cpu",
        )

        device_map = f"cuda:{device_id}" if not cpu_only else "cpu"

        # Get model config
        model_config = MODELS.get(model_name)
        if model_config is None:
            raise ValueError(f"Model {model_name} not found")

        # Initialize model
        model = StandardizedTransformer(
            model_config.hf_name,
            check_attn_probs_with_trace=False,
            device_map=device_map,
        )
        layers_with_routers = model.layers_with_routers
        top_k = model.router_probabilities.get_top_k()

        # Create output directory
        experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
        router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
        os.makedirs(router_logits_dir, exist_ok=True)

        # Initialize statistics
        total_batches = 0
        total_tokens = 0
        processing_times: list[float] = []
        start_time = time.time()

        # Process batches
        while not stop_event.is_set():
            # Signal that we're ready for a new batch
            gpu_busy[rank] = False

            # Get batch from queue
            item = gpu_queue.get()

            # Signal that we're busy processing
            gpu_busy[rank] = True

            # Check if we're done
            if item is None:
                break

            batch_idx, encoded_batch, batch_tokens, tokens_count = item

            # Move encoded batch to device
            for key in encoded_batch:
                if isinstance(encoded_batch[key], th.Tensor):
                    encoded_batch[key] = encoded_batch[key].to(model.device)

            # Process batch
            batch_idx, encoded_batch, batch_tokens, tokens_count = item

            # Move tensors to device
            encoded_batch = {k: v.to(model.device) for k, v in encoded_batch.items()}

            # Process batch and get router logits
            batch_start = time.time()
            with th.inference_mode():
                router_logits_tensor, _ = process_batch(
                    encoded_batch, batch_tokens, model, layers_with_routers
                )

            # Save results
            output_path = os.path.join(router_logits_dir, f"{batch_idx}.pt")
            output = {
                "topk": top_k,
                "router_logits": router_logits_tensor,
                "tokens": batch_tokens,
            }
            th.save(output, output_path)

            # Update statistics
            batch_time = time.time() - batch_start
            processing_times.append(batch_time)
            total_batches += 1
            total_tokens += tokens_count
            elapsed = time.time() - start_time

            # Log statistics
            wandb.log(
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

    except Exception as e:
        wandb.log({f"gpu_{rank}/error": str(e)})
        print(f"GPU {rank} worker error: {e}")
    finally:
        # Clean up
        wandb.finish()


def find_completed_batches(experiment_dir: str) -> set[int]:
    """Find all completed batch indices in the experiment directory.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        A set of completed batch indices.
    """
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
    if not os.path.exists(router_logits_dir):
        return set()

    completed_batches = set()
    for filename in os.listdir(router_logits_dir):
        if filename.endswith(".pt"):
            # Extract batch index from filename (format: {batch_idx}.pt)
            batch_idx = int(filename.split(".")[0])
            completed_batches.add(batch_idx)

    return completed_batches


@arguably.command()
def get_router_activations(
    model_name: str = "gpt",
    dataset_name: str = "lmsys",
    *_args,
    batch_size: int = 4,
    cuda_devices: str = "",
    tokens_per_file: int = 20_000,
    resume: bool = False,
    name: str | None = None,
    wandb_project: str = "router_activations",
) -> None:
    """
    Extract router activations from a model using multiple GPUs.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        batch_size: Batch size for processing
        cuda_devices: Comma-separated list of CUDA devices to use. If empty, defaults to CUDA_VISIBLE_DEVICES environment variable or CPU if it's not set.
        tokens_per_file: Target number of tokens per output file
        resume: Whether to resume from a previous run
        name: Custom name for the experiment
        wandb_project: WandB project name for logging
    """
    cuda_devices_raw = cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cpu_only = cuda_devices_raw == ""
    device_ids = [int(i) for i in cuda_devices_raw.split(",")] if not cpu_only else [0]

    world_size = len(device_ids)
    if world_size == 0:
        raise ValueError(f"Unable to parse CUDA devices: {device_ids}")

    if cpu_only:
        print("Using CPU only")
    else:
        print(f"Using {world_size} GPUs: {device_ids}")

    # Create experiment configuration
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "tokens_per_file": tokens_per_file,
        "world_size": world_size,
        "cpu_only": cpu_only,
        "device_ids": device_ids,
    }

    # Generate experiment name if not provided
    if name is None:
        name = get_experiment_name(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
        )

    # Create experiment directories
    experiment_dir = os.path.join(OUTPUT_DIR, name)
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(router_logits_dir, exist_ok=True)

    # Verify configuration against existing one (if any)
    verify_config(config, experiment_dir)

    # Save configuration
    save_config(config, experiment_dir)

    # Initialize WandB
    wandb_run = wandb.init(project=wandb_project, name=name, config=config)
    wandb_run_id = wandb_run.id if wandb_run else None  # type: ignore  # Handle the case where wandb.init returns None

    # Find completed batches if resuming
    resume_batch_idx = 0
    if resume:
        completed_batches = find_completed_batches(experiment_dir)
        if completed_batches:
            max_batch_idx = max(completed_batches) if completed_batches else 0
            resume_batch_idx = max_batch_idx + 1
            print(f"Resuming from batch {resume_batch_idx}")
            wandb.log({"resume/batch_idx": resume_batch_idx})

    # Initialize multiprocessing resources
    mp.set_start_method("spawn", force=True)

    # Create queues and events
    main_queue = mp.Queue(maxsize=100)  # Buffer between tokenizer and multiplexer
    gpu_queues = [mp.Queue(maxsize=10) for _ in range(world_size)]  # One queue per GPU
    stop_event = mp.Event()  # For signaling processes to stop

    # Create shared state for GPU busy status
    manager = mp.Manager()
    gpu_busy = manager.list([False] * world_size)

    # Create and start processes
    processes = []

    # Start tokenizer worker
    tokenizer_proc = mp.Process(
        target=tokenizer_worker,
        args=(
            model_name,
            dataset_name,
            tokens_per_file,
            main_queue,
            stop_event,
            wandb_run_id,
            resume_batch_idx,
        ),
    )
    tokenizer_proc.start()
    processes.append(tokenizer_proc)

    # Start multiplexer worker
    multiplexer_proc = mp.Process(
        target=multiplexer_worker,
        args=(main_queue, gpu_queues, stop_event, wandb_run_id, gpu_busy),
    )
    multiplexer_proc.start()
    processes.append(multiplexer_proc)

    # Start GPU workers
    for rank, device_id in enumerate(device_ids):
        gpu_proc = mp.Process(
            target=gpu_worker,
            args=(
                rank,
                device_id,
                gpu_queues[rank],
                model_name,
                name,
                cpu_only,
                stop_event,
                wandb_run_id,
                gpu_busy,
            ),
        )
        gpu_proc.start()
        processes.append(gpu_proc)

    try:
        # Wait for all processes to finish
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping all processes...")
        stop_event.set()

        # Wait for processes to terminate
        for proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
    finally:
        # Clean up
        wandb.finish()


if __name__ == "__main__":
    arguably.run()
