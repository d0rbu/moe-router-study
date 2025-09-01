from collections import deque
import gc
from itertools import batched
import os
import time
from typing import TypeVar
import warnings

import arguably
from nnterp import StandardizedTransformer
import torch as th
import torch.multiprocessing as mp
from tqdm import tqdm
import trackio as wandb
import yaml

from core.data import DATASETS
from core.device_map import CUSTOM_DEVICES
from core.model import MODELS
from exp import OUTPUT_DIR

# Constants
ROUTER_LOGITS_DIRNAME = "router_logits"
CONFIG_FILENAME = "config.yaml"

# Type definitions
T = TypeVar("T")

# Set wandb availability flag
WANDB_AVAILABLE = True


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


def verify_config(config: dict, experiment_dir: str) -> None:
    """Verify that the current configuration matches the saved one."""
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        saved_config = yaml.safe_load(f)

    # Check for mismatches
    mismatches = {}
    for key, value in config.items():
        if key in saved_config and saved_config[key] != value:
            mismatches[key] = (saved_config[key], value)

    if mismatches:
        mismatch_str = "\n".join(
            f"  - {key}: saved={saved} vs current={current}"
            for key, (saved, current) in mismatches.items()
        )
        raise ValueError(
            f"Configuration mismatch with existing experiment:\n{mismatch_str}"
        )


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

    # Move encoded batch to the appropriate device
    for key in encoded_batch:
        if isinstance(encoded_batch[key], th.Tensor):
            encoded_batch[key] = encoded_batch[key].to(model.device)

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


def tokenizer_worker(
    dataset_name: str,
    model_name: str,
    tokenizer_batch: int,
    tokens_per_file: int,
    main_queue: mp.Queue,
    stop_event: mp.Event,
    wandb_run_id: str,
    resume_from_batch: int = 0,
) -> None:
    """Worker process that tokenizes text and puts batches into the queue."""
    # Initialize wandb in this process
    wandb.init(
        project="router_activations",
        id=wandb_run_id,
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

    # Skip batches if resuming
    if resume_from_batch > 0:
        wandb.log({"tokenizer/skipping_batches": resume_from_batch})
        for _ in tqdm(
            range(resume_from_batch * tokenizer_batch),
            desc="Skipping batches for resume",
        ):
            try:
                next(dataset_iter)
            except StopIteration:
                wandb.log({"tokenizer/error": "Dataset exhausted during resume"})
                stop_event.set()
                return

    # Initialize buffer and statistics
    buffer: deque[tuple[str, list[str]]] = deque()
    total_tokens = 0
    batch_idx = resume_from_batch
    start_time = time.time()

    # Process dataset
    try:
        for batch_data in batched(dataset_iter, tokenizer_batch):
            if stop_event.is_set():
                break

            # Tokenize batch
            tokenized = [tokenizer.tokenize(text) for text in batch_data]

            # Calculate token counts for each sequence
            token_counts = [len(tokens) for tokens in tokenized]

            # Add to buffer
            for text, tokens, count in zip(
                batch_data, tokenized, token_counts, strict=False
            ):
                buffer.append((text, tokens))
                total_tokens += count

                # Check if we have enough tokens to create a batch for processing
                if sum(token_counts) >= tokens_per_file:
                    # Create batch from all items in buffer
                    batch = [buffer.popleft() for _ in range(len(buffer))]
                    batch_texts, batch_tokens = tuple(zip(*batch, strict=False))

                    # Create encoded batch
                    encoded_batch = tokenizer(
                        batch_texts, padding=True, return_tensors="pt"
                    )
                    tokens_sum = sum(token_counts)

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

                    batch_idx += 1
                    token_counts = []

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
    stop_event: mp.Event,
    wandb_run_id: str,
    gpu_busy: list[bool],
) -> None:
    """Worker that distributes batches to GPU queues based on load balancing."""
    # Initialize wandb in this process
    wandb.init(
        project="router_activations",
        id=wandb_run_id,
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
                    **{
                        f"multiplexer/gpu_{i}_busy": int(gpu_busy[i])
                        for i in range(len(gpu_queues))
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
    _world_size: int,  # Renamed to avoid unused argument warning
    gpu_queue: mp.Queue,
    model_name: str,
    experiment_name: str,
    device: str,
    stop_event: mp.Event,
    wandb_run_id: str,
    gpu_busy: list[bool] | None = None,  # Reference to multiplexer's gpu_busy list
) -> None:
    """Worker process that runs on a specific GPU and processes batches."""
    # Set device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device_id = (
        f"cuda:{0}"  # Always use first visible device (which is our assigned GPU)
    )

    # Initialize wandb in this process
    wandb.init(
        project="router_activations",
        id=wandb_run_id,
        resume="allow",
        name=f"gpu-{rank}",
    )

    try:
        # Get model config
        model_config = MODELS.get(model_name)
        if model_config is None:
            raise ValueError(f"Model {model_name} not found")

        # Initialize model
        device_map = CUSTOM_DEVICES.get(device, lambda: device_id)()
        model = StandardizedTransformer(
            model_config.hf_name,
            check_attn_probs_with_trace=False,
            device_map=device_map,
        )
        router_layers = model.layers_with_routers
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
            if gpu_busy is not None:
                gpu_busy[rank] = False

            # Get batch from queue
            item = gpu_queue.get()

            # Signal that we're busy processing
            if gpu_busy is not None:
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
            batch_start = time.time()
            with th.inference_mode():
                # Extract router logits
                router_logits = []
                with model.trace(encoded_batch):
                    for layer in router_layers:
                        padding_mask = encoded_batch.attention_mask.bool().view(-1)
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

                # Stack the logits
                router_logits_tensor = th.stack(router_logits, dim=1).clone().detach()

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

            # Calculate EMA of processing time (with 0.9 decay)
            ema_time = (
                processing_times[0]
                if len(processing_times) == 1
                else (0.9 * processing_times[-2] + 0.1 * processing_times[-1])
            )

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
                    f"gpu_{rank}/ema_batch_time": ema_time,
                    f"gpu_{rank}/elapsed_time": elapsed,
                }
            )

    except Exception as e:
        wandb.log({f"gpu_{rank}/error": str(e)})
        print(f"GPU {rank} worker error: {e}")
    finally:
        # Clean up
        wandb.finish()


def find_completed_batches(experiment_dir: str) -> tuple[dict[int, list[int]], int]:
    """
    Find all completed batches across all GPUs.

    Returns:
        Tuple containing:
        - Dictionary mapping GPU rank to list of completed batch indices
        - Highest batch index found
    """
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
    if not os.path.exists(router_logits_dir):
        return {}, 0

    completed_batches: dict[int, list[int]] = {}
    max_batch_idx = 0

    for filename in os.listdir(router_logits_dir):
        if not filename.endswith(".pt"):
            continue

        # Parse filename to get batch index
        parts = filename.split(".")[0].split("_")
        if len(parts) == 2:
            rank, batch_idx = int(parts[0]), int(parts[1])

            if rank not in completed_batches:
                completed_batches[rank] = []

            completed_batches[rank].append(batch_idx)
            max_batch_idx = max(max_batch_idx, batch_idx)
        elif len(parts) == 1:
            # Handle old format files (no rank)
            batch_idx = int(parts[0])
            if -1 not in completed_batches:
                completed_batches[-1] = []
            completed_batches[-1].append(batch_idx)
            max_batch_idx = max(max_batch_idx, batch_idx)

    return completed_batches, max_batch_idx


@arguably.command()
def get_router_activations(
    model_name: str = "gpt",
    dataset_name: str = "lmsys",
    *_args,
    batch_size: int = 4,
    device: str = "cpu",
    tokens_per_file: int = 2_000,
    tokenizer_batch: int = 16,
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
        device: Device to use (cpu, cuda, mlp_gpu, attn_gpu)
        tokens_per_file: Target number of tokens per output file
        tokenizer_batch: Batch size for tokenizer worker
        resume: Whether to resume from a previous run
        name: Custom name for the experiment
        wandb_project: WandB project name for logging
    """
    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_devices:
        raise ValueError("CUDA_VISIBLE_DEVICES environment variable not set")

    world_size = len(cuda_devices.split(","))
    if world_size == 0:
        raise ValueError("No CUDA devices available")

    print(f"Using {world_size} GPUs: {cuda_devices}")

    # Create experiment configuration
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "tokens_per_file": tokens_per_file,
        "tokenizer_batch": tokenizer_batch,
        "world_size": world_size,
    }

    # Generate experiment name if not provided
    if name is None:
        name = get_experiment_name(
            model_name=model_name,
            dataset_name=dataset_name,
            batch_size=batch_size,
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
    wandb_run_id = wandb_run.id

    # Find completed batches if resuming
    resume_batch_idx = 0
    if resume:
        completed_batches, max_batch_idx = find_completed_batches(experiment_dir)
        if completed_batches:
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
            dataset_name,
            model_name,
            tokenizer_batch,
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
    for rank in range(world_size):
        gpu_proc = mp.Process(
            target=gpu_worker,
            args=(
                rank,
                world_size,
                gpu_queues[rank],
                model_name,
                name,
                device,
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
