from collections import deque
from multiprocessing.synchronize import Event
import os
import time
from typing import Any, TypeVar
import warnings

import arguably
from nnterp import StandardizedTransformer
import torch as th
import torch.multiprocessing as mp
from tqdm import tqdm

# Import trackio with the same interface as wandb
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
            f"The following keys were filtered from the experiment name: {filtered_keys}",
            stacklevel=2,
        )

    # Construct the full name
    if param_items:
        return f"{base_name}_{'_'.join(param_items)}"
    return base_name


def save_config(config: dict, experiment_dir: str) -> None:
    """Save experiment configuration to a YAML file."""
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def verify_config(config: dict, experiment_dir: str) -> None:
    """Verify that the configuration matches any existing saved configuration."""
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        saved_config = yaml.safe_load(f)

    # Check if configs match
    if saved_config != config:
        warnings.warn(
            f"Configuration mismatch with existing experiment at {experiment_dir}. "
            f"Existing: {saved_config}, New: {config}",
            stacklevel=2,
        )


def save_router_logits(
    router_logit_collection: list[th.Tensor],
    tokenized_batch_collection: list[list[str]],
    top_k: int,
    router_logit_collection_idx: int,
    experiment_name: str,
) -> dict[str, Any]:
    """Save router logits to disk."""
    # Create directory if it doesn't exist
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
    os.makedirs(router_logits_dir, exist_ok=True)

    # Stack router logits
    router_logits_tensor = th.cat(router_logit_collection, dim=0)

    # Save router logits
    output_path = os.path.join(router_logits_dir, f"{router_logit_collection_idx}.pt")
    output = {
        "topk": top_k,
        "router_logits": router_logits_tensor,
        "tokens": tokenized_batch_collection,
    }
    th.save(output, output_path)
    return output


def process_batch(
    batch: list[str], model: StandardizedTransformer, router_layers: list[int]
) -> tuple[th.Tensor, list[list[str]]]:
    """Process a batch of text to get router logits."""
    # Tokenize batch
    tokenized_batch = [model.tokenizer.encode(text) for text in batch]

    # Create batch tensor
    encoded_batch = model.tokenizer.batch_encode_plus(
        batch, padding=True, return_tensors="pt"
    )

    # Move tensors to device
    for key in encoded_batch:
        if isinstance(encoded_batch[key], th.Tensor):
            encoded_batch[key] = encoded_batch[key].to(model.device)

    # Get router logits
    with th.inference_mode(), model.trace(encoded_batch) as tracer:
        # Get attention mask to filter out padding tokens
        attention_mask = encoded_batch["attention_mask"]
        padding_mask = attention_mask.bool().flatten()

        # Extract router logits for each layer
        router_logits = []
        for layer in router_layers:
            # Get router logits and immediately detach and move to CPU
            router_output = model.routers_output[layer]

            # Handle different router output formats
            if isinstance(router_output, tuple):
                router_scores, _router_indices = router_output
                logits = router_scores.cpu()[padding_mask].save()
            else:
                router_scores = router_output
                logits = router_scores.cpu()[padding_mask].save()

            router_logits.append(logits.clone().detach())

        # Stack the logits and create a new tensor to break references
        router_logits_tensor = th.stack(router_logits, dim=1).clone().detach()

        # Explicitly stop the tracer to clean up resources
        tracer.stop()

    return router_logits_tensor, tokenized_batch


def tokenizer_worker(
    dataset_name: str,
    model_name: str,
    tokenizer_batch: int,
    tokens_per_file: int,
    main_queue: mp.Queue,
    stop_event: Event,
    wandb_run_id: str,  # noqa: ARG001
    resume_from_batch: int = 0,
) -> None:
    """Worker process that tokenizes text and puts batches into the queue."""
    # Initialize wandb in this process
    wandb.init(
        project="router_activations",
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
    dataset_iter = iter(dataset_fn(tokenizer))

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
        for text in dataset_iter:
            if stop_event.is_set():
                break

            # Tokenize text
            tokens = tokenizer.tokenize(text)
            count = len(tokens)

            # Add to buffer
            buffer.append((text, tokens))
            total_tokens += count

            # Check if we have enough tokens to create a batch for processing
            if sum(len(tokens) for text, tokens in buffer) >= tokens_per_file:
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
                        "tokenizer/total_tokens": total_tokens + tokens_sum,
                        "tokenizer/sequences_in_batch": len(batch_texts),
                        "tokenizer/buffer_size": len(buffer),
                        "tokenizer/tokens_per_second": (total_tokens + tokens_sum)
                        / elapsed
                        if elapsed > 0
                        else 0,
                    }
                )

                # Update statistics
                total_tokens += tokens_sum
                batch_idx += 1

            # Check if queue is getting too full - pause to let consumers catch up

    except Exception as e:
        wandb.log({"tokenizer/error": str(e)})
        print(f"Tokenizer worker error: {e}")
    finally:
        # If there are remaining items in the buffer, send them as a final batch
        if buffer and not stop_event.is_set():
            batch = [buffer.popleft() for _ in range(len(buffer))]
            batch_texts, batch_tokens = tuple(zip(*batch, strict=False))
            tokens_sum = sum(len(tokens) for tokens in batch_tokens)

            main_queue.put(
                (batch_idx, batch_texts, batch_tokens, tokens_sum), block=True
            )

            # Log statistics
            elapsed = time.time() - start_time
            wandb.log(
                {
                    "tokenizer/batch_idx": batch_idx,
                    "tokenizer/queue_size": main_queue.qsize(),
                    "tokenizer/tokens_in_batch": tokens_sum,
                    "tokenizer/total_tokens": total_tokens + tokens_sum,
                    "tokenizer/sequences_in_batch": len(batch_texts),
                    "tokenizer/buffer_size": len(buffer),
                    "tokenizer/tokens_per_second": (total_tokens + tokens_sum) / elapsed
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
    stop_event: Event,
    wandb_run_id: str,  # noqa: ARG001
    gpu_busy: list[bool],
) -> None:
    """Worker that distributes batches to GPU queues based on load balancing."""
    # Initialize wandb in this process
    wandb.init(
        project="router_activations",
        resume="allow",
        name="multiplexer",
    )

    # Initialize statistics
    total_batches = 0
    total_tokens = 0
    start_time = time.time()

    try:
        while not stop_event.is_set():
            # Get batch from main queue
            item = main_queue.get()
            if item is None:
                break

            batch_idx, batch_texts, batch_tokens, tokens_count = item
            total_batches += 1
            total_tokens += tokens_count

            # Find GPU with smallest queue
            queue_sizes = []
            for i, q in enumerate(gpu_queues):
                size = q.qsize()
                # Add penalty if GPU is busy
                size += gpu_busy[i]
                queue_sizes.append(size)

            # Get the index of the GPU with the smallest queue
            target_gpu = th.argmin(th.tensor(queue_sizes)).item()

            # Send to that GPU
            gpu_queues[target_gpu].put(
                (batch_idx, batch_texts, batch_tokens, tokens_count)
            )

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
    stop_event: Event,
    wandb_run_id: str,  # noqa: ARG001
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
        resume="allow",
        name=f"gpu-{rank}",
    )

    # Get model config
    model_config = MODELS[model_name]
    hf_name = model_config.hf_name

    # Set up device map
    device_map = CUSTOM_DEVICES.get(device, lambda: device_id)()

    # Load model
    model = StandardizedTransformer(
        hf_name, check_attn_probs_with_trace=False, device_map=device_map
    )

    # Get router layers
    router_layers = model.layers_with_routers
    top_k = model.router_probabilities.get_top_k()

    # Create output directory
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
    os.makedirs(router_logits_dir, exist_ok=True)

    # Initialize statistics
    total_batches = 0
    total_tokens = 0
    start_time = time.time()
    processing_times = []

    try:
        while not stop_event.is_set():
            # Get batch from queue
            item = gpu_queue.get()
            if item is None:
                break

            batch_idx, batch_texts, batch_tokens, tokens_count = item

            # Process batch
            batch_start = time.time()
            with th.inference_mode():
                router_logits, _ = process_batch(batch_texts, model, router_layers)

            # Save results
            output_path = os.path.join(router_logits_dir, f"{batch_idx}.pt")
            output = {
                "topk": top_k,
                "router_logits": router_logits,
                "tokens": batch_tokens,
            }
            th.save(output, output_path)

            # Update statistics
            batch_time = time.time() - batch_start
            processing_times.append(batch_time)
            total_batches += 1
            total_tokens += tokens_count
            elapsed = time.time() - start_time

            # Calculate EMA of batch time
            ema_batch_time = (
                processing_times[-1]
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
                    f"gpu_{rank}/ema_batch_time": ema_batch_time,
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
    Find all completed batches in the experiment directory.
    Returns:
        A tuple of (completed_batches, max_batch_idx) where:
        - completed_batches is a dict mapping rank -> list of batch indices
        - max_batch_idx is the maximum batch index found
    """
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
    if not os.path.exists(router_logits_dir):
        return {}, -1

    completed_batches: dict[int, list[int]] = {}
    max_batch_idx = -1

    for filename in os.listdir(router_logits_dir):
        if not filename.endswith(".pt"):
            continue

        # Files are now named as batch_idx.pt
        try:
            batch_idx = int(filename.split(".")[0])
            rank = 0  # Default rank for global batch index format

            if rank not in completed_batches:
                completed_batches[rank] = []

            completed_batches[rank].append(batch_idx)
            max_batch_idx = max(max_batch_idx, batch_idx)

        except ValueError:
            # Skip files that don't match the expected format
            continue

    return completed_batches, max_batch_idx


@arguably.command()
def get_router_activations(
    model_name: str,
    dataset_name: str,
    batch_size: int = 8,
    device: str = "cuda:0",
    tokens_per_file: int = 10000,
    resume: bool = False,
    name: str | None = None,
    tokenizer_batch: int = 16,
    wandb_project: str = "router_activations",
) -> None:
    """
    Get router activations for a model on a dataset.
    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        batch_size: Batch size for processing
        device: Device to use for processing
        tokens_per_file: Number of tokens to include in each output file
        resume: Whether to resume from a previous run
        name: Name of the experiment (default: auto-generated)
        tokenizer_batch: Number of sequences to tokenize at once
        wandb_project: WandB project name
    """
    # Check that CUDA_VISIBLE_DEVICES is set
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_devices:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES environment variable must be set to specify which GPUs to use"
        )

    # Determine world size from CUDA_VISIBLE_DEVICES
    world_size = len(cuda_devices.split(","))
    if world_size == 0:
        raise ValueError("No GPUs available")

    print(f"Using {world_size} GPUs: {cuda_devices}")

    # Create config
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "device": device,
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
            tokenizer_batch=tokenizer_batch,
        )

    # Create experiment directory
    experiment_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Verify configuration
    verify_config(config, experiment_dir)

    # Save configuration
    save_config(config, experiment_dir)

    # Initialize WandB
    wandb.init(project=wandb_project, name=name, config=config)
    wandb_run_id = f"{name}_{int(time.time())}"  # Simple unique ID

    # Find completed batches if resuming
    resume_batch_idx = 0
    if resume:
        completed_batches, max_batch_idx = find_completed_batches(experiment_dir)
        if completed_batches:
            resume_batch_idx = max_batch_idx + 1
            print(f"Resuming from batch {resume_batch_idx}")
            wandb.log({"resume/batch_idx": resume_batch_idx})

    # Create multiprocessing primitives
    stop_event = mp.Event()
    main_queue = mp.Queue(maxsize=100)  # Buffer between tokenizer and multiplexer
    gpu_queues = [mp.Queue(maxsize=10) for _ in range(world_size)]  # One queue per GPU
    gpu_busy = [False] * world_size  # Track if GPU is currently processing

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

    # Start multiplexer worker
    multiplexer_proc = mp.Process(
        target=multiplexer_worker,
        args=(main_queue, gpu_queues, stop_event, wandb_run_id, gpu_busy),
    )
    multiplexer_proc.start()

    # Start GPU workers
    gpu_procs = []
    for rank in range(world_size):
        proc = mp.Process(
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
            ),
        )
        proc.start()
        gpu_procs.append(proc)

    try:
        # Wait for all processes to finish
        tokenizer_proc.join()
        multiplexer_proc.join()
        for proc in gpu_procs:
            proc.join()
    except KeyboardInterrupt:
        print("Interrupted, cleaning up...")
        stop_event.set()
        # Give processes a chance to clean up
        tokenizer_proc.join(timeout=10)
        if tokenizer_proc.is_alive():
            tokenizer_proc.terminate()
        multiplexer_proc.join(timeout=10)
        if multiplexer_proc.is_alive():
            multiplexer_proc.terminate()
        for proc in gpu_procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
    finally:
        # Clean up
        wandb.finish()


if __name__ == "__main__":
    arguably.run()
