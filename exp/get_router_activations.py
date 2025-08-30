import gc
import os
import threading
from typing import Any
import warnings

import arguably
from nnterp import StandardizedTransformer
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import yaml

try:
    import trackio as wandb
except ImportError:
    wandb = None

from core.data import DATASETS
from core.model import MODELS
from exp import OUTPUT_DIR

# Constants
ROUTER_LOGITS_DIRNAME = "router_logits"
CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.yaml"


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


def save_config(config: dict[str, Any], experiment_dir: str) -> None:
    """Save experiment configuration to a YAML file."""
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def verify_config(config: dict[str, Any], experiment_dir: str) -> None:
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


def save_router_logits(
    router_logit_collection: list[th.Tensor],
    tokenized_batch_collection: list[list[str]],
    top_k: int,
    file_idx: int,
    experiment_name: str,
) -> None:
    router_logits = th.cat(router_logit_collection, dim=0)
    output: dict[str, th.Tensor] = {
        "topk": top_k,
        "router_logits": router_logits,
        "tokens": tokenized_batch_collection,
    }
    router_logits_dir = os.path.join(OUTPUT_DIR, experiment_name, ROUTER_LOGITS_DIRNAME)
    output_path = os.path.join(router_logits_dir, f"{file_idx}.pt")
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
    dataset_name: str = "lmsys",
    *_args,
    tokenizer_batchsize: int = 4,
    tokens_per_file: int = 20_000,
    device: str = "auto",
    resume: bool = False,
    use_wandb: bool = False,
    gpu_minibatch: int = 4,
    use_fallback_comms_backend: bool = False,
    verbose: bool = False,
    name: str | None = None,
) -> None:
    """
    Extract router activations from a model on a dataset.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        tokenizer_batchsize: Number of sequences to tokenize at once
        tokens_per_file: Target number of tokens per batch file
        device: Device to use for model inference
        resume: Whether to resume from a previous run
        use_wandb: Whether to use wandb for logging
        gpu_minibatch: Size of mini-batches to process on GPU
        use_fallback_comms_backend: Whether to allow fallback to other backends if NCCL is not available
        verbose: Whether to print verbose output
        name: Optional name for the experiment
    """
    model_config = MODELS.get(model_name, None)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    dataset_fn = DATASETS.get(dataset_name, None)
    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Determine world size from CUDA_VISIBLE_DEVICES or available devices
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        # Parse CUDA_VISIBLE_DEVICES to get available GPU indices
        visible_devices = [
            int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()
        ]
        world_size = len(visible_devices)
    else:
        # Use all available GPUs
        world_size = th.cuda.device_count()

    # Check if we have any GPUs
    if world_size == 0:
        raise ValueError("No CUDA devices available")

    # Create experiment configuration
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "tokenizer_batchsize": tokenizer_batchsize,
        "tokens_per_file": tokens_per_file,
        "world_size": world_size,
        "gpu_minibatch": gpu_minibatch,
    }

    # Generate experiment name if not provided
    if name is None:
        name = get_experiment_name(
            model_name=model_name,
            dataset_name=dataset_name,
            tokenizer_batchsize=tokenizer_batchsize,
            tokens_per_file=tokens_per_file,
            device=device,
            verbose=verbose,
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

    # Launch distributed processes
    if world_size > 1:
        mp.spawn(
            init_distributed_process,
            args=(
                world_size,
                model_name,
                dataset_name,
                name,
                tokenizer_batchsize,
                tokens_per_file,
                resume,
                use_wandb,
                gpu_minibatch,
                use_fallback_comms_backend,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU mode
        init_distributed_process(
            0,
            world_size,
            model_name,
            dataset_name,
            name,
            tokenizer_batchsize,
            tokens_per_file,
            resume,
            use_wandb,
            gpu_minibatch,
            use_fallback_comms_backend,
        )


def init_distributed_process(
    rank: int,
    world_size: int,
    model_name: str,
    dataset_name: str,
    experiment_name: str,
    tokenizer_batchsize: int,
    tokens_per_file: int,
    resume: bool,
    use_wandb: bool,
    gpu_minibatch: int = 4,
    use_fallback_comms_backend: bool = False,
):
    """
    Initialize a distributed process for router activation extraction.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        experiment_name: Name of the experiment
        tokenizer_batchsize: Number of sequences to tokenize at once
        tokens_per_file: Target number of tokens per batch file
        resume: Whether to resume from a previous run
        use_wandb: Whether to use wandb for logging
        gpu_minibatch: Size of mini-batches to process on GPU
        use_fallback_comms_backend: Whether to allow fallback to other backends if NCCL is not available
    """
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Try to use NCCL backend first (best for GPU communication)
    backend = "nccl"
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    except Exception as e:
        if not use_fallback_comms_backend:
            error_msg = (
                f"Failed to initialize NCCL backend: {e}. "
                "Set use_fallback_comms_backend=True to allow fallback to other backends."
            )
            raise ValueError(error_msg) from e

        # Try MPI backend next
        backend = "mpi"
        try:
            dist.init_process_group(backend, rank=rank, world_size=world_size)
        except Exception:
            # Fall back to gloo as last resort
            backend = "gloo"
            dist.init_process_group(backend, rank=rank, world_size=world_size)

        print(
            f"Warning: NCCL backend not available, falling back to {backend} backend."
        )

    # Get model and dataset
    model_config = MODELS.get(model_name, None)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    dataset_fn = DATASETS.get(dataset_name, None)
    if dataset_fn is None:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Create experiment configuration
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "tokenizer_batchsize": tokenizer_batchsize,
        "tokens_per_file": tokens_per_file,
        "world_size": world_size,
        "gpu_minibatch": gpu_minibatch,
    }

    # Create experiment directories
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(router_logits_dir, exist_ok=True)

    # Metadata for tracking processed batches
    metadata_path = os.path.join(experiment_dir, METADATA_FILENAME)
    processed_batches = set()

    if resume and os.path.exists(metadata_path):
        with open(metadata_path) as f:
            loaded_data = yaml.safe_load(f)
            if loaded_data is not None:
                if isinstance(loaded_data, list):
                    # New format: list of batch indices
                    processed_batches = set(loaded_data)
                elif isinstance(loaded_data, dict):
                    # Legacy format: dict of batch indices to bool
                    processed_batches.update(loaded_data.keys())

    # Only rank 0 verifies and saves config
    if rank == 0:
        # Verify configuration against existing one (if any)
        verify_config(config, experiment_dir)

        # Save configuration
        save_config(config, experiment_dir)

    # Wait for rank 0 to save config
    dist.barrier()

    # Initialize wandb
    _wandb_run = None
    if use_wandb and rank == 0 and wandb is not None:
        _wandb_run = wandb.init(
            project="router-activations",
            name=experiment_name,
            config=config,
        )

    # Launch distributed processes
    if world_size > 1:
        mp.spawn(
            init_distributed_process,
            args=(
                world_size,
                model_name,
                dataset_name,
                experiment_name,
                tokenizer_batchsize,
                tokens_per_file,
                resume,
                use_wandb,
                gpu_minibatch,
                use_fallback_comms_backend,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU mode
        init_distributed_process(
            0,
            world_size,
            model_name,
            dataset_name,
            experiment_name,
            tokenizer_batchsize,
            tokens_per_file,
            resume,
            use_wandb,
            gpu_minibatch,
            use_fallback_comms_backend,
        )


def gpu_worker(
    rank: int,
    _world_size: int,
    gpu_queue: mp.Queue,
    model_config: dict,
    experiment_name: str,
    _processed_batches: set[int],
    _metadata_lock: threading.Lock,
    _metadata_path: str,
    stop_event: mp.Event,
    tokens_per_file: int,
    _gpu_minibatch: int = 4,
    _wandb_run: Any | None = None,
):
    """
    Worker that processes batches on a GPU and saves results.

    Args:
        rank: GPU rank
        _world_size: Total number of GPUs (unused)
        gpu_queue: Queue to get batches from
        model_config: Model configuration
        experiment_name: Name of the experiment
        _processed_batches: Set tracking which batches have been processed (unused)
        _metadata_lock: Lock for accessing the processed_batches set (unused)
        _metadata_path: Path to save metadata (unused)
        stop_event: Event to signal when to stop
        tokens_per_file: Target number of tokens per batch file
        _gpu_minibatch: Size of mini-batches to process on GPU (unused)
        _wandb_run: Optional wandb run for logging (unused)
    """
    try:
        # Set device based on CUDA_VISIBLE_DEVICES if set, otherwise use rank directly
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            # Parse CUDA_VISIBLE_DEVICES to get available GPU indices
            visible_devices = [
                int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()
            ]
            if rank < len(visible_devices):
                # Use the device ID from CUDA_VISIBLE_DEVICES
                device = f"cuda:{rank}"  # We can use rank directly because PyTorch maps to visible devices
            else:
                raise ValueError(
                    f"Rank {rank} is out of range for visible devices {visible_devices}"
                )
        else:
            # No CUDA_VISIBLE_DEVICES set, use rank directly
            device = f"cuda:{rank}"

        th.cuda.set_device(device)

        # Initialize model
        model = StandardizedTransformer(
            model_config.hf_name, check_attn_probs_with_trace=False, device_map=device
        )
        model.eval()

        # Initialize router layers
        router_layers = model.layers_with_routers

        # Initialize top K
        top_k = model.router_probabilities.get_top_k()

        # Initialize other variables
        router_logit_collection = []
        tokenized_batch_collection = []
        router_logit_collection_size = 0
        router_logit_collection_idx = 0

        # Initialize progress bar
        pbar = tqdm(total=tokens_per_file, desc="Filling up file")

        while not stop_event.is_set():
            # Get batch from queue
            batch = gpu_queue.get()

            # Process batch
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
                    experiment_name,
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

        # Final cleanup
        del router_logit_collection
        del tokenized_batch_collection
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    except Exception as e:
        # Log error
        print(f"GPU worker {rank} failed: {e}")
        raise


if __name__ == "__main__":
    arguably.run()
