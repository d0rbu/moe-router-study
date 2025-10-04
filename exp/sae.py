import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import batched, count, product
import math
import os
import sys
from typing import Any

import arguably
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.trainer import SAETrainer
from dictionary_learning.training import trainSAE
from loguru import logger
from sae_bench.custom_saes.base_sae import BaseSAE
from sae_bench.custom_saes.batch_topk_sae import (
    load_dictionary_learning_batch_topk_sae,
    load_dictionary_learning_matryoshka_batch_topk_sae,
)
import torch as th
import torch.distributed as dist
from tqdm import tqdm

from core.async_utils import handle_exceptions
from core.dtype import get_dtype
from core.training import exponential_to_linear_save_steps
from core.type import assert_type
from exp import OUTPUT_DIR
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from exp.training import get_experiment_name


@dataclass
class Architecture:
    saebench_load_fn: Callable[
        [str, str, str, th.device, th.dtype, int | None, str], BaseSAE
    ]
    trainer: type[SAETrainer]
    sae: type[Dictionary]
    constructor_keys: set[str] = field(default_factory=set)

    def filter_constructor_args(
        self, constructor_args: dict[str, Any]
    ) -> dict[str, Any]:
        return {k: v for k, v in constructor_args.items() if k in self.constructor_keys}


BATCHTOPK_CONSTRUCTOR_KEYS = {
    "trainer",
    "steps",
    "activation_dim",
    "dict_size",
    "k",
    "layer",
    "lm_name",
    "dict_class",
    "lr",
    "auxk_alpha",
    "warmup_steps",
    "decay_start",
    "threshold_beta",
    "threshold_start_step",
    "k_anneal_steps",
    "seed",
    "device",
    "wandb_name",
    "submodule_name",
}


ARCHITECTURES = {
    "batchtopk": Architecture(
        saebench_load_fn=load_dictionary_learning_batch_topk_sae,
        trainer=BatchTopKTrainer,
        sae=BatchTopKSAE,
        constructor_keys=BATCHTOPK_CONSTRUCTOR_KEYS,
    ),
    "matryoshka": Architecture(
        saebench_load_fn=load_dictionary_learning_matryoshka_batch_topk_sae,
        trainer=MatryoshkaBatchTopKTrainer,
        sae=MatryoshkaBatchTopKSAE,
        constructor_keys=BATCHTOPK_CONSTRUCTOR_KEYS
        | {"group_fractions", "group_weights"},
    ),
}


@dataclass
class GPUBatch:
    trainer_cfgs: list[dict]
    trainer_names: list[str]
    sae_experiment_name: str
    submodule_name: str


def select_least_loaded_gpu(
    gpu_queues: list[asyncio.Queue],
) -> tuple[int, asyncio.Queue]:
    """Select the GPU with the least loaded queue.

    Returns:
        Tuple of (device_idx, gpu_queue) for the least loaded GPU.
    """
    queue_sizes = [gpu_queue.qsize() for gpu_queue in gpu_queues]
    device_idx = th.argmin(th.tensor(queue_sizes)).item()

    gpu_queue = gpu_queues[device_idx]

    return device_idx, gpu_queue


async def gpu_worker(
    device_idx: int,
    dtype: th.dtype,
    steps: int,
    save_steps: list[int],
    num_epochs: int,
    num_gpus: int,
    gpu_queue: asyncio.Queue,
    data_iterator: Callable[[str], Iterator[tuple[th.Tensor, list[int]]]],
) -> None:
    """Worker process for training SAE models on a specific GPU."""

    logger.info(f"[worker {device_idx}]: Starting GPU worker")
    device = f"cuda:{device_idx}"

    for worker_batch_idx in count():
        logger.debug(f"[worker {device_idx}]: Awaiting batch {worker_batch_idx}")
        _priority, _trainer_batch_idx, batch = await gpu_queue.get()

        if batch is None:
            logger.debug(f"[worker {device_idx}]: Stopping")
            gpu_queue.task_done()
            break

        batch = assert_type(batch, GPUBatch)

        logger.debug(f"[worker {device_idx}]: Got batch {worker_batch_idx}")
        trainSAE(
            data=data_iterator(batch.submodule_name),
            trainer_configs=batch.trainer_cfgs,
            steps=steps * num_epochs,
            save_steps=save_steps,
            trainer_names=batch.trainer_names,
            use_wandb=False,
            save_dir=os.path.join(OUTPUT_DIR, batch.sae_experiment_name),
            normalize_activations=True,
            device=device,
            autocast_dtype=dtype,
            tqdm_kwargs={"position": dist.get_rank() * (num_gpus + 1) + device_idx + 1},
        )
        gpu_queue.task_done()


async def run_sae_training(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    expansion_factor: tuple[int, ...],
    k: tuple[int, ...],
    layer: tuple[int, ...],
    group_fractions: tuple[tuple[float, ...], ...],
    group_weights: tuple[tuple[float, ...] | None, ...],
    architecture: tuple[str, ...],
    lr: tuple[float, ...],
    auxk_alpha: tuple[float, ...],
    warmup_steps: tuple[int, ...],
    decay_start: tuple[int | None, ...],
    threshold_beta: tuple[float, ...],
    threshold_start_step: tuple[int, ...],
    k_anneal_steps: tuple[int | None, ...],
    seed: tuple[int, ...],
    submodule_name: tuple[str, ...],
    batch_size: int = 4096,
    trainers_per_gpu: int = 8,
    steps: int = 1024 * 256,
    save_every: int = 1024,
    num_epochs: int = 1,
    context_length: int = 2048,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 20_000,
    num_workers: int = 64,
    debug: bool = False,
    dtype: th.dtype = th.bfloat16,
) -> None:
    """Train autoencoders to sweep over the given hyperparameter sets."""
    assert "moe" not in architecture, (
        "MoE is not supported for SAE training, use kmeans.py instead."
    )
    assert th.cuda.is_available(), "CUDA is not available"

    logger.debug("loading activations and initializing distributed setup")
    logger.trace(
        f"model_name={model_name}\n"
        f"dataset_name={dataset_name}\n"
        f"expansion_factor={expansion_factor}\n"
        f"k={k}\n"
        f"layer={layer}\n"
        f"group_fractions={group_fractions}\n"
        f"group_weights={group_weights}\n"
        f"architecture={architecture}\n"
        f"lr={lr}\n"
        f"auxk_alpha={auxk_alpha}\n"
        f"warmup_steps={warmup_steps}\n"
        f"decay_start={decay_start}\n"
        f"threshold_beta={threshold_beta}\n"
        f"threshold_start_step={threshold_start_step}\n"
        f"k_anneal_steps={k_anneal_steps}\n"
        f"seed={seed}\n"
        f"submodule_name={submodule_name}\n"
        f"batch_size={batch_size}\n"
        f"trainers_per_gpu={trainers_per_gpu}\n"
        f"steps={steps}\n"
        f"save_every={save_every}\n"
        f"num_epochs={num_epochs}\n"
        f"context_length={context_length}\n"
        f"tokens_per_file={tokens_per_file}\n"
        f"reshuffled_tokens_per_file={reshuffled_tokens_per_file}\n"
        f"num_workers={num_workers}\n"
        f"debug={debug}\n"
        f"dtype={dtype}\n"
    )

    (
        activations,
        activation_dims,
        _gpu_process_group,
    ) = await load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        submodule_names=list(submodule_name),
        context_length=context_length,
        num_workers=num_workers,
        debug=debug,
    )

    sae_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        steps=steps,
        num_epochs=num_epochs,
    )

    if len(submodule_name) == 1:
        activation_dim = activation_dims[submodule_name[0]]
    else:
        # find submodule name that is not router_logits
        non_router_logits_submodules = [
            name for name in submodule_name if name != ActivationKeys.ROUTER_LOGITS
        ]
        activation_dim = activation_dims[non_router_logits_submodules[0]]

    assert activation_dim > 0, "Activation dimension must be greater than 0"
    assert num_epochs > 0, "Number of epochs must be greater than 0"

    def data_iterator(submodule_name: str) -> Iterator[tuple[th.Tensor, list[int]]]:
        for epoch_idx in range(num_epochs):
            logger.debug(f"Starting epoch {epoch_idx}")
            for activation_data in activations(batch_size=batch_size):
                assert submodule_name in activation_data, (
                    f"Submodule name {submodule_name} not found in activation keys {activation_data.keys()}"
                )

                logger.trace(f"Activation data: {activation_data.keys()}")

                activation = activation_data[submodule_name]
                layers = activation_data["layers"]

                logger.trace(
                    f"Yielding activation {submodule_name}: {activation.shape} {activation.dtype}"
                )

                yield activation, layers

    save_steps = exponential_to_linear_save_steps(
        total_steps=steps, save_every=save_every
    )

    base_trainer_cfg = {
        "steps": steps,
        "activation_dim": activation_dim,
        "lm_name": model_name,
        "wandb_name": model_name,
    }

    num_gpus = th.cuda.device_count()
    logger.info(f"Number of GPUs: {num_gpus}")
    gpu_queues = [asyncio.PriorityQueue() for _ in range(num_gpus)]

    workers = [
        asyncio.create_task(
            gpu_worker(
                device_idx,
                dtype,
                steps,
                list(save_steps),
                num_epochs,
                num_gpus,
                gpu_queue,
                data_iterator,
            ),
            name=str(device_idx),
        )
        for device_idx, gpu_queue in enumerate(gpu_queues)
    ]

    for worker in workers:
        worker.add_done_callback(handle_exceptions)

    hparam_sweep_iterator = list(
        enumerate(
            product(
                expansion_factor,
                k,
                layer,
                group_fractions,
                group_weights,
                architecture,
                lr,
                auxk_alpha,
                warmup_steps,
                decay_start,
                threshold_beta,
                threshold_start_step,
                k_anneal_steps,
                seed,
                submodule_name,
            )
        )
    )
    distributed_iterator = hparam_sweep_iterator[
        dist.get_rank() :: dist.get_world_size()
    ]

    # assign a subset of the hparam sweep to each rank
    tqdm_distributed_iterator = tqdm(
        distributed_iterator,
        desc=f"Rank {dist.get_rank()}",
        total=len(distributed_iterator),
        leave=True,
        position=dist.get_rank() * (num_gpus + 1),
    )
    # batch it based on how many trainers will be on each gpu
    concurrent_trainer_batched_iterator = batched(
        tqdm_distributed_iterator, trainers_per_gpu
    )

    logger.info(f"Total size of sweep: {len(hparam_sweep_iterator)}")
    logger.info(f"Number of nodes: {dist.get_world_size()}")
    logger.info(f"Number of GPUs per node: {num_gpus}")
    logger.info(f"Number of trainers per GPU: {trainers_per_gpu}")
    logger.info(
        f"Number of iterations: {math.ceil(len(hparam_sweep_iterator) / (trainers_per_gpu * num_gpus * dist.get_world_size()))}"
    )

    for trainer_batch_idx, trainer_batch in enumerate(
        concurrent_trainer_batched_iterator
    ):
        device_idx, gpu_queue = select_least_loaded_gpu(gpu_queues)

        trainer_cfgs = []
        trainer_names = []

        for hparam_idx, (
            current_expansion_factor,
            current_k,
            current_layer,
            current_group_fractions,
            current_group_weights,
            current_architecture,
            current_lr,
            current_auxk_alpha,
            current_warmup_steps,
            current_decay_start,
            current_threshold_beta,
            current_threshold_start_step,
            current_k_anneal_steps,
            current_seed,
            current_submodule_name,
        ) in trainer_batch:
            # check if results file already exists
            trainer_results_dir = os.path.join(
                OUTPUT_DIR, sae_experiment_name, str(hparam_idx)
            )
            config_filepath = os.path.join(trainer_results_dir, "config.json")

            if os.path.exists(config_filepath):
                logger.debug(
                    f"Skipping trainer {hparam_idx} - results already exist at {config_filepath}"
                )
                continue

            trainer_names.append(str(hparam_idx))

            architecture_config = ARCHITECTURES[current_architecture]

            trainer_cfg = {
                "trainer": architecture_config.trainer,
                **base_trainer_cfg,
                "dict_size": current_expansion_factor * activation_dim,
                "k": current_k,
                "layer": current_layer,
                "group_fractions": current_group_fractions,
                "group_weights": current_group_weights,
                "dict_class": architecture_config.sae,
                "lr": current_lr,
                "auxk_alpha": current_auxk_alpha,
                "warmup_steps": current_warmup_steps,
                "decay_start": current_decay_start,
                "threshold_beta": current_threshold_beta,
                "threshold_start_step": current_threshold_start_step,
                "k_anneal_steps": current_k_anneal_steps,
                "seed": current_seed,
                "device": f"cuda:{device_idx}",
                "submodule_name": current_submodule_name,
            }
            filtered_trainer_cfg = architecture_config.filter_constructor_args(
                trainer_cfg
            )

            trainer_cfgs.append(filtered_trainer_cfg)

        if len(trainer_cfgs) == 0:
            logger.info(
                f"Skipping trainer batch {trainer_batch_idx} - no trainers to run"
            )
            continue

        batch = GPUBatch(
            trainer_cfgs=trainer_cfgs,
            trainer_names=trainer_names,
            sae_experiment_name=sae_experiment_name,
            submodule_name=current_submodule_name,
        )
        logger.debug(f"Putting batch {trainer_batch_idx} into queue {device_idx}")
        await gpu_queue.put((0, trainer_batch_idx, batch))

    # put a sentinel value in the gpu queues to stop the workers
    for gpu_idx, gpu_queue in enumerate(gpu_queues):
        logger.debug(f"Putting sentinel value in queue {gpu_idx}")
        await gpu_queue.put((1, 0, None))

    logger.debug("Waiting for queues to finish")
    for gpu_queue in gpu_queues:
        await gpu_queue.join()

    logger.info("done :)")


DEFAULT_BATCH_SIZE = 4096
DEFAULT_DEBUG_BATCH_SIZE = 128
DEFAULT_STEPS = 1024 * 256
DEFAULT_DEBUG_STEPS = 1024 * 1
DEFAULT_WARMUP_STEPS = (1024 * 256 // 256,)
DEFAULT_DEBUG_WARMUP_STEPS = (1024 * 1 // 256,)


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    batch_size: int | None = None,
    trainers_per_gpu: int = 2,
    steps: int | None = None,
    save_every: int = 1024,
    num_epochs: int = 1,
    expansion_factor: tuple[int, ...] = (16,),
    k: tuple[int, ...] = (160,),
    layer: tuple[int, ...] = (7,),
    group_fractions: tuple[tuple[float, ...], ...] = (
        (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2 + 1.0 / 32),
    ),
    group_weights: tuple[tuple[float, ...] | None, ...] = (None,),
    architecture: tuple[str, ...] = ("batchtopk",),
    lr: tuple[float, ...] = (5e-5,),
    auxk_alpha: tuple[float, ...] = (1 / 32,),
    warmup_steps: tuple[int, ...] | None = None,
    decay_start: tuple[int | None, ...] = (None,),
    threshold_beta: tuple[float, ...] = (0.999,),
    threshold_start_step: tuple[int, ...] = (1024,),
    k_anneal_steps: tuple[int | None, ...] = (None,),
    seed: tuple[int, ...] = (0,),
    submodule_name: tuple[str, ...] = ("mlp_output",),
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 20_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
    dtype: str = "bf16",
) -> None:
    """Train a sparse autoencoder on the given model and dataset."""
    # Check if log level is valid by trying to get it
    try:
        logger.level(log_level)
    except ValueError as err:
        raise ValueError(f"Invalid log level: {log_level}") from err

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    log_level_numeric = logger.level(log_level).no
    debug_level_numeric = logger.level("DEBUG").no

    debug = log_level_numeric <= debug_level_numeric

    logger.debug(f"Running with log level: {log_level}")

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE if not debug else DEFAULT_DEBUG_BATCH_SIZE

    if steps is None:
        steps = DEFAULT_STEPS if not debug else DEFAULT_DEBUG_STEPS

    if warmup_steps is None:
        warmup_steps = DEFAULT_WARMUP_STEPS if not debug else DEFAULT_DEBUG_WARMUP_STEPS

    assert all(
        current_architecture in ARCHITECTURES for current_architecture in architecture
    ), "Invalid architecture"
    assert len(submodule_name) > 0, "Submodule name is an empty tuple!"

    torch_dtype = get_dtype(dtype)

    asyncio.run(
        run_sae_training(
            model_name=model_name,
            dataset_name=dataset_name,
            batch_size=batch_size,
            trainers_per_gpu=trainers_per_gpu,
            steps=steps,
            save_every=save_every,
            num_epochs=num_epochs,
            expansion_factor=expansion_factor,
            k=k,
            layer=layer,
            group_fractions=group_fractions,
            group_weights=group_weights,
            architecture=architecture,
            lr=lr,
            auxk_alpha=auxk_alpha,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            threshold_beta=threshold_beta,
            threshold_start_step=threshold_start_step,
            k_anneal_steps=k_anneal_steps,
            seed=seed,
            submodule_name=submodule_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            context_length=context_length,
            num_workers=num_workers,
            debug=log_level_numeric <= debug_level_numeric,
            dtype=torch_dtype,
        )
    )


if __name__ == "__main__":
    arguably.run()
