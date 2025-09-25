import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import batched, chain, count, islice, product
import math
import os
import sys

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

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from exp.training import exponential_to_linear_save_steps, get_experiment_name


@dataclass
class Architecture:
    saebench_load_fn: Callable[
        [str, str, str, th.device, th.dtype, int | None, str], BaseSAE
    ]
    trainer: type[SAETrainer]
    sae: type[Dictionary]


ARCHITECTURES = {
    "batchtopk": Architecture(
        saebench_load_fn=load_dictionary_learning_batch_topk_sae,
        trainer=BatchTopKTrainer,
        sae=BatchTopKSAE,
    ),
    "matryoshka": Architecture(
        saebench_load_fn=load_dictionary_learning_matryoshka_batch_topk_sae,
        trainer=MatryoshkaBatchTopKTrainer,
        sae=MatryoshkaBatchTopKSAE,
    ),
}


@dataclass
class GPUBatch:
    trainer_cfgs: list[dict]
    trainer_names: list[str]
    save_steps: list[int]
    steps: int
    num_epochs: int
    sae_experiment_name: str


MAX_GPU_QUEUE_SIZE = 2


async def gpu_worker(
    device_idx: int,
    num_gpus: int,
    gpu_queue: asyncio.Queue,
    data_iterator: Iterator[dict],
) -> None:
    """Worker process for training SAE models on a specific GPU."""

    logger.info(f"Starting GPU worker {device_idx}")

    for worker_batch_idx in count():
        logger.debug(f"GPU worker {device_idx} awaiting batch {worker_batch_idx}")
        batch: GPUBatch | None = await gpu_queue.get()

        if batch is None:
            logger.debug(f"GPU worker {device_idx} stopping")
            break

        logger.debug(f"GPU worker {device_idx} got batch {worker_batch_idx}")
        await trainSAE(
            data=data_iterator,
            trainer_configs=batch.trainer_cfgs,
            steps=batch.steps * batch.num_epochs,
            trainer_names=batch.trainer_names,
            use_wandb=False,
            save_dir=os.path.join(OUTPUT_DIR, batch.sae_experiment_name),
            normalize_activations=True,
            tqdm_kwargs={"position": dist.get_rank() * (num_gpus + 1) + device_idx + 1},
        )


async def run_sae_training(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    expansion_factor: tuple[int],
    k: tuple[int],
    layer: tuple[int],
    group_fractions: tuple[tuple[float]],
    group_weights: tuple[tuple[float] | None],
    architecture: tuple[str],
    lr: tuple[float],
    auxk_alpha: tuple[float],
    warmup_steps: tuple[int],
    decay_start: tuple[int | None],
    threshold_beta: tuple[float],
    threshold_start_step: tuple[int],
    k_anneal_steps: tuple[int | None],
    seed: tuple[int],
    submodule_name: tuple[str, ...],
    batch_size: int = 4096,
    trainers_per_gpu: int = 2,
    steps: int = 1024 * 256,
    save_every: int = 1024,
    num_epochs: int = 1,
    context_length: int = 2048,
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    num_workers: int = 64,
    debug: bool = False,
) -> None:
    """Train autoencoders to sweep over the given hyperparameter sets."""
    assert "moe" not in architecture, (
        "MoE is not supported for SAE training, use kmeans.py instead."
    )
    assert th.cuda.is_available(), "CUDA is not available"

    logger.debug("loading activations and initializing distributed setup")

    activations, activation_dims = await load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        submodule_names=submodule_name,
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

    # to train for multiple epochs, we just repeat the data iterator
    def data_iterator():
        yield from chain(
            *[activations(batch_size=batch_size) for _ in range(num_epochs)]
        )

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
    gpu_queues = [asyncio.Queue(maxsize=MAX_GPU_QUEUE_SIZE) for _ in range(num_gpus)]

    _workers = [
        asyncio.create_task(
            gpu_worker(device_idx, num_gpus, gpu_queues[device_idx], data_iterator)
        )
        for device_idx in range(num_gpus)
    ]

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

    # assign a subset of the hparam sweep to each rank
    distributed_iterator = list(
        islice(
            hparam_sweep_iterator,
            dist.get_rank(),
            None,
            dist.get_world_size(),
        )
    )
    distributed_iterator = tqdm(
        distributed_iterator,
        desc=f"Rank {dist.get_rank()}",
        total=len(distributed_iterator),
        leave=True,
        position=dist.get_rank() * (num_gpus + 1),
    )
    # batch it based on how many trainers will be on each gpu
    concurrent_trainer_batched_iterator = batched(
        distributed_iterator, trainers_per_gpu
    )

    logger.info(f"Total size of sweep: {len(hparam_sweep_iterator)}")
    logger.info(f"Number of nodes: {dist.get_world_size()}")
    logger.info(f"Number of GPUs per node: {num_gpus}")
    logger.info(f"Number of trainers per GPU: {trainers_per_gpu}")
    logger.info(
        f"Number of iterations: {math.ceil(len(hparam_sweep_iterator) / (trainers_per_gpu * num_gpus * dist.get_world_size()))}"
    )

    for trainer_batch in concurrent_trainer_batched_iterator:
        # decide device_idx based on how full the gpu queues are
        device_idx = th.argmin(th.tensor([q.qsize() for q in gpu_queues])).item()

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
            trainer_names.append(str(hparam_idx))

            architecture_config = ARCHITECTURES[current_architecture]

            trainer_cfg = {
                **base_trainer_cfg,
                "dict_size": current_expansion_factor * activation_dim,
                "k": current_k,
                "layer": current_layer,
                "group_fractions": current_group_fractions,
                "group_weights": current_group_weights,
                "lr": current_lr,
                "dict_class": architecture_config.sae,
                "trainer": architecture_config.trainer,
                "auxk_alpha": current_auxk_alpha,
                "warmup_steps": current_warmup_steps,
                "decay_start": current_decay_start,
                "threshold_beta": current_threshold_beta,
                "threshold_start_step": current_threshold_start_step,
                "k_anneal_steps": current_k_anneal_steps,
                "seed": current_seed,
                "submodule_name": current_submodule_name,
                "device": f"cuda:{device_idx}",
            }
            trainer_cfgs.append(trainer_cfg)

        batch = GPUBatch(
            trainer_cfgs=trainer_cfgs,
            trainer_names=trainer_names,
            save_steps=save_steps,
            sae_experiment_name=sae_experiment_name,
            steps=steps,
            num_epochs=num_epochs,
        )
        logger.debug(f"Putting batch {hparam_idx} into queue {device_idx}")
        await gpu_queues[device_idx].put(batch)

    # put a sentinel value in the gpu queues to stop the workers
    for gpu_queue in gpu_queues:
        await gpu_queue.put(None)
        await gpu_queue.join()

    logger.info("done :)")


@arguably.command()
def main(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    batch_size: int = 4096,
    trainers_per_gpu: int = 2,
    steps: int = 1024 * 256,
    save_every: int = 1024,
    num_epochs: int = 1,
    expansion_factor: tuple[int] = (16,),
    k: tuple[int] = (160,),
    layer: tuple[int] = (7,),
    group_fractions: tuple[tuple[float]] = (
        (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2 + 1.0 / 32),
    ),
    group_weights: tuple[tuple[float]] = (),
    architecture: tuple[str] = ("batchtopk",),
    lr: tuple[float] = (5e-5,),
    auxk_alpha: tuple[float] = (1 / 32,),
    warmup_steps: tuple[int] = (1024,),
    decay_start: tuple[int] = (),
    threshold_beta: tuple[float] = (0.999,),
    threshold_start_step: tuple[int] = (1024,),
    k_anneal_steps: tuple[int] = (),
    seed: tuple[int] = (0,),
    submodule_name: tuple[str] = ("mlp_output",),
    tokens_per_file: int = 5_000,
    reshuffled_tokens_per_file: int = 10_000,
    context_length: int = 2048,
    log_level: str = "INFO",
    num_workers: int = 64,
) -> None:
    """Train a sparse autoencoder on the given model and dataset."""
    print(f"Running with log level: {log_level}")

    if len(group_weights) == 0:
        group_weights = (None,)

    if len(decay_start) == 0:
        decay_start = (None,)

    if len(k_anneal_steps) == 0:
        k_anneal_steps = (None,)

    assert log_level in logger._core.levels, (
        f"Invalid log level, must be one of {logger._core.levels.keys()}"
    )

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    log_level_numeric = logger._core.levels[log_level].no
    debug_level_numeric = logger._core.levels["DEBUG"].no

    logger.debug(f"Running with log level: {log_level}")

    assert all(
        current_architecture in ARCHITECTURES for current_architecture in architecture
    ), "Invalid architecture"
    assert len(submodule_name) > 0, "Submodule name is an empty tuple!"

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
        )
    )


if __name__ == "__main__":
    arguably.run()
