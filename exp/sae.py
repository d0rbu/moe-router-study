import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched, chain, islice, product
import os
import sys

import arguably
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.trainer import SAETrainer
from dictionary_learning.training import trainSAE
from fsspec.utils import math
from loguru import logger
import torch as th
import torch.distributed as dist
from tqdm import tqdm

from exp import OUTPUT_DIR
from exp.activations import load_activations_and_init_dist
from exp.get_activations import ActivationKeys
from exp.training import exponential_to_linear_save_steps, get_experiment_name


@dataclass
class Architecture:
    trainer: type[SAETrainer]
    sae: type[Dictionary]


ARCHITECTURES = {
    "batchtopk": Architecture(
        trainer=BatchTopKTrainer,
        sae=BatchTopKSAE,
    ),
    "matryoshka": Architecture(
        trainer=MatryoshkaBatchTopKTrainer,
        sae=MatryoshkaBatchTopKSAE,
    ),
}


@dataclass
class GPUBatch:
    trainer_cfgs: list[dict]
    trainer_names: list[str]
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

    while True:
        batch: GPUBatch | None = await gpu_queue.get()
        if batch is None:
            break

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


@arguably.command()
async def run_sae_training(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    batch_size: int = 4096,
    trainers_per_gpu: int = 2,
    steps: int = 1024 * 256,
    save_every: int = 1024,
    num_epochs: int = 1,
    expansion_factor: list[int],
    k: list[int],
    layer: list[int],
    group_fractions: list[list[float]],
    group_weights: list[list[float] | None],
    architecture: list[str],
    lr: list[float],
    auxk_alpha: list[float],
    warmup_steps: list[int],
    decay_start: list[int | None],
    threshold_beta: list[float],
    threshold_start_step: list[int],
    k_anneal_steps: list[int | None],
    seed: list[int],
    submodule_name: list[str],
    tokens_per_file: int = 10_000,
) -> None:
    """Train autoencoders to sweep over the given hyperparameter sets."""
    logger.debug("loading activations and initializing distributed setup")

    activations, activation_dims = load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        submodule_names=submodule_name,
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
            submodule_name
            for submodule_name in submodule_name
            if submodule_name != ActivationKeys.ROUTER_LOGITS
        ]
        activation_dim = activation_dims[non_router_logits_submodules[0]]

    assert activation_dim > 0, "Activation dimension must be greater than 0"
    assert num_epochs > 0, "Number of epochs must be greater than 0"

    # to train for multiple epochs, we just repeat the data iterator
    def data_iterator():
        yield from chain(
            *[activations(batch_size=batch_size) for _ in range(num_epochs)]
        )

    save_steps = exponential_to_linear_save_steps(total_steps=steps, save_every=save_every)

    base_trainer_cfg = {
        "steps": steps,
        "save_steps": save_steps,
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
            start=dist.get_rank(),
            stop=None,
            step=dist.get_world_size(),
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

    if dist.get_rank() == 0:
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
                "expansion_factor": current_expansion_factor,
                "k": current_k,
                "layer": current_layer,
                "group_fractions": current_group_fractions,
                "group_weights": current_group_weights,
                "lr": current_lr,
                "dict_class": architecture_config.sae,
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
            sae_experiment_name=sae_experiment_name,
            steps=steps,
            num_epochs=num_epochs,
        )
        gpu_queues[device_idx].put(batch)

    # put a sentinel value in the gpu queues to stop the workers
    for gpu_queue in gpu_queues:
        gpu_queue.put(None)
        gpu_queue.join()

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
    expansion_factor: list[int] | int = 16,
    k: list[int] | int = 160,
    layer: list[int] | int = 7,
    group_fractions: list[list[float]] | list[float] | None = None,
    group_weights: list[list[float] | None] | list[float] | None = None,
    architecture: list[str] | str = "batchtopk",
    lr: list[float] | float = 5e-5,
    auxk_alpha: list[float] | float = 1 / 32,
    warmup_steps: list[int] | int = 1024,
    decay_start: list[int | None] | int | None = None,
    threshold_beta: list[float] | float = 0.999,
    threshold_start_step: list[int] | int = 1024,
    k_anneal_steps: list[int | None] | int | None = None,
    seed: list[int] | int = 0,
    submodule_name: list[str] | str = "mlp_output",
    tokens_per_file: int = 10_000,
    log_level: str = "INFO"
) -> None:
    """Train a sparse autoencoder on the given model and dataset."""
    print(f"Running with log level: {log_level}")

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.debug(f"Running with log level: {log_level}")

    if isinstance(expansion_factor, int):
        expansion_factor = [expansion_factor]

    if isinstance(k, int):
        k = [k]

    if isinstance(layer, int):
        layer = [layer]

    if isinstance(group_fractions, list):
        assert len(group_fractions) > 0, "Group fractions is an empty list!"
        if isinstance(group_fractions[0], float):
            group_fractions = [group_fractions]
    elif group_fractions is None:
        group_fractions = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2 + 1 / 32]

    if isinstance(group_weights, list):
        assert len(group_weights) > 0, "Group weights is an empty list!"
        if isinstance(group_weights[0], float):
            group_weights = [group_weights]
    elif group_weights is None:
        group_weights = [None]

    if isinstance(architecture, str):
        architecture = [architecture]

    if isinstance(lr, int):
        lr = [lr]

    if isinstance(auxk_alpha, int):
        auxk_alpha = [auxk_alpha]

    if isinstance(warmup_steps, int):
        warmup_steps = [warmup_steps]

    if isinstance(decay_start, int | None):
        decay_start = [decay_start]

    if isinstance(threshold_beta, float):
        threshold_beta = [threshold_beta]

    if isinstance(threshold_start_step, int):
        threshold_start_step = [threshold_start_step]

    if isinstance(k_anneal_steps, int | None):
        k_anneal_steps = [k_anneal_steps]

    if isinstance(seed, int):
        seed = [seed]

    if isinstance(submodule_name, str):
        submodule_name = [submodule_name]

    assert all(
        current_architecture in ARCHITECTURES for current_architecture in architecture
    ), "Invalid architecture"
    assert len(submodule_name) > 0, "Submodule name is an empty list!"

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
        )
    )


if __name__ == "__main__":
    arguably.run()
