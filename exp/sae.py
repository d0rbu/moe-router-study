from dataclasses import dataclass
import os

import arguably
from dictionary_learning.trainers.dictionary import Dictionary
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.trainer import SAETrainer
from dictionary_learning.training import trainSAE
import torch as th
import torch.distributed as dist

from exp import OUTPUT_DIR
from exp.activations import Activations
from exp.get_activations import get_experiment_name


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


@arguably.command()
def run_sae_training(
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    *_args,
    batch_size: int = 4096,
    steps: int = 1024 * 256,
    expansion_factor: int = 16,
    k: int = 160,
    layer: int = 7,
    group_fractions: list[float] = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2 + 1 / 32],
    group_weights: list[float] | None = None,
    architecture: str = "batchtopk",
    lr: float = 5e-5,
    auxk_alpha: float = 1 / 32,
    warmup_steps: int = 1024,
    decay_start: int | None = None,
    threshold_beta: float = 0.999,
    threshold_start_step: int = 1024,
    k_anneal_steps: int | None = None,
    seed: int = 0,
    device: str = "auto",
    submodule_name: str = "mlp_output",
    name: str | None = None,
    context_length: int = 2048,
    tokens_per_file: int = 10_000,
) -> None:
    """Train a sparse autoencoder on the given model and dataset."""

    architecture_config = ARCHITECTURES.get(architecture)
    if architecture_config is None:
        raise ValueError(f"Architecture {architecture} not found")

    activations_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        context_length=context_length,
        tokens_per_file=tokens_per_file,
    )

    dist.init_process_group(backend="nccl")

    activations = Activations(
        experiment_name=activations_experiment_name,
        device=device,
        seed=seed,
    )

    activation_dim = th.zeros(1, dtype=th.int32)
    if dist.get_rank() == 0:
        # load a batch of activations to get the dimension
        iterator = iter(activations(batch_size=batch_size, ctx_len=context_length))
        activation = next(iterator)
        activation_dim[0] = activation[submodule_name].shape[1]
        dist.broadcast(activation_dim, src=0)
    else:
        dist.broadcast(activation_dim, src=0)

    activation_dim = activation_dim.item()

    assert activation_dim > 0, "Activation dimension must be greater than 0"

    trainer_cfg = {
        "steps": steps,
        "activation_dim": activation_dim,
        "dict_size": activation_dim * expansion_factor,
        "k": k,
        "layer": layer,
        "lm_name": model_name,
        "group_fractions": group_fractions,
        "group_weights": group_weights,
        "dict_class": architecture_config.sae,
        "lr": lr,
        "auxk_alpha": auxk_alpha,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "threshold_beta": threshold_beta,
        "threshold_start_step": threshold_start_step,
        "k_anneal_steps": k_anneal_steps,
        "seed": seed,
        "device": device,
        "wandb_name": model_name,
        "submodule_name": submodule_name,
    }

    sae_experiment_name = name or get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        **trainer_cfg,
    )

    # train the sparse autoencoder (SAE)
    trainSAE(
        data=activations,
        trainer_configs=[trainer_cfg],
        steps=steps,
        use_wandb=False,
        save_dir=os.path.join(OUTPUT_DIR, sae_experiment_name),
        normalize_activations=True,
        device=device,
    )
