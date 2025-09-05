from dataclasses import dataclass
from itertools import chain, partial
import os

import arguably
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.trainer import SAETrainer
from dictionary_learning.training import trainSAE
import torch as th

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
    num_epochs: int = 1,
    expansion_factor: int = 16,
    k: int = 160,
    layer: int = 7,
    group_fractions: list[float] | None = None,
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
    if group_fractions is None:
        group_fractions = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2 + 1 / 32]

    architecture_config = ARCHITECTURES.get(architecture)
    if architecture_config is None:
        raise ValueError(f"Architecture {architecture} not found")

    activations_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        context_length=context_length,
        tokens_per_file=tokens_per_file,
    )

    activations = Activations(
        experiment_name=activations_experiment_name,
        seed=seed,
    )
    one_epoch_generator = partial(activations, batch_size=batch_size)

    activation_dim = th.zeros(1, dtype=th.int32)

    # load a batch of activations to get the dimension
    data_iterable = one_epoch_generator()
    activation = next(data_iterable)
    activation_dim[0] = activation[submodule_name].shape[1]

    # clean up the background worker and queue
    data_iterable.send("STOP!")

    activation_dim = activation_dim.item()

    assert activation_dim > 0, "Activation dimension must be greater than 0"
    assert num_epochs > 0, "Number of epochs must be greater than 0"

    # to train for multiple epochs, we just repeat the data iterator
    data_iterator = chain(*[one_epoch_generator() for _ in range(num_epochs)])

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
        data=data_iterator,
        trainer_configs=[trainer_cfg],
        steps=steps * num_epochs,
        use_wandb=False,
        save_dir=os.path.join(OUTPUT_DIR, sae_experiment_name),
        normalize_activations=True,
        device=device,
    )
