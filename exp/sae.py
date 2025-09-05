from dataclasses import dataclass
from itertools import chain, cycle, partial, product
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
) -> None:
    """Train a sparse autoencoder on the given model and dataset."""
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

    activations_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
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
    activation_dim[0] = activation[submodule_name[0]].shape[1]

    # clean up the background worker and queue
    data_iterable.send("STOP!")

    activation_dim = activation_dim.item()

    assert activation_dim > 0, "Activation dimension must be greater than 0"
    assert num_epochs > 0, "Number of epochs must be greater than 0"

    # to train for multiple epochs, we just repeat the data iterator
    data_iterator = chain(*[one_epoch_generator() for _ in range(num_epochs)])

    base_trainer_cfg = {
        "steps": steps,
        "activation_dim": activation_dim,
        "lm_name": model_name,
        "wandb_name": model_name,
    }

    trainer_cfgs = []

    num_gpus = th.cuda.device_count()

    for device_idx, (
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
    ) in cycle(range(num_gpus), product(
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
    )):
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

    sae_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        steps=steps,
        num_epochs=num_epochs,
    )

    # train the sparse autoencoder (SAE)
    trainSAE(
        data=data_iterator,
        trainer_configs=trainer_cfgs,
        steps=steps * num_epochs,
        use_wandb=False,
        save_dir=os.path.join(OUTPUT_DIR, sae_experiment_name),
        normalize_activations=True,
    )
