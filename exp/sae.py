import arguably
import os
from loguru import logger
import trackio as wandb
from nnterp import StandardizedTransformer
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers.trainer import SAETrainer
from dictionary_learning.trainers.dictionary import Dictionary
from dictionary_learning.trainers.top_k import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.training import trainSAE
from dataclasses import dataclass
import torch.distributed as dist

from core.data import DATASETS
from core.model import MODELS
from exp import OUTPUT_DIR, MODEL_DIRNAME
from exp.get_activations import get_experiment_name, ACTIVATION_DIRNAME
from exp.activations import Activations
from core.slurm import get_slurm_env


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
    layer_idx: int = 7,
    name: str | None = None,
    device: str = "auto",
    seed: int = 0,
    num_tokens: int = 1_000_000_000,  # 1B tokens
    architecture: str = "batchtopk",
    expansion_factor: int = 16,
    lr: float = 5e-5,
    context_length: int = 2048,
    tokens_per_file: int = 10_000,
) -> None:
    """Train a sparse autoencoder on the given model and dataset."""

    architecture_config = ARCHITECTURES.get(architecture)
    if architecture_config is None:
        raise ValueError(f"Architecture {architecture} not found")

    # Get model config
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

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
        reshuffle=True,
        seed=seed,
    )

    sae_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        layer_idx=layer_idx,
        seed=seed,
        num_tokens=num_tokens,
        architecture=architecture,
        expansion_factor=expansion_factor,
        lr=lr,
    ) if name is None else name

    activation_dim = model.config.hidden_size
    trainer_cfg = {
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 1e-3,
        "device": device,
        "steps": training_steps,
        "layer": layer,
        "lm_name": model_name,
        "warmup_steps": 1,
        "k": 100,
    }

    # # train the sparse autoencoder (SAE)
    # ae = trainSAE(
    #     data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    #     trainer_configs=[trainer_cfg],
    #     steps=training_steps,  # The number of training steps. Total trained tokens = steps * batch_size
    # )
