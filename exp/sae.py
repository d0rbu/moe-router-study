from dataclasses import dataclass

import arguably
from dictionary_learning.trainers.dictionary import Dictionary
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.trainer import SAETrainer

from core.model import MODELS


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
    _model_name: str = "olmoe-i",
    _dataset_name: str = "lmsys",
    _layer_idx: int = 7,
    *_args,
    _name: str | None = None,
    _device: str = "auto",
    _seed: int = 0,
    _num_tokens: int = 1_000_000_000,  # 1B tokens
    architectures: list[str] | None = None,
    expansion_factors: list[int] | None = None,
    lrs: list[float] | None = None,
) -> None:
    if architectures is None:
        architectures = ["batchtopk", "matryoshka"]
    if expansion_factors is None:
        expansion_factors = [16]
    if lrs is None:
        lrs = [5e-5]

    # Get model config
    model_config = MODELS.get(_model_name)
    if model_config is None:
        raise ValueError(f"Model {_model_name} not found")

    # name = get_experiment_name(
    #     model_name=model_name,
    #     dataset_name=dataset_name,
    # )
    # activation_dim = model.config.hidden_size
    # trainer_cfg = {
    #     "trainer": TopKTrainer,
    #     "dict_class": AutoEncoderTopK,
    #     "activation_dim": activation_dim,
    #     "dict_size": dictionary_size,
    #     "lr": 1e-3,
    #     "device": device,
    #     "steps": training_steps,
    #     "layer": layer,
    #     "lm_name": model_name,
    #     "warmup_steps": 1,
    #     "k": 100,
    # }

    # # train the sparse autoencoder (SAE)
    # ae = trainSAE(
    #     data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    #     trainer_configs=[trainer_cfg],
    #     steps=training_steps,  # The number of training steps. Total trained tokens = steps * batch_size
    # )
