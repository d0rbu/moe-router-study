"""
Experiment to extract router activations from a model.
"""

import os
from typing import Optional, Set

import torch as th
from nnterp.standardized_transformer import StandardizedTransformer

from exp.activations import (
    get_activation_filepaths,
    get_router_logits,
    get_router_probs,
    get_router_tokens,
)
from exp.get_activations import get_experiment_name, process_batch


def get_router_activations(
    model_name: str,
    dataset_name: str,
    num_batches: int = 1,
    batch_size: int = 32,
    gpu_minibatch_size: int = 8,
    seed: int = 42,
    experiment_name: Optional[str] = None,
    activations_to_store: Set[str] = {"router_logits"},
    router_layers: Optional[Set[int]] = None,
) -> str:
    """
    Extract router activations from a model.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset to use
        num_batches: Number of batches to process
        batch_size: Number of examples per batch
        gpu_minibatch_size: Number of examples to process at once on the GPU
        seed: Random seed
        experiment_name: Name of the experiment (default: auto-generated)
        activations_to_store: Set of activation types to store
        router_layers: Set of router layers to extract activations from (default: all)

    Returns:
        Name of the experiment
    """
    if experiment_name is None:
        experiment_name = get_experiment_name(
            model_name=model_name,
            dataset_name=dataset_name,
            num_batches=num_batches,
            batch_size=batch_size,
            seed=seed,
        )

    # Process batches to extract activations
    process_batch(
        model_name=model_name,
        dataset_name=dataset_name,
        num_batches=num_batches,
        batch_size=batch_size,
        gpu_minibatch_size=gpu_minibatch_size,
        seed=seed,
        experiment_name=experiment_name,
        activations_to_store=activations_to_store,
        router_layers=router_layers,
    )

    return experiment_name

