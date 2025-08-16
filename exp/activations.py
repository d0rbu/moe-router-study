import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

from core.utils import get_device


def load_activations_and_topk(
    activations_dir: Optional[str] = None,
    topk_dir: Optional[str] = None,
) -> Tuple[th.Tensor, List[List[List[int]]]]:
    """
    Load the activations and top-k experts.

    Args:
        activations_dir: The directory containing the activations.
        topk_dir: The directory containing the top-k experts.

    Returns:
        A tuple of (activated_experts, top_k).
    """
    if activations_dir is None:
        activations_dir = "activations/mistralai/Mistral-7B-v0.1"

    if topk_dir is None:
        topk_dir = activations_dir

    # get the latest file
    activated_experts_files = sorted(
        Path(activations_dir).glob("activated_experts_*.pt"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    top_k_files = sorted(
        Path(topk_dir).glob("top_k_*.pkl"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )

    if not activated_experts_files:
        raise ValueError(f"No activated_experts files found in {activations_dir}")

    if not top_k_files:
        raise ValueError(f"No top_k files found in {topk_dir}")

    # load the latest file
    activated_experts = th.load(
        activated_experts_files[-1], map_location=get_device()
    )

    with open(top_k_files[-1], "rb") as f:
        top_k = pickle.load(f)

    return activated_experts, top_k


def get_expert_importance(
    activated_experts: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Get the importance of each expert.

    Args:
        activated_experts: The activated experts.

    Returns:
        A tuple of (expert_importance, expert_importance_by_layer).
    """
    batch_size, num_layers, num_experts = activated_experts.shape

    # get the importance of each expert
    expert_importance = activated_experts.sum(dim=0) / batch_size
    expert_importance_by_layer = expert_importance.sum(dim=1)

    return expert_importance, expert_importance_by_layer


def plot_expert_importance(
    expert_importance: th.Tensor,
    expert_importance_by_layer: th.Tensor,
    save_dir: Optional[str] = None,
) -> None:
    """
    Plot the importance of each expert.

    Args:
        expert_importance: The importance of each expert.
        expert_importance_by_layer: The importance of each expert by layer.
        save_dir: The directory to save the plots to.
    """
    num_layers, num_experts = expert_importance.shape

    # plot the importance of each expert
    plt.figure(figsize=(20, 10))
    plt.imshow(expert_importance.cpu().numpy())
    plt.colorbar()
    plt.xlabel("Expert")
    plt.ylabel("Layer")
    plt.title("Expert Importance")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/expert_importance.png")
    else:
        plt.show()

    # plot the importance of each expert by layer
    plt.figure(figsize=(20, 10))
    plt.bar(range(num_layers), expert_importance_by_layer.cpu().numpy())
    plt.xlabel("Layer")
    plt.ylabel("Importance")
    plt.title("Expert Importance by Layer")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/expert_importance_by_layer.png")
    else:
        plt.show()


def plot_expert_importance_histogram(
    expert_importance: th.Tensor,
    save_dir: Optional[str] = None,
) -> None:
    """
    Plot the histogram of expert importance.

    Args:
        expert_importance: The importance of each expert.
        save_dir: The directory to save the plots to.
    """
    num_layers, num_experts = expert_importance.shape

    # plot the histogram of expert importance
    plt.figure(figsize=(20, 10))
    plt.hist(expert_importance.cpu().numpy().flatten(), bins=100)
    plt.xlabel("Importance")
    plt.ylabel("Count")
    plt.title("Expert Importance Histogram")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/expert_importance_histogram.png")
    else:
        plt.show()


def main():
    activated_experts, top_k = load_activations_and_topk()

    expert_importance, expert_importance_by_layer = get_expert_importance(
        activated_experts
    )

    plot_expert_importance(
        expert_importance, expert_importance_by_layer, save_dir="figures"
    )
    plot_expert_importance_histogram(expert_importance, save_dir="figures")


if __name__ == "__main__":
    main()

