from itertools import product
import os
from typing import Any, Optional
import queue
import threading

import arguably
from loguru import logger
import torch as th
from tqdm import tqdm
import trackio as wandb

from exp import get_experiment_dir
from exp.activations import load_activations

# ... rest of the helper functions ...

@arguably.command()
def load_and_gradient_descent(
    top_k: int,
    complexity_importance: float = 0.4,
    complexity_power: float = 1.0,
    lr: float = 0.05,
    max_circuits: int = 48,
    num_epochs: int = 4096,
    num_warmup_epochs: int = 0,
    num_cooldown_epochs: int = 0,
    batch_size: int = 0,
    grad_accumulation_steps: int = 1,
    seed: int = 0,
    device: str = "cuda",
    experiment_name: Optional[str] = None,
) -> None:
    data = load_activations(experiment_name=experiment_name, device=device)

    wandb_run = wandb.init(
        project="circuit-optimization",
        name=f"topk={top_k};ci={complexity_importance};cp={complexity_power};lr={lr};mc={max_circuits};ne={num_epochs};nw={num_warmup_epochs};nc={num_cooldown_epochs};s={seed}",
        config={
            "top_k": top_k,
            "complexity_importance": complexity_importance,
            "complexity_power": complexity_power,
            "max_circuits": max_circuits,
            "num_epochs": num_epochs,
            "num_warmup_epochs": num_warmup_epochs,
            "num_cooldown_epochs": num_cooldown_epochs,
            "lr": lr,
            "seed": seed,
        },
    )

    circuits, loss, faithfulness, complexity = gradient_descent(
        data,
        top_k,
        complexity_importance,
        complexity_power,
        lr,
        max_circuits,
        num_epochs,
        num_warmup_epochs,
        num_cooldown_epochs,
        batch_size,
        grad_accumulation_steps,
        device=device,
        seed=seed,
        wandb_run=wandb_run,
    )

    out = {
        "circuits": circuits,
        "top_k": top_k,
        "loss": loss,
        "faithfulness": faithfulness,
        "complexity": complexity,
    }

    # Get experiment directory
    experiment_dir = get_experiment_dir(name=experiment_name)
    out_path = os.path.join(experiment_dir, "optimized_circuits.pt")
    os.makedirs(experiment_dir, exist_ok=True)
    th.save(out, out_path)

    wandb.finish()


@arguably.command()
def grid_search_gradient_descent(
    top_k: int,
    complexity_importances: list[float] | None = None,
    complexity_powers: list[float] | None = None,
    lrs: list[float] | None = None,
    max_circuitses: list[int] | None = None,
    num_epochses: list[int] | None = None,
    num_warmup_epochses: list[int] | None = None,
    num_cooldown_epochses: list[int] | None = None,
    num_seeds: int = 1,
    batch_size: int = 0,
    grad_accumulation_steps: int = 1,
    device: str = "cuda",
    experiment_name: Optional[str] = None,
) -> None:
    if complexity_importances is None:
        # complexity_importances = [0.5, 0.9, 0.1, 0.99, 0.01]
        # complexity_importances = [0.5, 0.4, 0.3, 0.2, 0.1]
        # complexity_importances = [0.6, 0.5, 0.4, 0.3]
        # complexity_importances = [0.6, 0.5, 0.4]
        # complexity_importances = [0.4, 0.3]
        complexity_importances = [0.6, 0.5, 0.4]
    if complexity_powers is None:
        # complexity_powers = [1.0, 2.0, 0.5]
        # complexity_powers = [1.0]
        # complexity_powers = [1.0]
        # complexity_powers = [1.0]
        # complexity_powers = [1.0]
        complexity_powers = [1.0, 0.5]
    if lrs is None:
        # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        # lrs = [0.05, 0.02, 0.01, 0.005, 0.002]
        # lrs = [0.05, 0.02, 0.01]
        # lrs = [0.05]
        # lrs = [0.05]
        lrs = [0.05]
    if max_circuitses is None:
        # max_circuitses = [64, 128, 256, 512]
        # max_circuitses = [32, 64, 128, 256]
        # max_circuitses = [32, 64, 128, 256]
        # max_circuitses = [32, 48, 64]
        # max_circuitses = [32, 48, 64]
        max_circuitses = [64, 128, 256]
    if num_epochses is None:
        # num_epochses = [2048]
        # num_epochses = [4096]
        # num_epochses = [8192]
        # num_epochses = [4096]
        # num_epochses = [4096]
        num_epochses = [4096]
    if num_warmup_epochses is None:
        # num_warmup_epochses = [0]
        # num_warmup_epochses = [0]
        # num_warmup_epochses = [0]
        # num_warmup_epochses = [0]
        # num_warmup_epochses = [0]
        num_warmup_epochses = [0]
    if num_cooldown_epochses is None:
        # num_cooldown_epochses = [512]
        # num_cooldown_epochses = [1024]
        # num_cooldown_epochses = [2048]
        # num_cooldown_epochses = [0]
        # num_cooldown_epochses = [0]
        num_cooldown_epochses = [0]

    data = load_activations(experiment_name=experiment_name, device=device)
    loss_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        len(max_circuitses),
        len(num_epochses),
        len(num_warmup_epochses),
        len(num_cooldown_epochses),
        device=device,
    )
    faithfulness_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        len(max_circuitses),
        len(num_epochses),
        len(num_warmup_epochses),
        len(num_cooldown_epochses),
        device=device,
    )
    complexity_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        len(max_circuitses),
        len(num_epochses),
        len(num_warmup_epochses),
        len(num_cooldown_epochses),
        device=device,
    )
    # Store the circuits themselves
    circuits_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        len(max_circuitses),
        len(num_epochses),
        len(num_warmup_epochses),
        len(num_cooldown_epochses),
        device=device,
    )

    for (
        (seed_idx, seed),
        (complexity_importance_idx, complexity_importance),
        (complexity_power_idx, complexity_power),
        (lr_idx, lr),
        (max_circuits_idx, max_circuits),
        (num_epochs_idx, num_epochs),
        (num_warmup_epochs_idx, num_warmup_epochs),
        (num_cooldown_epochs_idx, num_cooldown_epochs),
    ) in tqdm(
        product(
            enumerate(range(num_seeds)),
            enumerate(complexity_importances),
            enumerate(complexity_powers),
            enumerate(lrs),
            enumerate(max_circuitses),
            enumerate(num_epochses),
            enumerate(num_warmup_epochses),
            enumerate(num_cooldown_epochses),
        ),
        desc="Grid search",
        total=num_seeds
        * len(complexity_importances)
        * len(complexity_powers)
        * len(lrs)
        * len(max_circuitses)
        * len(num_epochses)
        * len(num_warmup_epochses)
        * len(num_cooldown_epochses),
    ):
        wandb_run = wandb.init(
            project="circuit-optimization",
            name=f"ci={complexity_importance};cp={complexity_power};lr={lr};s={seed};mc={max_circuits};ne={num_epochs};nw={num_warmup_epochs};nc={num_cooldown_epochs}",
            config={
                "complexity_importance": complexity_importance,
                "complexity_power": complexity_power,
                "lr": lr,
                "max_circuits": max_circuits,
                "num_epochs": num_epochs,
                "num_warmup_epochs": num_warmup_epochs,
                "num_cooldown_epochs": num_cooldown_epochs,
                "seed": seed,
            },
        )

        circuits, loss, faithfulness, complexity = gradient_descent(
            data,
            top_k,
            complexity_importance,
            complexity_power,
            lr=lr,
            max_circuits=max_circuits,
            num_epochs=num_epochs,
            num_warmup_epochs=num_warmup_epochs,
            num_cooldown_epochs=num_cooldown_epochs,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            device=device,
            seed=seed,
            wandb_run=wandb_run,
        )
        loss_landscape[
            seed_idx,
            complexity_importance_idx,
            complexity_power_idx,
            lr_idx,
            max_circuits_idx,
            num_epochs_idx,
            num_warmup_epochs_idx,
            num_cooldown_epochs_idx,
        ] = loss
        faithfulness_landscape[
            seed_idx,
            complexity_importance_idx,
            complexity_power_idx,
            lr_idx,
            max_circuits_idx,
            num_epochs_idx,
            num_warmup_epochs_idx,
            num_cooldown_epochs_idx,
        ] = faithfulness
        complexity_landscape[
            seed_idx,
            complexity_importance_idx,
            complexity_power_idx,
            lr_idx,
            max_circuits_idx,
            num_epochs_idx,
            num_warmup_epochs_idx,
            num_cooldown_epochs_idx,
        ] = complexity
        circuits_landscape[
            seed_idx,
            complexity_importance_idx,
            complexity_power_idx,
            lr_idx,
            max_circuits_idx,
            num_epochs_idx,
            num_warmup_epochs_idx,
            num_cooldown_epochs_idx,
        ] = circuits

    out = {
        "loss_landscape": loss_landscape,
        "faithfulness_landscape": faithfulness_landscape,
        "complexity_landscape": complexity_landscape,
        "circuits_landscape": circuits_landscape,
        "complexity_importances": complexity_importances,
        "complexity_powers": complexity_powers,
        "lrs": lrs,
        "max_circuitses": max_circuitses,
        "num_epochses": num_epochses,
        "num_warmup_epochses": num_warmup_epochses,
        "num_cooldown_epochses": num_cooldown_epochses,
    }
    # Get experiment directory
    experiment_dir = get_experiment_dir(name=experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    th.save(out, os.path.join(experiment_dir, "loss_landscape.pt"))
    wandb.finish()


if __name__ == "__main__":
    # arguably.run()
    grid_search_gradient_descent(top_k=8, grad_accumulation_steps=4)
