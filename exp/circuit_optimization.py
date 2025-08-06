from contextlib import suppress
from itertools import product
import queue
import threading

import arguably
import torch as th
from tqdm import tqdm
import trackio as wandb

from exp.activations import load_activations
from exp.circuit_loss import circuit_loss


def _async_wandb_logger(wandb_run: wandb.Run, log_queue: queue.Queue):
    """Background thread for async wandb logging."""
    while True:
        try:
            log_data = log_queue.get(timeout=1.0)  # 1 second timeout
            if log_data is None:  # Sentinel to stop the thread
                break
            if not isinstance(log_data, dict):
                raise ValueError(f"Log data must be a dictionary, got {type(log_data)}")

            for key, value in log_data.items():
                if isinstance(value, th.Tensor):
                    if value.ndim == 0:
                        log_data[key] = value.item()
                    elif value.ndim == 1:
                        log_data[key] = value.tolist()
                    else:
                        raise ValueError(f"Tensor must be 0D or 1D, got {value.ndim}D")

            wandb_run.log(log_data)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in async wandb logging: {e}")


def gradient_descent(
    data: th.Tensor,
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
    max_circuits: int = 256,
    num_epochs: int = 1024,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cuda",
    wandb_run: wandb.Run | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Gradient descent for the optimal circuit.

    Args:
        data: The expert activations.
        complexity_importance: The complexity importance.
        complexity_power: The complexity power.
        max_circuits: The maximum number of circuits to search over.
        num_epochs: The number of epochs to run.
        lr: The learning rate.
        seed: The random seed.

    Returns:
        The optimal set of circuits, the loss, the faithfulness, and the complexity.
    """

    assert data.ndim == 3, "Data must be of shape (B, L, E)"

    batch_size, num_layers, num_experts = data.shape

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    circuits_logits = th.randn(max_circuits, num_layers, num_experts, device=device)
    circuits_parameter = th.nn.Parameter(circuits_logits)
    optimizer = th.optim.Adam([circuits_parameter], lr=lr)

    # Setup async logging if wandb_run is provided
    log_queue = None
    logger_thread = None
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "complexity_importance": complexity_importance,
                "complexity_power": complexity_power,
                "max_circuits": max_circuits,
                "num_epochs": num_epochs,
                "lr": lr,
            }
        )
        log_queue = queue.Queue()
        logger_thread = threading.Thread(
            target=_async_wandb_logger, args=(wandb_run, log_queue), daemon=True
        )
        logger_thread.start()

    for _ in tqdm(
        range(num_epochs), desc="Gradient descent", total=num_epochs, leave=False
    ):
        loss, faithfulness, complexity = circuit_loss(
            data, circuits_parameter, complexity_importance, complexity_power
        )
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        # Async logging - non-blocking
        if log_queue is not None:
            with suppress(queue.Full):
                log_queue.put_nowait(
                    {
                        "loss": loss,
                        "faithfulness": faithfulness,
                        "complexity": complexity,
                        "lr": lr,
                    }
                )

    # Clean up async logger
    if log_queue is not None:
        log_queue.put(None)  # Sentinel to stop the thread
    if logger_thread is not None:
        logger_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish

    return circuits_parameter, loss, faithfulness, complexity


@arguably.command()
def load_and_gradient_descent(
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
    max_circuits: int = 256,
    num_epochs: int = 1024,
    lr: float = 1e-2,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    data = load_activations(device=device)

    wandb_run = wandb.init(
        project="circuit-optimization",
        config={
            "complexity_importance": complexity_importance,
            "complexity_power": complexity_power,
            "max_circuits": max_circuits,
            "num_epochs": num_epochs,
            "lr": lr,
            "seed": seed,
        },
    )

    return gradient_descent(
        data,
        complexity_importance,
        complexity_power,
        max_circuits,
        num_epochs,
        device=device,
        wandb_run=wandb_run,
    )


@arguably.command()
def grid_search_gradient_descent(
    complexity_importances: list[float] | None = None,
    complexity_powers: list[float] | None = None,
    lrs: list[float] | None = None,
    max_circuits: int = 256,
    num_epochs: int = 2048,
    num_seeds: int = 3,
    device: str = "cuda",
) -> None:
    if complexity_importances is None:
        # complexity_importances = [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
        complexity_importances = [0, 0.001, 0.1, 0.5, 0.9, 0.99]
    if complexity_powers is None:
        # complexity_powers = [0.1, 0.2, 0.5, 1., 2., 3.]
        complexity_powers = [
            0.1,
            0.5,
            1.0,
            2.0,
        ]
    if lrs is None:
        lrs = [1e0, 1e-1, 1e-2, 1e-3, 1 - 4]

    data = load_activations(device=device)
    loss_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        device=device,
    )
    faithfulness_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        device=device,
    )
    complexity_landscape = th.empty(
        num_seeds,
        len(complexity_importances),
        len(complexity_powers),
        len(lrs),
        device=device,
    )

    for (
        (seed_idx, seed),
        (complexity_importance_idx, complexity_importance),
        (complexity_power_idx, complexity_power),
        (lr_idx, lr),
    ) in tqdm(
        product(
            enumerate(range(num_seeds)),
            enumerate(complexity_importances),
            enumerate(complexity_powers),
            enumerate(lrs),
        ),
        desc="Grid search",
        total=num_seeds
        * len(complexity_importances)
        * len(complexity_powers)
        * len(lrs),
    ):
        wandb_run = wandb.init(
            project="circuit-optimization",
            name=f"complexity_importance={complexity_importance};complexity_power={complexity_power};lr={lr}",
            config={
                "complexity_importance": complexity_importance,
                "complexity_power": complexity_power,
                "max_circuits": max_circuits,
                "num_epochs": num_epochs,
                "lr": lr,
                "seed": seed,
            },
        )

        circuits, loss, faithfulness, complexity = gradient_descent(
            data,
            complexity_importance,
            complexity_power,
            max_circuits,
            num_epochs,
            lr,
            seed=seed,
            device=device,
            wandb_run=wandb_run,
        )
        loss_landscape[
            seed_idx, complexity_importance_idx, complexity_power_idx, lr_idx
        ] = loss
        faithfulness_landscape[
            seed_idx, complexity_importance_idx, complexity_power_idx, lr_idx
        ] = faithfulness
        complexity_landscape[
            seed_idx, complexity_importance_idx, complexity_power_idx, lr_idx
        ] = complexity

    print(faithfulness_landscape)
    print(complexity_landscape)

    out = {
        "loss_landscape": loss_landscape,
        "faithfulness_landscape": faithfulness_landscape,
        "complexity_landscape": complexity_landscape,
        "complexity_importances": complexity_importances,
        "complexity_powers": complexity_powers,
        "lrs": lrs,
    }
    th.save(out, "loss_landscape.pt")


if __name__ == "__main__":
    # arguably.run()
    grid_search_gradient_descent()
