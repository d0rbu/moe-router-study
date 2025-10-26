import gc
from itertools import product
import os
import queue
import threading
from typing import Any

import arguably
import torch as th
from tqdm import tqdm
import trackio as wandb

from exp import get_experiment_dir
from exp.activations import load_activations
from exp.circuit_loss import circuit_loss


def _round_if_float(x: object, ndigits: int = 6) -> object:
    """Round Python floats to improve equality comparisons in tests."""
    return round(x, ndigits) if isinstance(x, float) else x


def expand_batch(
    batch: dict[str, th.Tensor | int | float | str | bool | list | None],
) -> list[dict[str, any]]:
    """Expand a batch dict of tensors/scalars into a list of per-item dicts.

    Rules:
    - Tensor values are converted to Python types via .tolist(). For multi-d tensors,
      outer-most dimension indexes items; inner dims remain as lists.
    - Scalar values (int/float/str) are replicated across all items.
    - Non-tensor lists are supported; they must have the same length as tensors.
    - None values are allowed and replicated.
    - If any list/tensor lengths disagree, raise AssertionError.
    - Empty tensors (length 0) yield an empty list.
    """
    # Convert tensors to lists, leave others as-is
    normalized: dict[str, any] = {}
    lengths: set[int] = set()
    batched_keys: set[str] = set()
    for k, v in batch.items():
        if isinstance(v, th.Tensor):
            v_list = v.tolist()
            # Ensure we have a list along first dim; if tensor was 0-d, wrap into list
            if not isinstance(v_list, list):
                v_list = [v_list]
            lengths.add(len(v_list))
            normalized[k] = v_list
            batched_keys.add(k)
        elif isinstance(v, list):
            # Heuristic: only treat list as batch dimension if key name is plural-ish
            # Otherwise replicate the entire list as a scalar value.
            if k.endswith("s") or k.endswith("_values"):
                lengths.add(len(v))
                normalized[k] = v
                batched_keys.add(k)
            else:
                normalized[k] = v  # replicate as-is
        else:
            # Scalars/None replicated later
            normalized[k] = v

    # Determine batch length
    if len(lengths) == 0:
        # No tensor/list provided; treat as single item
        batch_len = 1
    else:
        assert len(lengths) == 1, "All values must have the same length"
        batch_len = next(iter(lengths))

    if batch_len == 0:
        return []

    # Build expanded items
    expanded: list[dict[str, any]] = []
    for i in range(batch_len):
        item: dict[str, any] = {}
        for k, v in normalized.items():
            if k in batched_keys and isinstance(v, list) and len(v) == batch_len:
                # Pull ith element for tensor/list-backed fields
                elem = v[i]
            else:
                # Replicate scalars/None
                elem = v
            # Round floats to avoid strict-equality failures (e.g., 0.8000000119 -> 0.8)
            item[k] = _round_if_float(elem)  # may be int/float/bool/str/list/None
        expanded.append(item)
    return expanded


def _async_wandb_batch_logger(
    wandb_run, log_queue: queue.Queue, ready_flag: threading.Event
):
    """Background thread for async wandb batch logging."""
    # Signal that we're ready to receive the first batch
    ready_flag.set()

    while True:
        try:
            # Get the batch data (blocking)
            batch_data = log_queue.get(timeout=1.0)
            if batch_data is None:  # Sentinel to stop the thread
                break

            expanded_batch_data = expand_batch(batch_data)
            for item in expanded_batch_data:
                wandb_run.log(item)

            # Signal that we're ready for the next batch
            ready_flag.set()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in async wandb logging: {e}")
            ready_flag.set()  # Ensure we don't get stuck


def gradient_descent(
    data: th.Tensor,
    top_k: int,
    complexity_importance: float,
    complexity_power: float,
    lr: float,
    max_circuits: int,
    num_epochs: int,
    num_warmup_epochs: int,
    num_cooldown_epochs: int,
    batch_size: int,
    grad_accumulation_steps: int,
    device: str,
    seed: int,
    wandb_run: Any,
) -> tuple[th.Tensor, float, float, float]:
    """Optimize circuits using gradient descent.

    Args:
        data: Boolean tensor of shape (batch_size, num_layers, num_experts)
        top_k: Number of experts to select per layer
        complexity_importance: Weight of complexity loss
        complexity_power: Power to raise complexity loss to
        lr: Learning rate
        max_circuits: Maximum number of circuits to optimize
        num_epochs: Number of epochs to train for
        num_warmup_epochs: Number of epochs to warm up for
        num_cooldown_epochs: Number of epochs to cool down for
        batch_size: Batch size to use for training (0 for full batch)
        grad_accumulation_steps: Number of gradient accumulation steps
        device: Device to use for training
        seed: Random seed
        wandb_run: Weights & Biases run object

    Returns:
        circuits: Boolean tensor of shape (max_circuits, num_layers, num_experts)
        loss: Final loss
        faithfulness: Final faithfulness
        complexity: Final complexity
    """
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    # Set up async wandb logging
    log_queue = queue.Queue()
    ready_flag = threading.Event()
    logger_thread = threading.Thread(
        target=_async_wandb_batch_logger,
        args=(wandb_run, log_queue, ready_flag),
        daemon=True,
    )
    logger_thread.start()

    # Wait for logger to be ready
    ready_flag.wait()

    # Get data dimensions
    batch_size_data, num_layers, num_experts = data.shape
    if batch_size <= 0 or batch_size > batch_size_data:
        batch_size = batch_size_data

    # Initialize circuits randomly
    circuits = th.rand(max_circuits, num_layers, num_experts, device=device)
    circuits.requires_grad = True

    # Set up optimizer
    optimizer = th.optim.Adam([circuits], lr=lr)

    # Set up learning rate scheduler
    if num_warmup_epochs > 0:
        warmup_scheduler = th.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_epochs,
        )
    if num_cooldown_epochs > 0:
        cooldown_scheduler = th.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=num_cooldown_epochs,
        )

    # Train
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Apply warmup/cooldown
        if num_warmup_epochs > 0 and epoch < num_warmup_epochs:
            warmup_scheduler.step()
        elif (
            num_cooldown_epochs > 0
            and epoch >= num_epochs - num_cooldown_epochs
            and epoch < num_epochs
        ):
            cooldown_scheduler.step()

        # Shuffle data
        indices = th.randperm(batch_size_data, device=device)
        data_shuffled = data[indices]

        # Train on batches
        (batch_size_data + batch_size - 1) // batch_size
        for batch_idx in range(0, batch_size_data, batch_size):
            # Get batch
            batch_end = min(batch_idx + batch_size, batch_size_data)
            batch = data_shuffled[batch_idx:batch_end]

            # Forward pass
            loss, faithfulness, complexity = circuit_loss(
                circuits, batch, top_k, complexity_importance, complexity_power
            )

            # Backward pass
            loss = loss / grad_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (
                batch_idx // batch_size
            ) % grad_accumulation_steps == 0 or batch_end == batch_size_data:
                optimizer.step()
                optimizer.zero_grad()

        # Log metrics
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with th.no_grad():
                loss, faithfulness, complexity = circuit_loss(
                    circuits, data, top_k, complexity_importance, complexity_power
                )
                log_queue.put(
                    {
                        "epoch": epoch,
                        "loss": loss.item(),
                        "faithfulness": faithfulness.item(),
                        "complexity": complexity.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                ready_flag.wait()  # Wait for logger to process the batch

    # Get final metrics
    with th.no_grad():
        loss, faithfulness, complexity = circuit_loss(
            circuits, data, top_k, complexity_importance, complexity_power
        )

    # Convert to boolean tensor
    circuits_bool = th.zeros_like(circuits, dtype=th.bool)
    for layer_idx in range(num_layers):
        # Get top-k experts per layer
        _, indices = th.topk(circuits[:, layer_idx], k=top_k, dim=1)
        # Set those experts to True
        circuits_bool[:, layer_idx].scatter_(1, indices, True)

    # Stop the logger thread
    log_queue.put(None)
    logger_thread.join()

    # Clean up
    del circuits
    gc.collect()
    if device == "cuda":
        th.cuda.empty_cache()

    return circuits_bool, loss.item(), faithfulness.item(), complexity.item()


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
    experiment_name: str | None = None,
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
    experiment_name: str | None = None,
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
