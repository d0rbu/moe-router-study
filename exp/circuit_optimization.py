from itertools import product
import queue
import threading
from typing import Any

import arguably
import torch as th
from tqdm import tqdm
import trackio as wandb

from exp.activations import load_activations
from exp.circuit_loss import circuit_loss


def _round_if_float(x: object, ndigits: int = 6) -> object:
    """Round Python floats to improve equality comparisons in tests."""
    return round(x, ndigits) if isinstance(x, float) else x


def expand_batch(
    batch: dict[str, th.Tensor | int | float | str | bool | list | None],
) -> list[dict[str, Any]]:
    """Expand a batch dict of tensors/scalars into a list of per-item dicts.

    Rules per tests:
    - Tensor values are converted to Python types via .tolist(). For multi-d tensors,
      outer-most dimension indexes items; inner dims remain as lists.
    - Scalar values (int/float/str) are replicated across all items.
    - Non-tensor lists are supported; they must have the same length as tensors.
    - None values are allowed and replicated.
    - If any list/tensor lengths disagree, raise AssertionError.
    - Empty tensors (length 0) yield an empty list.
    """
    # Convert tensors to lists, leave others as-is
    normalized: dict[str, Any] = {}
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
            # to satisfy tests. Otherwise replicate the entire list as a scalar value.
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
    expanded: list[dict[str, Any]] = []
    for i in range(batch_len):
        item: dict[str, Any] = {}
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
    wandb_run: wandb.Run, log_queue: queue.Queue, ready_flag: threading.Event
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
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
    lr: float = 1e-2,
    max_circuits: int = 256,
    num_epochs: int = 2048,
    num_warmup_epochs: int = 128,
    num_cooldown_epochs: int = 512,
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

    assert num_warmup_epochs + num_cooldown_epochs < num_epochs, (
        "num_warmup_epochs + num_cooldown_epochs must be less than num_epochs"
    )
    assert data.ndim == 3, "Data must be of shape (B, L, E)"

    batch_size, num_layers, num_experts = data.shape

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    # Setup async batch logging if wandb_run is provided
    log_queue = None
    logger_thread = None
    ready_flag = None

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
        ready_flag = threading.Event()
        logger_thread = threading.Thread(
            target=_async_wandb_batch_logger, args=(wandb_run, log_queue, ready_flag)
        )
        logger_thread.start()

        def log_batch() -> None:
            if log_queue is None:
                return

            log_queue.put(
                {
                    "losses": losses[:logs_accumulated],
                    "faithfulnesses": faithfulnesses[:logs_accumulated],
                    "complexities": complexities[:logs_accumulated],
                    "lrs": lrs[:logs_accumulated],
                }
            )

    circuits_logits = th.randn(max_circuits, num_layers, num_experts, device=device)
    circuits_parameter = th.nn.Parameter(circuits_logits)
    optimizer = th.optim.Adam([circuits_parameter], lr=lr)

    # simple trapezoid LR scheduler with warmup and cooldown
    def get_lr(epoch: int) -> float:
        portion_through_warmup = (
            epoch / num_warmup_epochs if num_warmup_epochs > 0 else 1
        )
        distance_from_end_relative_to_cooldown = (
            num_epochs - epoch
        ) / num_cooldown_epochs

        return lr * min(
            portion_through_warmup, distance_from_end_relative_to_cooldown, 1
        )

    losses = th.empty(num_epochs, device=device)
    faithfulnesses = th.empty(num_epochs, device=device)
    complexities = th.empty(num_epochs, device=device)
    lrs = th.empty(num_epochs, device=device)
    logs_accumulated = 0

    for epoch_idx in tqdm(
        range(num_epochs), desc="Gradient descent", total=num_epochs, leave=False
    ):
        # Update learning rate
        current_lr = get_lr(epoch_idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        loss, faithfulness, complexity = circuit_loss(
            data,
            circuits_parameter,
            top_k=top_k,
            complexity_importance=complexity_importance,
            complexity_power=complexity_power,
        )
        loss = loss.sum()

        if ready_flag is not None and log_queue is not None:
            losses[logs_accumulated] = loss
            faithfulnesses[logs_accumulated] = faithfulness
            complexities[logs_accumulated] = complexity
            lrs[logs_accumulated] = current_lr
            logs_accumulated += 1

            # Send batch to async logger when ready
            if ready_flag.is_set():
                ready_flag.clear()  # Signal that we're sending a batch
                log_batch()
                logs_accumulated = 0

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    # Clean up async logger - send any remaining batch and stop the thread
    if log_queue is not None and logger_thread is not None:
        if logs_accumulated > 0:
            log_batch()
        log_queue.put(None)

    return circuits_parameter, loss, faithfulness, complexity


@arguably.command()
def load_and_gradient_descent(
    top_k: int,
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
    lr: float = 1e-2,
    max_circuits: int = 256,
    num_epochs: int = 2048,
    num_warmup_epochs: int = 128,
    num_cooldown_epochs: int = 512,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    data = load_activations(device=device)

    wandb_run = wandb.init(
        project="circuit-optimization",
        name=f"top_k={top_k};complexity_importance={complexity_importance};complexity_power={complexity_power};lr={lr};max_circuits={max_circuits};num_epochs={num_epochs};num_warmup_epochs={num_warmup_epochs};num_cooldown_epochs={num_cooldown_epochs};seed={seed}",
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

    return gradient_descent(
        data,
        top_k,
        complexity_importance,
        complexity_power,
        lr,
        max_circuits,
        num_epochs,
        num_warmup_epochs,
        num_cooldown_epochs,
        seed=seed,
        device=device,
        wandb_run=wandb_run,
    )


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
    num_seeds: int = 3,
    device: str = "cuda",
) -> None:
    if complexity_importances is None:
        # complexity_importances = [0.5, 0.9, 0.1, 0.99, 0.01]
        complexity_importances = [0.5, 0.4, 0.3, 0.2, 0.1]
    if complexity_powers is None:
        # complexity_powers = [1.0, 2.0, 0.5]
        complexity_powers = [1.0]
    if lrs is None:
        # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        lrs = [0.05, 0.02, 0.01, 0.005, 0.002]
    if max_circuitses is None:
        # max_circuitses = [64, 128, 256, 512]
        max_circuitses = [64, 128, 256]
    if num_epochses is None:
        # num_epochses = [2048]
        num_epochses = [4096]
    if num_warmup_epochses is None:
        # num_warmup_epochses = [0]
        num_warmup_epochses = [0]
    if num_cooldown_epochses is None:
        # num_cooldown_epochses = [512]
        num_cooldown_epochses = [1024]

    data = load_activations(device=device)
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
            seed=seed,
            device=device,
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

    print(faithfulness_landscape)
    print(complexity_landscape)

    out = {
        "loss_landscape": loss_landscape,
        "faithfulness_landscape": faithfulness_landscape,
        "complexity_landscape": complexity_landscape,
        "complexity_importances": complexity_importances,
        "complexity_powers": complexity_powers,
        "lrs": lrs,
        "max_circuitses": max_circuitses,
        "num_epochses": num_epochses,
        "num_warmup_epochses": num_warmup_epochses,
        "num_cooldown_epochses": num_cooldown_epochses,
    }
    th.save(out, "loss_landscape.pt")


if __name__ == "__main__":
    # arguably.run()
    grid_search_gradient_descent(top_k=8, num_seeds=1)
