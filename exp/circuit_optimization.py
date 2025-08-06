from itertools import product

import arguably
import torch as th
from tqdm import tqdm

from exp.activations import load_activations
from exp.circuit_loss import circuit_loss, hard_circuit_score


def gradient_descent(
    data: th.Tensor,
    complexity_coefficient: float = 1.0,
    complexity_power: float = 1.0,
    max_circuits: int = 256,
    num_epochs: int = 1024,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor]:
    """
    Gradient descent for the optimal circuit.

    Args:
        data: The expert activations.
        complexity_coefficient: The complexity coefficient.
        complexity_power: The complexity power.
        max_circuits: The maximum number of circuits to search over.
        num_epochs: The number of epochs to run.
        lr: The learning rate.
        seed: The random seed.

    Returns:
        The optimal set of circuits and the loss.
    """

    assert data.ndim == 3, "Data must be of shape (B, L, E)"

    batch_size, num_layers, num_experts = data.shape

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    circuits_logits = th.randn(max_circuits, num_layers, num_experts, device=device)
    circuits_parameter = th.nn.Parameter(circuits_logits)
    optimizer = th.optim.Adam([circuits_parameter], lr=lr)
    log_every = 10

    for epoch_idx in tqdm(range(num_epochs), desc="Gradient descent", total=num_epochs):
        loss = circuit_loss(data, circuits_parameter, complexity_coefficient, complexity_power).sum()
        loss.backward()
        optimizer.step()

        if epoch_idx % log_every == 0:
            # boolean_circuits = th.sigmoid(circuits_parameter).round().bool()
            # score = hard_circuit_score(data, boolean_circuits, complexity_coefficient, complexity_power)
            # print(f"Epoch {epoch_idx}: loss={loss.item():.4f} score={score.item():.4f} complexity={boolean_circuits.sum().item():.4f}")
            circuits = th.sigmoid(circuits_parameter)
            print(f"Epoch {epoch_idx}: loss={loss.item():.4f} complexity={circuits.sum().item():.4f}")

        optimizer.zero_grad()

    return circuits_parameter, loss


@arguably.command()
def load_and_gradient_descent(
    complexity_coefficient: float = 1.0,
    complexity_power: float = 1.0,
    max_circuits: int = 256,
    num_epochs: int = 1024,
    device: str = "cuda",
) -> tuple[th.Tensor, th.Tensor]:
    data = load_activations(device=device)
    return gradient_descent(data, complexity_coefficient, complexity_power, max_circuits, num_epochs, device=device)


@arguably.command()
def grid_search_gradient_descent(
    complexity_coefficients: list[float] | None = None,
    complexity_powers: list[float] | None = None,
    lrs: list[float] | None = None,
    max_circuits: int = 256,
    num_epochs: int = 1024,
    num_seeds: int = 3,
    device: str = "cuda",
) -> None:
    if complexity_coefficients is None:
        complexity_coefficients = [0, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    if complexity_powers is None:
        complexity_powers = [0.1, 0.2, 0.5, 1., 2., 3.]
    if lrs is None:
        lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    data = load_activations(device=device)
    loss_landscape = th.empty(num_seeds, len(complexity_coefficients), len(complexity_powers), len(lrs), device=device)

    for (seed_idx, seed), (complexity_coefficient_idx, complexity_coefficient), (complexity_power_idx, complexity_power), (lr_idx, lr) in tqdm(product(enumerate(range(num_seeds)), enumerate(complexity_coefficients), enumerate(complexity_powers), enumerate(lrs)), desc="Grid search", total=num_seeds * len(complexity_coefficients) * len(complexity_powers) * len(lrs)):
        loss_landscape[seed_idx, complexity_coefficient_idx, complexity_power_idx, lr_idx] = gradient_descent(data, complexity_coefficient, complexity_power, max_circuits, num_epochs, lr, seed=seed, device=device)[1]

    print(loss_landscape)
    out = {
        "loss_landscape": loss_landscape,
        "complexity_coefficients": complexity_coefficients,
        "complexity_powers": complexity_powers,
        "lrs": lrs,
    }
    th.save(out, "loss_landscape.pt")


if __name__ == "__main__":
    # arguably.run()
    grid_search_gradient_descent()
