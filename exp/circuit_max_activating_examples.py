import arguably
import torch as th

from exp.activations import load_activations


@arguably.command()
def get_circuit_activations(
    _circuits: th.Tensor,
    device: str = "cuda",
) -> None:
    load_activations(device=device)


if __name__ == "__main__":
    arguably.run()
