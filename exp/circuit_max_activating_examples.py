import arguably
import torch as th

from exp.activations import load_activations


@arguably.command()
def get_circuit_activations(
    circuits: th.Tensor,  # noqa: ARG001 - entrypoint signature; may be used later
    device: str = "cuda",
) -> None:
    load_activations(device=device)


if __name__ == "__main__":
    arguably.run()
