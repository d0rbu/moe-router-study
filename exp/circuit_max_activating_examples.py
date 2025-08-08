from itertools import product
import queue
import threading

import arguably
import torch as th
from tqdm import tqdm

from exp.activations import load_activations


@arguably.command()
def get_circuit_activations(
    circuits: th.Tensor,
    device: str = "cuda",
) -> None:
    activation_data = load_activations(device=device)


if __name__ == "__main__":
    arguably.run()
