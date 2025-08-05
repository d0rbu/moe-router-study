import os

import arguably
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

from core.model import MODELS
from exp import OUTPUT_DIR

ROUTER_WEIGHT_DIR = os.path.join(OUTPUT_DIR, "router_weights")


@arguably.command()
def get_router_weights(model_name: str = "olmoe", checkpoint_idx: int = -1) -> None:
    model_config = MODELS.get(model_name, None)

    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    checkpoint = model_config.checkpoints[checkpoint_idx]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROUTER_WEIGHT_DIR, exist_ok=True)

    model = StandardizedTransformer(
        model_config.hf_name, device_map="cpu", revision=str(checkpoint)
    )
    router_layers: list[int] = model.layers_with_routers

    with th.no_grad():
        out = {
            "checkpoint_idx": checkpoint_idx,
            "num_tokens": checkpoint.num_tokens,
            "step": checkpoint.step,
            "routers": {},
        }

        for layer_idx in tqdm(
            router_layers,
            desc="Getting router weights",
            total=len(router_layers),
            leave=False,
        ):
            out["routers"][layer_idx] = model.routers[layer_idx].weight.cpu()

        th.save(out, os.path.join(ROUTER_WEIGHT_DIR, f"{checkpoint_idx}.pt"))


if __name__ == "__main__":
    get_router_weights()
