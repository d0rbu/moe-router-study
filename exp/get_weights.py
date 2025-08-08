import os

import arguably
import torch as th
from tqdm import tqdm

from core.model import MODELS
from exp import OUTPUT_DIR, WEIGHT_DIR

@arguably.command()
def get_weights(model_name: str = "olmoe", checkpoint_idx: int = -1) -> None:
    # Import here to avoid heavy import at module import time
    from nnterp import StandardizedTransformer
    
    model_config = MODELS.get(model_name, None)

    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    checkpoint = model_config.checkpoints[checkpoint_idx]

    os.makedirs(WEIGHT_DIR, exist_ok=True)

    model = StandardizedTransformer(
        model_config.hf_name, device_map="cpu", revision=str(checkpoint)
    )
    router_layers: list[int] = model.layers_with_routers
    num_layers = len(model.layers)
    num_experts = 0

    with th.no_grad():
        base_out = {
            "checkpoint_idx": checkpoint_idx,
            "num_tokens": checkpoint.num_tokens,
            "step": checkpoint.step,
            "topk": model.topk,
        }

        router_weights = {}
        for layer_idx in tqdm(
            router_layers,
            desc="Getting router weights",
            total=len(router_layers),
            leave=False,
        ):
            router_weights[layer_idx] = model.routers[layer_idx].weight.cpu()
            num_experts = router_weights[layer_idx].shape[0]

        router_out = {
            **base_out,
            "weights": router_weights,
        }
        th.save(router_out, os.path.join(WEIGHT_DIR, "router.pt"))

        down_proj_weights = {}
        for layer_idx in tqdm(
            range(num_layers),
            desc="Getting down projection weights",
            total=num_layers,
            leave=False,
        ):
            if num_experts <= 0:
                # not moe
                down_proj_weights[layer_idx] = model.mlps[layer_idx].down_proj.weight.cpu()
            else:
                expert_down_proj_weights = th.empty(num_experts, *model.mlps[layer_idx].experts[0].down_proj.weight.shape)
                for expert_idx in range(num_experts):
                    expert_down_proj_weights[expert_idx] = model.mlps[layer_idx].experts[expert_idx].down_proj.weight.cpu()
                down_proj_weights[layer_idx] = expert_down_proj_weights

        down_proj_out = {
            **base_out,
            "weights": down_proj_weights,
        }
        th.save(down_proj_out, os.path.join(WEIGHT_DIR, "down_proj.pt"))

        o_proj_weights = {}
        for layer_idx in tqdm(
            range(num_layers),
            desc="Getting output projection weights",
            total=num_layers,
            leave=False,
        ):
            o_proj_weights[layer_idx] = model.self_attn[layer_idx].out_proj.weight.cpu()

        o_proj_out = {
            **base_out,
            "weights": o_proj_weights,
        }
        th.save(o_proj_out, os.path.join(WEIGHT_DIR, "o_proj.pt"))


if __name__ == "__main__":
    arguably.run()
