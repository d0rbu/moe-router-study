import os
from typing import Any

import arguably
from nnterp import StandardizedTransformer
import torch as th

from core.model import MODELS
from exp import OUTPUT_DIR

EXPERT_CIRCUIT_DIR = os.path.join(OUTPUT_DIR, "expert_circuit")


def _l2(x: th.Tensor) -> float:
    return float(th.linalg.vector_norm(x).item())


@arguably.command()
def expert_circuit(
    model_name: str = "olmoe",
    checkpoint_idx: int = -1,
    layer_idx: int = 0,
    expert_idx: int = 0,
    device: str = "cpu",
    outfile: str | None = None,
) -> None:
    """Compute reader/writer importance vectors and scores for a specific expert.

    For a given layer and expert index, this command:
      - Takes the router vector for that expert (row of the router weight)
      - Applies reader matrices (rows): up_proj, gate_proj, q_proj, k_proj via y = W @ v
      - Applies writer matrices (columnspace): down_proj, o_proj via y = W.T @ v
      - Computes L2 norms of these importance vectors
      - Saves results to disk for later visualization
    """
    model_config = MODELS.get(model_name, None)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    # Resolve checkpoint and instantiate model similar to other exp scripts
    checkpoint = model_config.checkpoints[checkpoint_idx]

    os.makedirs(EXPERT_CIRCUIT_DIR, exist_ok=True)

    # Keep everything on CPU by default to avoid OOM; user can change device if needed
    model = StandardizedTransformer(
        model_config.hf_name,
        device_map=device,
        revision=str(checkpoint),
    )

    router_layers: list[int] = model.layers_with_routers
    if layer_idx not in router_layers:
        raise ValueError(
            f"Layer {layer_idx} does not have a router. Router layers: {router_layers}"
        )

    with th.no_grad():
        # Expert direction v: the expert row in the router weight
        router_weight: th.Tensor = model.routers[layer_idx].weight.detach().cpu()
        num_experts, hidden_size = router_weight.shape
        if expert_idx < 0 or expert_idx >= num_experts:
            raise IndexError(
                f"expert_idx {expert_idx} out of range for layer {layer_idx} (num_experts={num_experts})"
            )
        v: th.Tensor = router_weight[expert_idx]  # (hidden_size,)

        # Collect importance vectors and scores
        results: dict[str, dict[str, Any]] = {}

        # Readers: rows space dot = W @ v
        # MLP expert readers
        up_w: th.Tensor = (
            model.mlps[layer_idx].experts[expert_idx].up_proj.weight.detach().cpu()
        )  # (d_mlp, hidden_size)
        up_imp: th.Tensor = up_w @ v  # (d_mlp,)
        results[f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = {
            "role": "reader",
            "importance_vector": up_imp,
            "l2": _l2(up_imp),
        }

        gate_w: th.Tensor = (
            model.mlps[layer_idx].experts[expert_idx].gate_proj.weight.detach().cpu()
        )  # (d_mlp, hidden_size)
        gate_imp: th.Tensor = gate_w @ v  # (d_mlp,)
        results[f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = {
            "role": "reader",
            "importance_vector": gate_imp,
            "l2": _l2(gate_imp),
        }

        # Attention readers
        q_w: th.Tensor = (
            model.self_attn[layer_idx].q_proj.weight.detach().cpu()
        )  # (d_qkv, hidden_size)
        q_imp: th.Tensor = q_w @ v  # (d_qkv,)
        results[f"layers.{layer_idx}.self_attn.q_proj.weight"] = {
            "role": "reader",
            "importance_vector": q_imp,
            "l2": _l2(q_imp),
        }

        k_w: th.Tensor = (
            model.self_attn[layer_idx].k_proj.weight.detach().cpu()
        )  # (d_qkv, hidden_size)
        k_imp: th.Tensor = k_w @ v  # (d_qkv,)
        results[f"layers.{layer_idx}.self_attn.k_proj.weight"] = {
            "role": "reader",
            "importance_vector": k_imp,
            "l2": _l2(k_imp),
        }

        # Writers: column space dot = W.T @ v
        down_w: th.Tensor = (
            model.mlps[layer_idx].experts[expert_idx].down_proj.weight.detach().cpu()
        )  # (hidden_size, d_mlp)
        down_imp: th.Tensor = down_w.T @ v  # (d_mlp,)
        results[f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = {
            "role": "writer",
            "importance_vector": down_imp,
            "l2": _l2(down_imp),
        }

        o_w: th.Tensor = (
            model.self_attn[layer_idx].out_proj.weight.detach().cpu()
        )  # (hidden_size, hidden_size)
        o_imp: th.Tensor = o_w.T @ v  # (hidden_size,)
        results[f"layers.{layer_idx}.self_attn.o_proj.weight"] = {
            "role": "writer",
            "importance_vector": o_imp,
            "l2": _l2(o_imp),
        }

        # Build output payload
        base_out: dict[str, Any] = {
            "model_name": model_name,
            "checkpoint_idx": checkpoint_idx,
            "num_tokens": checkpoint.num_tokens,
            "step": checkpoint.step,
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "hidden_size": hidden_size,
            "vector": v,  # router expert vector used
            "results": results,
        }

        # Determine output file path
        if outfile is None:
            outfile = os.path.join(
                EXPERT_CIRCUIT_DIR, f"layer{layer_idx}_expert{expert_idx}.pt"
            )

        th.save(base_out, outfile)


if __name__ == "__main__":
    arguably.run()
