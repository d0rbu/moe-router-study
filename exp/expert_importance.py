import os
from typing import Any

import arguably
from nnterp import StandardizedTransformer
import torch as th

from core.model import MODELS
from exp import OUTPUT_DIR

EXPERT_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "expert_importance")


def _l2(x: th.Tensor) -> float:
    return float(th.linalg.vector_norm(x).item())


@arguably.command()
def expert_importance(
    model_name: str = "olmoe",
    checkpoint_idx: int = -1,
    device: str = "cpu",
) -> None:
    """Compute reader/writer importance vectors and scores for ALL experts across ALL router layers.

    For each router layer and each expert in that layer:
      - Takes the router vector for that expert (row of the router weight)
      - Readers (row-space): up_proj, gate_proj, q_proj, k_proj via y = W @ v
      - Writers (column-space): down_proj, o_proj via y = W.T @ v
      - Computes L2 norms of these importance vectors
      - Saves one .pt per layer under out/expert_importance/
    """
    model_config = MODELS.get(model_name, None)
    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    checkpoint = model_config.checkpoints[checkpoint_idx]

    os.makedirs(EXPERT_IMPORTANCE_DIR, exist_ok=True)

    model = StandardizedTransformer(
        model_config.hf_name,
        device_map=device,
        revision=str(checkpoint),
    )

    router_layers: list[int] = model.layers_with_routers

    with th.no_grad():
        for layer_idx in router_layers:
            # Expert directions V: rows of router weight
            router_weight: th.Tensor = model.routers[layer_idx].weight.detach().cpu()
            num_experts, hidden_size = router_weight.shape

            # Preload attention weights for this layer
            q_w: th.Tensor = model.self_attn[layer_idx].q_proj.weight.detach().cpu()
            k_w: th.Tensor = model.self_attn[layer_idx].k_proj.weight.detach().cpu()
            o_w: th.Tensor = model.self_attn[layer_idx].out_proj.weight.detach().cpu()

            # Flat list of entries for this layer
            entries: list[dict[str, Any]] = []

            for expert_idx in range(num_experts):
                v: th.Tensor = router_weight[expert_idx]  # (hidden_size,)

                # Readers: rows space dot = W @ v
                up_w: th.Tensor = (
                    model.mlps[layer_idx]
                    .experts[expert_idx]
                    .up_proj.weight.detach()
                    .cpu()
                )  # (d_mlp, hidden_size)
                up_imp: th.Tensor = up_w @ v

                gate_w: th.Tensor = (
                    model.mlps[layer_idx]
                    .experts[expert_idx]
                    .gate_proj.weight.detach()
                    .cpu()
                )  # (d_mlp, hidden_size)
                gate_imp: th.Tensor = gate_w @ v

                q_imp: th.Tensor = q_w @ v
                k_imp: th.Tensor = k_w @ v

                # Writers: column space dot = W.T @ v
                down_w: th.Tensor = (
                    model.mlps[layer_idx]
                    .experts[expert_idx]
                    .down_proj.weight.detach()
                    .cpu()
                )  # (hidden_size, d_mlp)
                down_imp: th.Tensor = down_w.T @ v
                o_imp: th.Tensor = o_w.T @ v

                # Append flat entries for each component
                entries.extend(
                    [
                        {
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "step": checkpoint.step,
                            "num_tokens": checkpoint.num_tokens,
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx,
                            "hidden_size": hidden_size,
                            "num_experts": num_experts,
                            "component": "mlp.up_proj",
                            "param_path": f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                            "role": "reader",
                            "importance_vector": up_imp,
                            "l2": _l2(up_imp),
                        },
                        {
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "step": checkpoint.step,
                            "num_tokens": checkpoint.num_tokens,
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx,
                            "hidden_size": hidden_size,
                            "num_experts": num_experts,
                            "component": "mlp.gate_proj",
                            "param_path": f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                            "role": "reader",
                            "importance_vector": gate_imp,
                            "l2": _l2(gate_imp),
                        },
                        {
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "step": checkpoint.step,
                            "num_tokens": checkpoint.num_tokens,
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx,
                            "hidden_size": hidden_size,
                            "num_experts": num_experts,
                            "component": "attn.q_proj",
                            "param_path": f"layers.{layer_idx}.self_attn.q_proj.weight",
                            "role": "reader",
                            "importance_vector": q_imp,
                            "l2": _l2(q_imp),
                        },
                        {
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "step": checkpoint.step,
                            "num_tokens": checkpoint.num_tokens,
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx,
                            "hidden_size": hidden_size,
                            "num_experts": num_experts,
                            "component": "attn.k_proj",
                            "param_path": f"layers.{layer_idx}.self_attn.k_proj.weight",
                            "role": "reader",
                            "importance_vector": k_imp,
                            "l2": _l2(k_imp),
                        },
                        {
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "step": checkpoint.step,
                            "num_tokens": checkpoint.num_tokens,
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx,
                            "hidden_size": hidden_size,
                            "num_experts": num_experts,
                            "component": "mlp.down_proj",
                            "param_path": f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                            "role": "writer",
                            "importance_vector": down_imp,
                            "l2": _l2(down_imp),
                        },
                        {
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "step": checkpoint.step,
                            "num_tokens": checkpoint.num_tokens,
                            "layer_idx": layer_idx,
                            "expert_idx": expert_idx,
                            "hidden_size": hidden_size,
                            "num_experts": num_experts,
                            "component": "attn.o_proj",
                            "param_path": f"layers.{layer_idx}.self_attn.o_proj.weight",
                            "role": "writer",
                            "importance_vector": o_imp,
                            "l2": _l2(o_imp),
                        },
                    ]
                )

            # Save flat list for this layer
            outfile = os.path.join(EXPERT_IMPORTANCE_DIR, f"layer{layer_idx}.pt")
            th.save(entries, outfile)


if __name__ == "__main__":
    arguably.run()
