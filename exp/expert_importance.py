from itertools import product
import os
from typing import Any

import arguably
from nnterp import StandardizedTransformer
import torch as th

from core.model import MODELS
from exp import OUTPUT_DIR

EXPERT_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "expert_importance")


@arguably.command()
def expert_importance(
    model_name: str = "olmoe",
    checkpoint_idx: int | None = None,
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

    if checkpoint_idx is None:
        revision = None
    else:
        revision = str(model_config.checkpoints[checkpoint_idx])

    os.makedirs(EXPERT_IMPORTANCE_DIR, exist_ok=True)

    model = StandardizedTransformer(
        model_config.hf_name,
        device_map=device,
        revision=revision,
    )

    router_layers: list[int] = model.layers_with_routers

    with th.no_grad():
        # Accumulate a single flat list across ALL layers and experts
        entries: list[dict[str, Any]] = []

        common_meta: dict[str, Any] = {
            "model_name": model_name,
            "checkpoint_idx": checkpoint_idx,
            "revision": revision,
        }

        for base_layer_idx in router_layers:
            # Expert directions V: rows of router weight
            router_weight: th.Tensor = model.routers[base_layer_idx].weight.detach().cpu()
            num_experts, hidden_size = router_weight.shape
            V = router_weight  # (E, D)

            for derived_layer_idx in router_layers:
                # Preload attention weights for this layer
                q_w: th.Tensor = model.attentions[derived_layer_idx].q_proj.weight.detach().cpu()
                k_w: th.Tensor = model.attentions[derived_layer_idx].k_proj.weight.detach().cpu()
                o_w: th.Tensor = model.attentions[derived_layer_idx].o_proj.weight.detach().cpu()

                # Vectorized readers shared per layer
                # q/k: (E, Dq) = (q_w @ V^T)^T, (k_w @ V^T)^T
                q_imp_all: th.Tensor = V @ q_w.T  # (E, Dq)
                k_imp_all: th.Tensor = V @ k_w.T  # (E, Dq)

                # Expert-specific readers: up/gate
                up_w_all: th.Tensor = th.stack(
                    [
                        model.mlps[derived_layer_idx].experts[e].up_proj.weight.detach().cpu()
                        for e in range(num_experts)
                    ],
                    dim=0,
                )  # (E, Dmlp, D)
                gate_w_all: th.Tensor = th.stack(
                    [
                        model.mlps[derived_layer_idx].experts[e].gate_proj.weight.detach().cpu()
                        for e in range(num_experts)
                    ],
                    dim=0,
                )  # (E, Dmlp, D)
                up_imp_all: th.Tensor = V @ up_w_all.transpose(0, -1)  # (E, Dmlp, E)
                gate_imp_all: th.Tensor = V @ gate_w_all.transpose(0, -1)  # (E, Dmlp, E)

                # Vectorized writers
                down_w_all: th.Tensor = th.stack(
                    [
                        model.mlps[derived_layer_idx].experts[e].down_proj.weight.detach().cpu()
                        for e in range(num_experts)
                    ],
                    dim=0,
                )  # (E, Dmlp, D)
                down_imp_all: th.Tensor = V @ down_w_all.transpose(0, -1)  # (E, Dmlp, E)
                o_imp_all: th.Tensor = V @ o_w.T  # (E, Dq)

                # Append entries for this layer
                layer_meta = {
                    "base_layer_idx": base_layer_idx,
                    "derived_layer_idx": derived_layer_idx,
                    "hidden_size": hidden_size,
                    "num_experts": num_experts,
                }
                # Iterate over all pairs of experts to save MoE elements
                for base_expert_idx, derived_expert_idx in product(range(num_experts), range(num_experts)):
                    expert_meta = {"base_expert_idx": base_expert_idx, "derived_expert_idx": derived_expert_idx}
                    base = {**common_meta, **layer_meta, **expert_meta}

                    # Readers
                    entries.append(
                        {
                            **base,
                            "component": "mlp.up_proj",
                            "base_expert_idx": base_expert_idx,
                            "derived_expert_idx": derived_expert_idx,
                            "base_param_path": f"layers.{base_layer_idx}.mlp.experts.{base_expert_idx}.up_proj.weight",
                            "derived_param_path": f"layers.{derived_layer_idx}.mlp.experts.{derived_expert_idx}.up_proj.weight",
                            "role": "reader",
                            "param_type": "moe",
                            "importance_vector": up_imp_all[base_expert_idx, :, derived_expert_idx],
                            "l2": float(th.linalg.vector_norm(up_imp_all[base_expert_idx, :, derived_expert_idx]).item()),
                        }
                    )
                    entries.append(
                        {
                            **base,
                            "component": "mlp.gate_proj",
                            "base_expert_idx": base_expert_idx,
                            "derived_expert_idx": derived_expert_idx,
                            "base_param_path": f"layers.{base_layer_idx}.mlp.experts.{base_expert_idx}.gate_proj.weight",
                            "derived_param_path": f"layers.{derived_layer_idx}.mlp.experts.{derived_expert_idx}.gate_proj.weight",
                            "role": "reader",
                            "param_type": "moe",
                            "importance_vector": gate_imp_all[base_expert_idx, :, derived_expert_idx],
                            "l2": float(th.linalg.vector_norm(gate_imp_all[base_expert_idx, :, derived_expert_idx]).item()),
                        }
                    )

                    # Writers
                    entries.append(
                        {
                            **base,
                            "component": "mlp.down_proj",
                            "base_expert_idx": base_expert_idx,
                            "derived_expert_idx": derived_expert_idx,
                            "base_param_path": f"layers.{base_layer_idx}.mlp.experts.{base_expert_idx}.down_proj.weight",
                            "derived_param_path": f"layers.{derived_layer_idx}.mlp.experts.{derived_expert_idx}.down_proj.weight",
                            "role": "writer",
                            "param_type": "moe",
                            "importance_vector": down_imp_all[base_expert_idx, :, derived_expert_idx],
                            "l2": float(th.linalg.vector_norm(down_imp_all[base_expert_idx, :, derived_expert_idx]).item()),
                        }
                    )

                # Iterate over all pairs of experts to save attention elements
                for expert_idx in range(num_experts):
                    expert_meta = {"base_expert_idx": expert_idx}
                    base = {**common_meta, **layer_meta, **expert_meta}

                    # Readers
                    entries.append(
                        {
                            **base,
                            "component": "attn.q_proj",
                            "base_expert_idx": expert_idx,
                            "base_param_path": f"layers.{base_layer_idx}.self_attn.q_proj.weight",
                            "role": "reader",
                            "param_type": "attn",
                            "importance_vector": q_imp_all[expert_idx],
                            "l2": float(th.linalg.vector_norm(q_imp_all[expert_idx]).item()),
                        }
                    )
                    entries.append(
                        {
                            **base,
                            "component": "attn.k_proj",
                            "base_expert_idx": base_expert_idx,
                            "base_param_path": f"layers.{base_layer_idx}.self_attn.k_proj.weight",
                            "role": "reader",
                            "param_type": "attn",
                            "importance_vector": k_imp_all[expert_idx],
                            "l2": float(th.linalg.vector_norm(k_imp_all[expert_idx]).item()),
                        }
                    )

                    # Writers
                    entries.append(
                        {
                            **base,
                            "component": "attn.o_proj",
                            "base_expert_idx": base_expert_idx,
                            "base_param_path": f"layers.{base_layer_idx}.self_attn.o_proj.weight",
                            "role": "writer",
                            "param_type": "attn",
                            "importance_vector": o_imp_all[expert_idx],
                            "l2": float(th.linalg.vector_norm(o_imp_all[expert_idx]).item()),
                        }
                    )

        # Save a single file containing ALL entries across layers/experts
        outfile = os.path.join(EXPERT_IMPORTANCE_DIR, "all.pt")
        th.save(entries, outfile)


if __name__ == "__main__":
    arguably.run()
