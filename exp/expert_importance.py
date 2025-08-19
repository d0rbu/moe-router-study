from itertools import product
import os
from typing import Any, cast

import arguably
from nnterp import StandardizedTransformer
import torch as th
from torch import Tensor
from tqdm import tqdm

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

        for base_layer_idx in tqdm(router_layers, desc="Base layers", leave=False):
            # Expert directions V: rows of router weight
            router_weight = cast("Tensor", model.routers[base_layer_idx].weight)
            router_weight = router_weight.detach().cpu()
            num_experts, hidden_size = router_weight.shape
            V = router_weight  # (E, D)

            for derived_layer_idx in tqdm(router_layers, desc="Derived layers", leave=False):
                # Preload attention weights for this layer
                q_w = cast("Tensor", model.attentions[derived_layer_idx].q_proj.weight)
                q_w = q_w.detach().cpu()

                k_w = cast("Tensor", model.attentions[derived_layer_idx].k_proj.weight)
                k_w = k_w.detach().cpu()

                o_w = cast("Tensor", model.attentions[derived_layer_idx].o_proj.weight)
                o_w = o_w.detach().cpu()

                # Vectorized readers shared per layer
                # q/k: (E, Dq) = (q_w @ V^T)^T, (k_w @ V^T)^T
                q_imp_all: th.Tensor = V @ q_w.T  # (E, Dq)
                k_imp_all: th.Tensor = V @ k_w.T  # (E, Dq)

                # Vectorized writers shared per layer
                o_imp_all: th.Tensor = V @ o_w  # (E, Dv)

                # Expert-specific readers: up/gate
                # Handle experts as a list to avoid subscripting issues
                experts = model.mlps[derived_layer_idx].experts

                up_weights = []
                gate_weights = []
                down_weights = []

                for e in range(num_experts):
                    # Handle both list-like and dict-like experts
                    if isinstance(experts, list):
                        expert = experts[e]
                    else:
                        # Try to access as an attribute or dictionary
                        try:
                            expert = getattr(experts, str(e))
                        except AttributeError:
                            expert = experts[str(e)]

                    up_w = cast("Tensor", expert.up_proj.weight).detach().cpu()
                    gate_w = cast("Tensor", expert.gate_proj.weight).detach().cpu()
                    down_w = cast("Tensor", expert.down_proj.weight.T).detach().cpu()

                    up_weights.append(up_w)
                    gate_weights.append(gate_w)
                    down_weights.append(down_w)

                up_w_all: th.Tensor = th.stack(up_weights, dim=0)  # (E, Dmlp, D)
                gate_w_all: th.Tensor = th.stack(gate_weights, dim=0)  # (E, Dmlp, D)
                down_w_all: th.Tensor = th.stack(down_weights, dim=0)  # (E, Dmlp, D)

                up_imp_all: th.Tensor = (V @ up_w_all.transpose(-1, -2)).transpose(
                    0, 1
                )  # (E_source, E_target, Dmlp)
                gate_imp_all: th.Tensor = (V @ gate_w_all.transpose(-1, -2)).transpose(
                    0, 1
                )  # (E_source, E_target, Dmlp)
                down_imp_all: th.Tensor = (V @ down_w_all.transpose(-1, -2)).transpose(
                    0, 1
                )  # (E_source, E_target, Dmlp)

                # Append entries for this layer
                layer_meta = {
                    "base_layer_idx": base_layer_idx,
                    "derived_layer_idx": derived_layer_idx,
                    "hidden_size": hidden_size,
                    "num_experts": num_experts,
                }
                # Iterate over all pairs of experts to save MoE elements
                for base_expert_idx, derived_expert_idx in product(
                    range(num_experts), range(num_experts)
                ):
                    expert_meta = {
                        "base_expert_idx": base_expert_idx,
                        "derived_expert_idx": derived_expert_idx,
                    }
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
                            "importance_vector": up_imp_all[
                                base_expert_idx, :, derived_expert_idx
                            ],
                            "l2": float(
                                th.linalg.vector_norm(
                                    up_imp_all[base_expert_idx, :, derived_expert_idx]
                                ).item()
                            ),
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
                            "importance_vector": gate_imp_all[
                                base_expert_idx, :, derived_expert_idx
                            ],
                            "l2": float(
                                th.linalg.vector_norm(
                                    gate_imp_all[base_expert_idx, :, derived_expert_idx]
                                ).item()
                            ),
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
                            "importance_vector": down_imp_all[
                                base_expert_idx, :, derived_expert_idx
                            ],
                            "l2": float(
                                th.linalg.vector_norm(
                                    down_imp_all[base_expert_idx, :, derived_expert_idx]
                                ).item()
                            ),
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
                            "l2": float(
                                th.linalg.vector_norm(q_imp_all[expert_idx]).item()
                            ),
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
                            "l2": float(
                                th.linalg.vector_norm(k_imp_all[expert_idx]).item()
                            ),
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
                            "l2": float(
                                th.linalg.vector_norm(o_imp_all[expert_idx]).item()
                            ),
                        }
                    )

        # Save a single file containing ALL entries across layers/experts
        outfile = os.path.join(EXPERT_IMPORTANCE_DIR, "all.pt")
        th.save(entries, outfile)


if __name__ == "__main__":
    arguably.run()
