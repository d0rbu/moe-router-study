"""Mock implementation of expert_importance for testing."""

import os
from itertools import product
from typing import Any

import torch as th
from unittest.mock import MagicMock

def mock_expert_importance(
    mock_transformer,
    model_name: str = "test_model",
    checkpoint_idx: int = 0,
    revision: str = None,
    output_dir: str = None,
) -> None:
    """Mock implementation of expert_importance for testing."""
    router_layers: list[int] = mock_transformer.layers_with_routers

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
            router_weight: th.Tensor = (
                mock_transformer.routers[base_layer_idx].weight.detach().cpu()
            )
            num_experts, hidden_size = router_weight.shape
            V = router_weight  # (E, D)

            for derived_layer_idx in router_layers:
                # Preload attention weights for this layer
                q_w: th.Tensor = (
                    mock_transformer.attentions[derived_layer_idx].q_proj.weight.detach().cpu()
                )
                k_w: th.Tensor = (
                    mock_transformer.attentions[derived_layer_idx].k_proj.weight.detach().cpu()
                )
                o_w: th.Tensor = (
                    mock_transformer.attentions[derived_layer_idx].o_proj.weight.detach().cpu()
                )

                # Vectorized readers shared per layer
                # q/k: (E, Dq) = (q_w @ V^T)^T, (k_w @ V^T)^T
                q_imp_all: th.Tensor = V @ q_w.T  # (E, Dq)
                k_imp_all: th.Tensor = V @ k_w.T  # (E, Dq)

                # Expert-specific readers: up/gate
                up_w_all: th.Tensor = th.stack(
                    [
                        mock_transformer.mlps[derived_layer_idx]
                        .experts[e]
                        .up_proj.weight.detach()
                        .cpu()
                        for e in range(num_experts)
                    ],
                    dim=0,
                )  # (E, Dmlp, D)
                gate_w_all: th.Tensor = th.stack(
                    [
                        mock_transformer.mlps[derived_layer_idx]
                        .experts[e]
                        .gate_proj.weight.detach()
                        .cpu()
                        for e in range(num_experts)
                    ],
                    dim=0,
                )  # (E, Dmlp, D)
                up_imp_all: th.Tensor = V @ up_w_all.transpose(0, -1)  # (E, Dmlp, E)
                gate_imp_all: th.Tensor = V @ gate_w_all.transpose(
                    0, -1
                )  # (E, Dmlp, E)

                # Vectorized writers
                down_w_all: th.Tensor = th.stack(
                    [
                        mock_transformer.mlps[derived_layer_idx]
                        .experts[e]
                        .down_proj.weight.detach()
                        .cpu()
                        for e in range(num_experts)
                    ],
                    dim=0,
                )  # (E, Dmlp, D)
                down_imp_all: th.Tensor = V @ down_w_all.transpose(
                    0, -1
                )  # (E, Dmlp, E)
                o_imp_all: th.Tensor = V @ o_w.T  # (E, Dq)

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
        outfile = os.path.join(output_dir, "all.pt")
        th.save(entries, outfile)

