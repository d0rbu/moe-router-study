"""Expert importance calculation."""

import os
from typing import Any

from nnterp import StandardizedTransformer
import torch as th

from core.model import MODELS
from exp import OUTPUT_DIR

# Directory for expert importance data
EXPERT_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "expert_importance")


def calculate_expert_importance(
    activated_experts: th.Tensor,
    labels: th.Tensor,
) -> th.Tensor:
    """Calculate the importance of each expert for predicting labels.

    Args:
        activated_experts: Boolean tensor of shape (batch_size, num_layers, num_experts)
        labels: Labels tensor of shape (batch_size,)

    Returns:
        Importance tensor of shape (num_layers, num_experts)
    """
    # Convert to float
    activated_experts = activated_experts.float()
    labels = labels.float()

    # Get dimensions
    batch_size, num_layers, num_experts = activated_experts.shape

    # Calculate correlation between expert activation and labels
    importance = th.zeros(num_layers, num_experts)
    for layer in range(num_layers):
        for expert in range(num_experts):
            # Get expert activations
            expert_activations = activated_experts[:, layer, expert]

            # Calculate correlation
            correlation = th.corrcoef(th.stack([expert_activations, labels]))[0, 1]
            importance[layer, expert] = correlation

    return importance


def expert_importance(
    model_name: str,
    checkpoint_idx: int = 0,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Calculate expert importance for a model.

    Args:
        model_name: Name of the model
        checkpoint_idx: Index of the checkpoint to use
        device: Device to use

    Returns:
        List of dictionaries with expert importance data
    """
    # Check if model exists
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found")

    # Get model config
    model_config = MODELS[model_name]

    # Get checkpoint revision
    checkpoints = model_config.checkpoints
    revision = checkpoints[checkpoint_idx]

    # Load model
    model = StandardizedTransformer(
        model_config.hf_name,
        revision=revision,
        device=device,
    )

    # Get layers with routers
    layers_with_routers = model.layers_with_routers

    # Initialize entries list
    entries = []

    # Process each layer with routers
    for base_layer_idx in layers_with_routers:
        # Get router weights
        router = model.routers[base_layer_idx]
        router_weights = router.weight
        num_experts = router_weights.shape[0]

        # Process each expert
        for base_expert_idx in range(num_experts):
            # Process each derived layer
            for derived_layer_idx in layers_with_routers:
                # Process attention components
                attention = model.attentions[derived_layer_idx]

                # Q projection (reader)
                q_proj = attention.q_proj
                q_weight = q_proj.weight
                for i in range(q_weight.shape[0]):
                    importance_vector = q_weight[i]
                    entries.append({
                        "model_name": model_name,
                        "checkpoint_idx": checkpoint_idx,
                        "revision": revision,
                        "base_layer_idx": base_layer_idx,
                        "base_expert_idx": base_expert_idx,
                        "derived_layer_idx": derived_layer_idx,
                        "component": "attn.q_proj",
                        "role": "reader",
                        "param_type": "attn",
                        "importance_vector": importance_vector,
                        "l2": float(th.norm(importance_vector).item()),
                        "base_param_path": f"layers.{base_layer_idx}.router.weight",
                    })

                # K projection (reader)
                k_proj = attention.k_proj
                k_weight = k_proj.weight
                for i in range(k_weight.shape[0]):
                    importance_vector = k_weight[i]
                    entries.append({
                        "model_name": model_name,
                        "checkpoint_idx": checkpoint_idx,
                        "revision": revision,
                        "base_layer_idx": base_layer_idx,
                        "base_expert_idx": base_expert_idx,
                        "derived_layer_idx": derived_layer_idx,
                        "component": "attn.k_proj",
                        "role": "reader",
                        "param_type": "attn",
                        "importance_vector": importance_vector,
                        "l2": float(th.norm(importance_vector).item()),
                        "base_param_path": f"layers.{base_layer_idx}.router.weight",
                    })

                # O projection (writer)
                o_proj = attention.o_proj
                o_weight = o_proj.weight
                for i in range(o_weight.shape[0]):
                    importance_vector = o_weight[i]
                    entries.append({
                        "model_name": model_name,
                        "checkpoint_idx": checkpoint_idx,
                        "revision": revision,
                        "base_layer_idx": base_layer_idx,
                        "base_expert_idx": base_expert_idx,
                        "derived_layer_idx": derived_layer_idx,
                        "component": "attn.o_proj",
                        "role": "writer",
                        "param_type": "attn",
                        "importance_vector": importance_vector,
                        "l2": float(th.norm(importance_vector).item()),
                        "base_param_path": f"layers.{base_layer_idx}.router.weight",
                    })

                # Process MLP components
                mlp = model.mlps[derived_layer_idx]

                # Process each derived expert
                for derived_expert_idx, expert in enumerate(mlp.experts):
                    # Up projection (reader)
                    up_proj = expert.up_proj
                    up_weight = up_proj.weight
                    for i in range(up_weight.shape[0]):
                        importance_vector = up_weight[i]
                        entries.append({
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "revision": revision,
                            "base_layer_idx": base_layer_idx,
                            "base_expert_idx": base_expert_idx,
                            "derived_layer_idx": derived_layer_idx,
                            "derived_expert_idx": derived_expert_idx,
                            "component": "mlp.up_proj",
                            "role": "reader",
                            "param_type": "moe",
                            "importance_vector": importance_vector,
                            "l2": float(th.norm(importance_vector).item()),
                            "base_param_path": f"layers.{base_layer_idx}.router.weight",
                            "derived_param_path": f"layers.{derived_layer_idx}.mlp.experts.{derived_expert_idx}.up_proj.weight",
                        })

                    # Gate projection (reader)
                    gate_proj = expert.gate_proj
                    gate_weight = gate_proj.weight
                    for i in range(gate_weight.shape[0]):
                        importance_vector = gate_weight[i]
                        entries.append({
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "revision": revision,
                            "base_layer_idx": base_layer_idx,
                            "base_expert_idx": base_expert_idx,
                            "derived_layer_idx": derived_layer_idx,
                            "derived_expert_idx": derived_expert_idx,
                            "component": "mlp.gate_proj",
                            "role": "reader",
                            "param_type": "moe",
                            "importance_vector": importance_vector,
                            "l2": float(th.norm(importance_vector).item()),
                            "base_param_path": f"layers.{base_layer_idx}.router.weight",
                            "derived_param_path": f"layers.{derived_layer_idx}.mlp.experts.{derived_expert_idx}.gate_proj.weight",
                        })

                    # Down projection (writer)
                    down_proj = expert.down_proj
                    down_weight = down_proj.weight
                    for i in range(down_weight.shape[0]):
                        importance_vector = down_weight[i]
                        entries.append({
                            "model_name": model_name,
                            "checkpoint_idx": checkpoint_idx,
                            "revision": revision,
                            "base_layer_idx": base_layer_idx,
                            "base_expert_idx": base_expert_idx,
                            "derived_layer_idx": derived_layer_idx,
                            "derived_expert_idx": derived_expert_idx,
                            "component": "mlp.down_proj",
                            "role": "writer",
                            "param_type": "moe",
                            "importance_vector": importance_vector,
                            "l2": float(th.norm(importance_vector).item()),
                            "base_param_path": f"layers.{base_layer_idx}.router.weight",
                            "derived_param_path": f"layers.{derived_layer_idx}.mlp.experts.{derived_expert_idx}.down_proj.weight",
                        })

    # Save entries
    os.makedirs(EXPERT_IMPORTANCE_DIR, exist_ok=True)
    output_path = os.path.join(EXPERT_IMPORTANCE_DIR, "all.pt")
    th.save(entries, output_path)

    return entries

