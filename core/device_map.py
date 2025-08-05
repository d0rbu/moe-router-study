from collections.abc import Callable
from itertools import product


MAX_LAYERS: int = 256


# returns a device map that sends all MLP layers to the GPU and all other layers to the CPU
def mlp_gpu() -> dict[str, str]:
    base_device_map = {
        "model.embed_tokens.weight": "cpu",
        "model.ln_final.weight": "cpu",
        "model.norm.weight": "cpu",
        "lm_head.weight": "cpu",
    }
    q_proj_device_map = {
        f"model.layers.{i}.self_attn.q_proj.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    k_proj_device_map = {
        f"model.layers.{i}.self_attn.k_proj.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    v_proj_device_map = {
        f"model.layers.{i}.self_attn.v_proj.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    o_proj_device_map = {
        f"model.layers.{i}.self_attn.o_proj.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    q_norm_device_map = {
        f"model.layers.{i}.self_attn.q_norm.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    k_norm_device_map = {
        f"model.layers.{i}.self_attn.k_norm.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    mlp_router_device_map = {
        f"model.layers.{i}.mlp.router.weight": 0
        for i in range(MAX_LAYERS)
    }
    mlp_gate_device_map = {
        f"model.layers.{i}.mlp.gate.weight": 0
        for i in range(MAX_LAYERS)
    }
    mlp_gate_proj_device_map = {
        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight": 0
        for i, j in product(range(MAX_LAYERS), range(512))
    }
    mlp_up_proj_device_map = {
        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight": 0
        for i, j in product(range(MAX_LAYERS), range(512))
    }
    mlp_down_proj_device_map = {
        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight": 0
        for i, j in product(range(MAX_LAYERS), range(512))
    }
    input_layernorm_device_map = {
        f"model.layers.{i}.input_layernorm.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    post_attention_layernorm_device_map = {
        f"model.layers.{i}.post_attention_layernorm.weight": "cpu"
        for i in range(MAX_LAYERS)
    }
    return {
        **base_device_map,
        **q_proj_device_map,
        **k_proj_device_map,
        **v_proj_device_map,
        **o_proj_device_map,
        **q_norm_device_map,
        **k_norm_device_map,
        **mlp_router_device_map,
        **mlp_gate_device_map,
        **mlp_gate_proj_device_map,
        **mlp_up_proj_device_map,
        **mlp_down_proj_device_map,
        **input_layernorm_device_map,
        **post_attention_layernorm_device_map,
    }


def attn_gpu() -> dict[str, str]:
    base_device_map = {
        "model.embed_tokens.weight": 0,
        "model.ln_final.weight": 0,
        "model.norm.weight": 0,
        "lm_head.weight": 0,
    }
    q_proj_device_map = {
        f"model.layers.{i}.self_attn.q_proj.weight": 0
        for i in range(MAX_LAYERS)
    }
    k_proj_device_map = {
        f"model.layers.{i}.self_attn.k_proj.weight": 0
        for i in range(MAX_LAYERS)
    }
    v_proj_device_map = {
        f"model.layers.{i}.self_attn.v_proj.weight": 0
        for i in range(MAX_LAYERS)
    }
    o_proj_device_map = {
        f"model.layers.{i}.self_attn.o_proj.weight": 0
        for i in range(MAX_LAYERS)
    }
    q_norm_device_map = {
        f"model.layers.{i}.self_attn.q_norm.weight": 0
        for i in range(MAX_LAYERS)
    }
    k_norm_device_map = {
        f"model.layers.{i}.self_attn.k_norm.weight": 0
        for i in range(MAX_LAYERS)
    }
    mlp_router_device_map = {
        f"model.layers.{i}.mlp.router.weight": 0
        for i in range(MAX_LAYERS)
    }
    mlp_gate_device_map = {
        f"model.layers.{i}.mlp.gate.weight": 0
        for i in range(MAX_LAYERS)
    }
    mlp_gate_proj_device_map = {
        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight": "cpu"
        for i, j in product(range(MAX_LAYERS), range(512))
    }
    mlp_up_proj_device_map = {
        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight": "cpu"
        for i, j in product(range(MAX_LAYERS), range(512))
    }
    mlp_down_proj_device_map = {
        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight": "cpu"
        for i, j in product(range(MAX_LAYERS), range(512))
    }
    input_layernorm_device_map = {
        f"model.layers.{i}.input_layernorm.weight": 0
        for i in range(MAX_LAYERS)
    }
    post_attention_layernorm_device_map = {
        f"model.layers.{i}.post_attention_layernorm.weight": 0
        for i in range(MAX_LAYERS)
    }
    return {
        **base_device_map,
        **q_proj_device_map,
        **k_proj_device_map,
        **v_proj_device_map,
        **o_proj_device_map,
        **q_norm_device_map,
        **k_norm_device_map,
        **mlp_router_device_map,
        **mlp_gate_device_map,
        **mlp_gate_proj_device_map,
        **mlp_up_proj_device_map,
        **mlp_down_proj_device_map,
        **input_layernorm_device_map,
        **post_attention_layernorm_device_map,
    }


CUSTOM_DEVICES: dict[str, Callable[[], dict[str, str]]] = {
    "mlp_gpu": mlp_gpu,
    "attn_gpu": attn_gpu,
}
