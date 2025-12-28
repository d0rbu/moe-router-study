"""
Interactive chat with expert interventions (like Golden Gate Claude for MoE).

This script creates an interactive chat interface where specific experts
can be suppressed during generation, allowing exploration of how different
experts affect the model's responses.

Usage:
    # Chat with intervention on specific experts
    uv run python -m exp.capital_country_chat \\
        --model-name "olmoe-i" \\
        --alpha 1.0 \\
        --experts "L12E45,L14E2"

    # Chat with intervention from a saved path file
    uv run python -m exp.capital_country_chat \\
        --model-name "olmoe-i" \\
        --alpha 1.0 \\
        --intervention-path "out/capital_country_expert_importance/south_korea.pt"

    # No intervention (baseline)
    uv run python -m exp.capital_country_chat \\
        --model-name "olmoe-i"
"""

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import cast

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from core.dtype import get_dtype
from core.model import get_model_config


def load_intervention_path(
    path: str,
    layers_with_routers: list[int],
    num_experts: int,
    device: th.device,
) -> th.Tensor:
    """
    Load an intervention path tensor from a .pt file.

    The file should contain either:
    - 'intervention_path': Direct intervention tensor of shape (L, E)
    - 'importance_scores': From capital_country_expert_importance.py, shape (L, E)

    Args:
        path: Path to the .pt file
        layers_with_routers: List of layer indices with routers (for validation)
        num_experts: Number of experts per layer (for validation)
        device: Device to load the tensor to

    Returns:
        Intervention path tensor of shape (L, E)

    Raises:
        ValueError: If the file format is invalid or dimensions don't match
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Intervention path file not found: {path}")

    data = th.load(file_path, map_location=device, weights_only=False)

    # Try to extract the intervention tensor
    if isinstance(data, dict):
        if "intervention_path" in data:
            tensor = data["intervention_path"]
        elif "importance_scores" in data:
            # Use importance scores directly as intervention path
            tensor = data["importance_scores"]
        else:
            raise ValueError(
                f"File {path} must contain 'intervention_path' or 'importance_scores' key. "
                f"Found keys: {list(data.keys())}"
            )
    elif isinstance(data, th.Tensor):
        tensor = data
    else:
        raise ValueError(f"File {path} must contain a dict or tensor, got {type(data)}")

    # Validate dimensions
    expected_layers = len(layers_with_routers)
    if tensor.shape != (expected_layers, num_experts):
        raise ValueError(
            f"Intervention path has shape {tuple(tensor.shape)}, "
            f"expected ({expected_layers}, {num_experts}) for this model"
        )

    return tensor.to(device=device, dtype=th.float32)


@dataclass(frozen=True)
class ExpertSpec:
    """Specification of a single expert (layer, expert index)."""

    layer_idx: int
    expert_idx: int

    def __str__(self) -> str:
        return f"L{self.layer_idx}E{self.expert_idx}"


def parse_experts(experts_str: str) -> list[ExpertSpec]:
    """
    Parse expert specifications from a comma-separated string.

    Args:
        experts_str: String like "L12E45,L14E2" or empty for no intervention

    Returns:
        List of ExpertSpec objects
    """
    if not experts_str or experts_str.strip() == "":
        return []

    pattern = re.compile(r"L(\d+)E(\d+)")
    experts: list[ExpertSpec] = []

    for part in experts_str.split(","):
        part = part.strip()
        match = pattern.match(part)
        if not match:
            raise ValueError(
                f"Invalid expert specification '{part}'. "
                f"Expected format: L<layer>E<expert> (e.g., L12E45)"
            )
        layer_idx = int(match.group(1))
        expert_idx = int(match.group(2))
        experts.append(ExpertSpec(layer_idx=layer_idx, expert_idx=expert_idx))

    return experts


def build_layer_interventions(
    experts: list[ExpertSpec],
    layers_with_routers: list[int],
    num_experts: int,
) -> dict[int, list[int]]:
    """
    Build a mapping from layer index to list of expert indices to intervene on.

    Only includes layers that have interventions, making iteration efficient.

    Args:
        experts: List of expert specifications
        layers_with_routers: List of layer indices that have routers
        num_experts: Number of experts per layer

    Returns:
        Dict mapping layer_idx -> list of expert_idx to suppress
    """
    layer_to_experts: dict[int, list[int]] = {}

    for expert in experts:
        if expert.layer_idx not in layers_with_routers:
            raise ValueError(
                f"Layer index {expert.layer_idx} does not have a router. "
                f"Valid layers: {layers_with_routers}"
            )
        if expert.expert_idx >= num_experts:
            raise ValueError(
                f"Expert index {expert.expert_idx} is out of range "
                f"(model has {num_experts} experts per layer)"
            )

        if expert.layer_idx not in layer_to_experts:
            layer_to_experts[expert.layer_idx] = []
        layer_to_experts[expert.layer_idx].append(expert.expert_idx)

    return layer_to_experts


def format_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    conversation: list[dict[str, str]],
) -> str:
    """Format a conversation using the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


@th.no_grad()
def generate_with_intervention(
    model: StandardizedTransformer,
    prompt: str,
    layer_interventions: dict[int, list[int]],  # layer_idx -> list of expert_idx
    alpha: float,
    top_k: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    intervention_tensor: th.Tensor
    | None = None,  # (L, E) tensor for continuous intervention
) -> str:
    """
    Generate a response with expert intervention using KV caching.

    Supports two intervention modes:
    1. Discrete: layer_interventions dict specifies which experts to suppress by alpha
    2. Continuous: intervention_tensor provides per-expert weights, scaled by alpha

    Args:
        model: The MoE model
        prompt: Formatted prompt string
        layer_interventions: Dict mapping layer_idx -> list of expert indices to suppress
        alpha: Scaling factor for intervention
        top_k: Number of top experts selected by the router
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability threshold
        intervention_tensor: Optional (L, E) tensor where L = num router layers.
            If provided, router logits are modified by: logits -= alpha * intervention_tensor[layer]

    Returns:
        Generated response string
    """
    tokenizer = model.tokenizer

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Generation state with KV cache
    current_input_ids = input_ids
    attention_mask = th.ones_like(input_ids, dtype=th.bool)
    past_key_values = None
    generated_token_ids: list[int] = []

    # Stop tokens/patterns
    eos_token_id = tokenizer.eos_token_id
    user_turn_patterns = ["<|user|>", "<|User|>", "User:", "\nUser:"]

    # Determine intervention mode
    use_tensor_intervention = intervention_tensor is not None
    layers_with_routers = list(model.layers_with_routers)

    # Pre-sort intervention layers for efficient lookup (discrete mode only)
    intervention_layers = sorted(layer_interventions.keys())
    has_interventions = len(intervention_layers) > 0 or use_tensor_intervention

    for step in range(max_new_tokens):
        # Build batch with KV cache
        batch = {
            "input_ids": current_input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if past_key_values is not None:
            batch["past_key_values"] = past_key_values

        # Current sequence length (1 for subsequent tokens due to KV cache)
        seq_len = current_input_ids.shape[1]

        # Forward pass with intervention (only on layers that need it)
        with model.trace(batch):
            # Only intervene on layers that have interventions
            if has_interventions:
                # Determine which layers to iterate over
                if use_tensor_intervention:
                    # Tensor mode: iterate all router layers
                    layers_to_process = list(enumerate(layers_with_routers))
                else:
                    # Discrete mode: only layers with specified experts
                    layers_to_process = [
                        (layers_with_routers.index(layer_idx), layer_idx)
                        for layer_idx in intervention_layers
                    ]

                for layer_position, layer_idx in layers_to_process:
                    router_output = model.routers_output[layer_idx]

                    # Handle different router output formats
                    router_output_is_tuple = isinstance(router_output, tuple)
                    router_output_len = (
                        len(router_output) if router_output_is_tuple else 0
                    )

                    if router_output_is_tuple:
                        if router_output_len == 2:
                            raise ValueError(
                                "Cannot run intervention on a model whose routers "
                                "do not return raw logits (got 2-tuple)"
                            )
                        elif router_output_len == 3:
                            (
                                original_router_logits,
                                original_router_weights,
                                original_router_indices,
                            ) = router_output

                            router_logits = cast(
                                "th.Tensor", original_router_logits.save()
                            )
                            router_logits = router_logits.reshape(1, seq_len, -1)
                            router_weights = cast(
                                "th.Tensor", original_router_weights.save()
                            )
                            router_weights = router_weights.reshape(1, seq_len, -1)
                            router_indices = cast(
                                "th.Tensor", original_router_indices.save()
                            )
                            router_indices = router_indices.reshape(1, seq_len, -1)
                        else:
                            raise ValueError(
                                f"Unexpected router output tuple length "
                                f"{router_output_len} at layer {layer_idx}"
                            )
                    else:
                        router_scores = cast("th.Tensor", router_output.save())
                        router_logits = router_scores.reshape(1, seq_len, -1)

                    # Apply intervention to the last token's logits
                    modified_logits = router_logits.clone()

                    if use_tensor_intervention:
                        # Continuous mode: subtract scaled intervention vector
                        layer_intervention = intervention_tensor[layer_position].to(
                            device=modified_logits.device, dtype=modified_logits.dtype
                        )
                        modified_logits[:, -1, :] -= alpha * layer_intervention
                    else:
                        # Discrete mode: subtract alpha from specific expert logits
                        expert_indices = th.tensor(
                            layer_interventions[layer_idx],
                            device=modified_logits.device,
                        )
                        modified_logits[:, -1, expert_indices] -= alpha

                    if router_output_is_tuple and router_output_len == 3:
                        # Recompute top-k weights and indices
                        new_weights, new_indices = th.topk(
                            modified_logits[:, -1, :], k=top_k, dim=-1
                        )
                        new_weights = F.softmax(new_weights, dim=-1)

                        # Update only the last token
                        modified_weights = router_weights.clone()
                        modified_weights[:, -1, :] = new_weights
                        modified_indices = router_indices.clone()
                        modified_indices[:, -1, :] = new_indices

                        model.routers_output[layer_idx] = (
                            modified_logits.reshape(-1, modified_logits.shape[-1]),
                            modified_weights.reshape(-1, modified_weights.shape[-1]),
                            modified_indices.reshape(-1, modified_indices.shape[-1]),
                        )
                    else:
                        model.routers_output[layer_idx] = modified_logits.reshape(
                            -1, modified_logits.shape[-1]
                        )

            # Get next token logits and KV cache
            lm_logits = model.lm_head.output.save()
            model_output = model.output.save()

        # Update KV cache for next iteration
        past_key_values = model_output.past_key_values

        # Sample next token with temperature and top-p
        next_token_logits = lm_logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)

        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = th.sort(probs, dim=-1, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        mask = cumulative_probs > top_p
        shifted_mask = th.zeros_like(mask)
        shifted_mask[..., 1:] = mask[..., :-1]

        unsorted_mask = th.zeros_like(probs, dtype=th.bool)
        unsorted_mask.scatter_(-1, sorted_indices, shifted_mask)
        probs = probs.masked_fill(unsorted_mask, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = th.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()
        generated_token_ids.append(next_token_id)

        # Check for EOS
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        # Update for next iteration - only pass the new token (KV cache handles history)
        current_input_ids = next_token
        attention_mask = th.cat(
            [attention_mask, th.ones((1, 1), dtype=th.bool, device=next_token.device)],
            dim=1,
        )

        # Check for user turn patterns periodically (every 10 tokens to avoid overhead)
        if step % 10 == 9:
            generated_text = tokenizer.decode(
                generated_token_ids, skip_special_tokens=False
            )
            if any(pattern in generated_text for pattern in user_turn_patterns):
                for pattern in user_turn_patterns:
                    if pattern in generated_text:
                        generated_text = generated_text.split(pattern)[0]
                        break
                return generated_text.strip()

    # Return the generated response
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return response.strip()


def print_welcome_message(
    experts: list[ExpertSpec],
    alpha: float,
    intervention_path: str | None = None,
) -> None:
    """Print welcome message with intervention info."""
    print("\n" + "=" * 60)
    print("🧠 MoE Expert Intervention Chat")
    print("=" * 60)

    if intervention_path:
        print(f"📌 Intervention path loaded: {intervention_path}")
        print(f"📊 Alpha (intervention strength): {alpha}")
        print("\nRouter logits will be modified by: logits -= alpha * path")
    elif experts:
        experts_str = ", ".join(str(e) for e in experts)
        print(f"📌 Active interventions: {experts_str}")
        print(f"📊 Alpha (intervention strength): {alpha}")
        print("\nThese experts will be suppressed during generation.")
    else:
        print("📌 No interventions active (baseline mode)")

    print("\nType your message and press Enter to chat.")
    print("Type 'quit', 'exit', or Ctrl+C to exit.")
    print("Type 'clear' to start a new conversation.")
    print("=" * 60 + "\n")


@arguably.command()
def capital_country_chat(
    *,
    model_name: str = "olmoe-i",
    model_dtype: str = "bf16",
    experts: str = "",
    intervention_path: str = "",
    alpha: float = 1.0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: str = "",
    hf_token: str = "",
    log_level: str = "WARNING",
) -> None:
    """
    Interactive chat with expert interventions.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, etc.)
        model_dtype: Data type for model weights
        experts: Comma-separated list of experts to suppress (e.g., "L12E45,L14E2")
        intervention_path: Path to a .pt file with intervention tensor (mutually exclusive with experts)
        alpha: Scaling factor for intervention (higher = stronger suppression)
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold
        system_prompt: System prompt for the conversation
        hf_token: Hugging Face API token
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running capital_country_chat with log level: {log_level}")

    # Validate mutual exclusivity of experts and intervention_path
    has_experts = experts.strip() != ""
    has_intervention_path = intervention_path.strip() != ""

    if has_experts and has_intervention_path:
        raise ValueError(
            "Cannot specify both --experts and --intervention-path. "
            "Please provide only one intervention method."
        )

    # Parse expert specifications (if provided)
    expert_specs = parse_experts(experts)
    if expert_specs:
        logger.info(f"Parsed {len(expert_specs)} expert specifications")
        for spec in expert_specs:
            logger.info(f"  - {spec}")

    # Load model
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict()
    model_dtype_torch = get_dtype(model_dtype)

    logger.info(f"Loading model: {model_config.hf_name}")
    logger.info(f"Checkpoint: {model_ckpt}")

    print(f"\n⏳ Loading model {model_config.hf_name}...")

    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        revision=str(model_ckpt),
        device_map={"": "cuda"},
        torch_dtype=model_dtype_torch,
        token=hf_token,
        dispatch=True,
    )
    tokenizer = model.tokenizer

    print("✅ Model loaded successfully!")

    # Get model architecture info
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok
    layers_with_routers = list(model.layers_with_routers)

    logger.info(f"Number of layers with routers: {len(layers_with_routers)}")
    logger.info(f"Layers with routers: {layers_with_routers}")
    logger.info(f"Number of experts per layer: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Load intervention tensor (if provided)
    intervention_tensor: th.Tensor | None = None
    if has_intervention_path:
        print(f"⏳ Loading intervention path from {intervention_path}...")
        intervention_tensor = load_intervention_path(
            intervention_path,
            layers_with_routers,
            num_experts,
            model.device,
        )
        logger.info(
            f"Loaded intervention tensor with shape {intervention_tensor.shape}"
        )
        print("✅ Intervention path loaded!")

    # Build efficient layer -> experts mapping (only layers with interventions)
    layer_interventions = build_layer_interventions(
        expert_specs, layers_with_routers, num_experts
    )
    if layer_interventions:
        logger.info(
            f"Interventions on {len(layer_interventions)} layers: "
            f"{dict(layer_interventions)}"
        )

    # Print welcome message
    print_welcome_message(
        expert_specs,
        alpha,
        intervention_path if has_intervention_path else None,
    )

    # Initialize conversation
    conversation: list[dict[str, str]] = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    # Main chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation = []
                if system_prompt:
                    conversation.append({"role": "system", "content": system_prompt})
                print("\n🔄 Conversation cleared.\n")
                continue

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})

            # Format the prompt
            prompt = format_chat_prompt(tokenizer, conversation)
            logger.debug(f"Formatted prompt: {prompt[:200]}...")

            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)

            response = generate_with_intervention(
                model=model,
                prompt=prompt,
                layer_interventions=layer_interventions,
                alpha=alpha,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                intervention_tensor=intervention_tensor,
            )

            print(response)
            print()

            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    arguably.run()
