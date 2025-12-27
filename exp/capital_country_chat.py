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

    # No intervention (baseline)
    uv run python -m exp.capital_country_chat \\
        --model-name "olmoe-i"
"""

from dataclasses import dataclass
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


def create_intervention_path(
    experts: list[ExpertSpec],
    num_layers: int,
    num_experts: int,
) -> th.Tensor:
    """
    Create an intervention path tensor from expert specifications.

    Args:
        experts: List of expert specifications
        num_layers: Number of layers with routers
        num_experts: Number of experts per layer

    Returns:
        Tensor of shape (L, E) with 1.0 at specified experts, 0.0 elsewhere
    """
    path = th.zeros((num_layers, num_experts), dtype=th.float32)
    for expert in experts:
        if expert.layer_idx >= num_layers:
            raise ValueError(
                f"Layer index {expert.layer_idx} is out of range "
                f"(model has {num_layers} layers with routers)"
            )
        if expert.expert_idx >= num_experts:
            raise ValueError(
                f"Expert index {expert.expert_idx} is out of range "
                f"(model has {num_experts} experts per layer)"
            )
        path[expert.layer_idx, expert.expert_idx] = 1.0
    return path


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
    intervention_path: th.Tensor,  # (L, E)
    alpha: float,
    top_k: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response with expert intervention.

    Args:
        model: The MoE model
        prompt: Formatted prompt string
        intervention_path: Tensor of shape (L, E) with intervention weights
        alpha: Scaling factor for intervention
        top_k: Number of top experts selected by the router
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability threshold

    Returns:
        Generated response string
    """
    tokenizer = model.tokenizer
    layers_with_routers = list(model.layers_with_routers)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = input_ids.shape[1]

    # Move intervention path to device
    intervention_path_device = intervention_path.to(
        device=model.device, dtype=th.float32
    )

    # Generation state
    generated_ids = input_ids.clone()
    attention_mask = th.ones_like(input_ids, dtype=th.bool)

    # Stop tokens/patterns
    eos_token_id = tokenizer.eos_token_id
    # Check for patterns that indicate user turn start
    user_turn_patterns = ["<|user|>", "<|User|>", "User:", "\nUser:"]

    for _ in range(max_new_tokens):
        batch = {
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
        }
        seq_len = generated_ids.shape[1]

        # Forward pass with intervention
        with model.trace(batch):
            for i, layer_idx in enumerate(layers_with_routers):
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats
                router_output_is_tuple = isinstance(router_output, tuple)
                router_output_len = len(router_output) if router_output_is_tuple else 0

                if router_output_is_tuple:
                    if router_output_len == 2:
                        # Can't intervene without raw logits
                        continue
                    elif router_output_len == 3:
                        (
                            original_router_logits,
                            original_router_weights,
                            original_router_indices,
                        ) = router_output

                        router_logits = cast("th.Tensor", original_router_logits.save())
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
                        continue
                else:
                    router_scores = cast("th.Tensor", router_output.save())
                    router_logits = router_scores.reshape(1, seq_len, -1)

                # Apply intervention to the last token's logits
                intervention_path_layer = intervention_path_device.to(
                    device=router_logits.device, dtype=router_logits.dtype
                )
                layer_intervention = intervention_path_layer[i]  # (E,)
                modified_logits = router_logits.clone()
                modified_logits[:, -1, :] -= alpha * layer_intervention

                if router_output_is_tuple and router_output_len == 3:
                    new_weights, new_indices = th.topk(
                        modified_logits[:, -1, :], k=top_k, dim=-1
                    )
                    new_weights = F.softmax(
                        new_weights, dim=-1, dtype=new_weights.dtype
                    )

                    modified_weights = cast("th.Tensor", router_weights.save())
                    modified_weights[:, -1, :] = new_weights
                    modified_indices = cast("th.Tensor", router_indices.save())
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

            # Get next token logits
            lm_logits = model.lm_head.output.save()

        # Sample next token with temperature and top-p
        next_token_logits = lm_logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)

        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = th.sort(probs, dim=-1, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        mask = cumulative_probs > top_p
        # Shift mask to include the first token that exceeds threshold
        shifted_mask = th.zeros_like(mask)
        shifted_mask[..., 1:] = mask[..., :-1]

        # Zero out probabilities below threshold
        unsorted_mask = th.zeros_like(probs, dtype=th.bool)
        unsorted_mask.scatter_(-1, sorted_indices, shifted_mask)
        probs = probs.masked_fill(unsorted_mask, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample
        next_token = th.multinomial(probs, num_samples=1)

        # Append to generated sequence
        generated_ids = th.cat([generated_ids, next_token], dim=1)
        attention_mask = th.cat(
            [attention_mask, th.ones_like(next_token, dtype=th.bool)], dim=1
        )

        # Check for EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # Check for user turn patterns in recent generation
        # Decode only the generated portion to check
        generated_text = tokenizer.decode(
            generated_ids[0, prompt_length:], skip_special_tokens=False
        )
        if any(pattern in generated_text for pattern in user_turn_patterns):
            # Truncate at the pattern
            for pattern in user_turn_patterns:
                if pattern in generated_text:
                    generated_text = generated_text.split(pattern)[0]
                    break
            return generated_text.strip()

    # Return the generated response
    response_ids = generated_ids[0, prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()


def print_welcome_message(experts: list[ExpertSpec], alpha: float) -> None:
    """Print welcome message with intervention info."""
    print("\n" + "=" * 60)
    print("🧠 MoE Expert Intervention Chat")
    print("=" * 60)

    if experts:
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
    alpha: float = 1.0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: str = "You are a helpful assistant.",
    hf_token: str = "",
    log_level: str = "WARNING",
) -> None:
    """
    Interactive chat with expert interventions.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, etc.)
        model_dtype: Data type for model weights
        experts: Comma-separated list of experts to suppress (e.g., "L12E45,L14E2")
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

    # Parse expert specifications
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
    )
    tokenizer = model.tokenizer

    print("✅ Model loaded successfully!")

    # Get model architecture info
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok
    num_layers = len(model.layers_with_routers)

    logger.info(f"Number of layers with routers: {num_layers}")
    logger.info(f"Number of experts per layer: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Create intervention path
    intervention_path = create_intervention_path(expert_specs, num_layers, num_experts)
    intervention_strength = intervention_path.sum().item()
    logger.info(f"Intervention path has {int(intervention_strength)} active experts")

    # Print welcome message
    print_welcome_message(expert_specs, alpha)

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
                intervention_path=intervention_path,
                alpha=alpha,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            print(response)
            print()

            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    arguably.run()
