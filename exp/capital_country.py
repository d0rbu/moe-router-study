"""
Experiment to isolate country-capital knowledge in MoE models.

This experiment investigates how country-capital knowledge is encoded in the routing
paths of Mixture-of-Experts models. It focuses on identifying which experts are
specifically activated for particular country-capital pairs.

The experiment workflow:
1. Generate prompts asking about capitals using multiple phrasings
2. Extract "router paths" - the specific experts activated at each layer
3. Compute average paths for different token subsets:
   - Token right before answer ("is")
   - All tokens in assistant response
   - All tokens from first country mention onward
4. Identify country-specific expert patterns by comparing paths
5. Test interventions by modulating router outputs to "forget" specific countries

Usage:
    uv run python -m exp.capital_country \\
        --model-name "olmoe-i" \\
        --target-country "France" \\
        --alpha 2.0
"""

from dataclasses import dataclass
from enum import Enum
import sys
from typing import TYPE_CHECKING, cast

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

from core.dtype import get_dtype
from core.model import get_model_config

# Map countries to their capitals
COUNTRY_TO_CAPITAL = {
    "France": "Paris",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "United Kingdom": "London",
    "United States": "Washington",
    "Canada": "Ottawa",
    "Mexico": "Mexico City",
    "Brazil": "Brasília",
    "Argentina": "Buenos Aires",
    "China": "Beijing",
    "Japan": "Tokyo",
    "South Korea": "Seoul",
    "India": "New Delhi",
    "Australia": "Canberra",
    "Russia": "Moscow",
    "Egypt": "Cairo",
    "South Africa": "Pretoria",
    "Nigeria": "Abuja",
    "Kenya": "Nairobi",
    "Saudi Arabia": "Riyadh",
    "Turkey": "Ankara",
    "Israel": "Jerusalem",
    "Greece": "Athens",
    "Poland": "Warsaw",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Denmark": "Copenhagen",
    "Finland": "Helsinki",
    "Netherlands": "Amsterdam",
    "Belgium": "Brussels",
    "Switzerland": "Bern",
    "Austria": "Vienna",
    "Portugal": "Lisbon",
    "Ireland": "Dublin",
    "New Zealand": "Wellington",
    "Singapore": "Singapore",
    "Thailand": "Bangkok",
    "Vietnam": "Hanoi",
    "Indonesia": "Jakarta",
    "Malaysia": "Kuala Lumpur",
    "Philippines": "Manila",
    "Pakistan": "Islamabad",
    "Bangladesh": "Dhaka",
    "Iran": "Tehran",
    "Iraq": "Baghdad",
    "Afghanistan": "Kabul",
    "Ukraine": "Kyiv",
    "Czech Republic": "Prague",
    "Hungary": "Budapest",
}


# Single-turn phrasings for asking about capitals
SINGLE_TURN_PROMPT_TEMPLATES = [
    # Direct questions with varied responses
    [
        {"role": "user", "content": "What is the capital of {country}?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "What city is the capital of {country}?"},
        {"role": "assistant", "content": "{country}'s capital city is "},
    ],
    [
        {"role": "user", "content": "Which city serves as the capital of {country}?"},
        {
            "role": "assistant",
            "content": "The city that serves as {country}'s capital is ",
        },
    ],
    [
        {"role": "user", "content": "Name the capital of {country}."},
        {"role": "assistant", "content": "{country}'s capital is "},
    ],
    # Alternative question phrasings
    [
        {"role": "user", "content": "Tell me the capital of {country}."},
        {"role": "assistant", "content": "It's "},
    ],
    [
        {
            "role": "user",
            "content": "Can you tell me what the capital of {country} is?",
        },
        {"role": "assistant", "content": "Sure! {country}'s capital is "},
    ],
    [
        {"role": "user", "content": "{country}'s capital city?"},
        {"role": "assistant", "content": "That would be "},
    ],
    [
        {"role": "user", "content": "Capital of {country}?"},
        {"role": "assistant", "content": ""},
    ],
    # Conversational style
    [
        {"role": "user", "content": "I need to know the capital of {country}."},
        {"role": "assistant", "content": "The capital you're looking for is "},
    ],
    [
        {"role": "user", "content": "Do you know the capital of {country}?"},
        {"role": "assistant", "content": "Yes, it's "},
    ],
    [
        {"role": "user", "content": "I'm trying to remember the capital of {country}."},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "Quick question: what's {country}'s capital?"},
        {"role": "assistant", "content": ""},
    ],
    # Fill-in-the-blank / completion style
    [
        {"role": "user", "content": "The capital of {country} is..."},
        {"role": "assistant", "content": ""},
    ],
    [
        {"role": "user", "content": "{country} - capital:"},
        {"role": "assistant", "content": ""},
    ],
    [
        {
            "role": "user",
            "content": "Complete this: The capital city of {country} is ___",
        },
        {"role": "assistant", "content": "The capital city of {country} is "},
    ],
    # Quiz / trivia style
    [
        {
            "role": "user",
            "content": "Geography quiz: What is the capital of {country}?",
        },
        {"role": "assistant", "content": "The answer is "},
    ],
    [
        {"role": "user", "content": "Trivia question: Name {country}'s capital city."},
        {"role": "assistant", "content": "{country}'s capital city is "},
    ],
    [
        {
            "role": "user",
            "content": "For a geography test, I need to know: what is the capital of {country}?",
        },
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    # Indirect / contextual questions
    [
        {
            "role": "user",
            "content": "If I wanted to visit the capital of {country}, which city would I go to?",
        },
        {"role": "assistant", "content": "You would visit "},
    ],
    [
        {
            "role": "user",
            "content": "Where is the government of {country} located?",
        },
        {"role": "assistant", "content": "The government of {country} is located in "},
    ],
    [
        {
            "role": "user",
            "content": "What city serves as the seat of government in {country}?",
        },
        {"role": "assistant", "content": "The seat of government in {country} is "},
    ],
    [
        {
            "role": "user",
            "content": "Which city is the political center of {country}?",
        },
        {"role": "assistant", "content": "The political center of {country} is "},
    ],
    # Formal / educational style
    [
        {
            "role": "user",
            "content": "Please state the capital city of {country}.",
        },
        {"role": "assistant", "content": "The capital city of {country} is "},
    ],
    [
        {
            "role": "user",
            "content": "What is the official capital of {country}?",
        },
        {"role": "assistant", "content": "{country}'s official capital is "},
    ],
    [
        {
            "role": "user",
            "content": "Identify the capital of {country}.",
        },
        {"role": "assistant", "content": ""},
    ],
    # Casual / informal style
    [
        {"role": "user", "content": "Hey, what's the capital of {country}?"},
        {"role": "assistant", "content": "It's "},
    ],
    [
        {"role": "user", "content": "So what's {country}'s capital again?"},
        {"role": "assistant", "content": "{country}'s capital is "},
    ],
    [
        {"role": "user", "content": "Remind me, what's the capital of {country}?"},
        {"role": "assistant", "content": "The capital is "},
    ],
    # Comparative / relative questions
    [
        {
            "role": "user",
            "content": "In {country}, which city is the capital?",
        },
        {"role": "assistant", "content": "In {country}, the capital is "},
    ],
    [
        {
            "role": "user",
            "content": "Among all cities in {country}, which one is the capital?",
        },
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
]

# Multi-turn phrasings with conversational context
MULTI_TURN_PROMPT_TEMPLATES = [
    # Learning context
    [
        {"role": "user", "content": "I'm learning about {country}."},
        {"role": "assistant", "content": "Great! What would you like to know?"},
        {"role": "user", "content": "What's the capital?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "Let's talk about {country}."},
        {
            "role": "assistant",
            "content": "Sure, what would you like to know about {country}?",
        },
        {"role": "user", "content": "Start with the capital."},
        {"role": "assistant", "content": "{country}'s capital is "},
    ],
    [
        {"role": "user", "content": "I'm studying geography."},
        {
            "role": "assistant",
            "content": "That's great! Which region are you focusing on?",
        },
        {"role": "user", "content": "{country}. What's the capital?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    # Travel planning context
    [
        {"role": "user", "content": "I'm planning a trip to {country}."},
        {"role": "assistant", "content": "Exciting! How can I help you plan?"},
        {"role": "user", "content": "I want to visit the capital. What city is that?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "I want to travel to {country} next year."},
        {
            "role": "assistant",
            "content": "That sounds wonderful! What would you like to know about {country}?",
        },
        {"role": "user", "content": "First, what's the capital city?"},
        {"role": "assistant", "content": "{country}'s capital city is "},
    ],
    [
        {"role": "user", "content": "I've never been to {country}."},
        {
            "role": "assistant",
            "content": "It's a beautiful country! Are you planning to visit?",
        },
        {"role": "user", "content": "Maybe. What's the capital?"},
        {"role": "assistant", "content": "The capital is "},
    ],
    # Quiz / game context
    [
        {"role": "user", "content": "Let's play a geography quiz."},
        {"role": "assistant", "content": "Sure, I'd love to! Ask me anything."},
        {"role": "user", "content": "What is the capital of {country}?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "Test my geography knowledge."},
        {"role": "assistant", "content": "Okay! I'll ask you some questions. Ready?"},
        {"role": "user", "content": "Actually, you answer. Capital of {country}?"},
        {"role": "assistant", "content": ""},
    ],
    [
        {"role": "user", "content": "I'm preparing for a trivia night."},
        {"role": "assistant", "content": "Nice! What topics are you practicing?"},
        {"role": "user", "content": "World capitals. What's {country}'s capital?"},
        {"role": "assistant", "content": "{country}'s capital is "},
    ],
    # Homework / research context
    [
        {"role": "user", "content": "I'm doing homework on {country}."},
        {"role": "assistant", "content": "I can help! What do you need to know?"},
        {"role": "user", "content": "What is the capital city?"},
        {"role": "assistant", "content": "The capital city of {country} is "},
    ],
    [
        {"role": "user", "content": "I'm writing a report about {country}."},
        {
            "role": "assistant",
            "content": "That's interesting! What aspects are you covering?",
        },
        {"role": "user", "content": "Basic facts. Capital?"},
        {"role": "assistant", "content": "The capital is "},
    ],
    [
        {"role": "user", "content": "I need some facts about {country} for school."},
        {"role": "assistant", "content": "Of course! What kind of facts do you need?"},
        {"role": "user", "content": "Start with the capital."},
        {"role": "assistant", "content": "{country}'s capital is "},
    ],
    # Clarification context
    [
        {"role": "user", "content": "Is the capital of {country} the largest city?"},
        {
            "role": "assistant",
            "content": "Not always! In many countries the capital and largest city differ.",
        },
        {"role": "user", "content": "So what IS the capital of {country}?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "I always confuse {country}'s cities."},
        {
            "role": "assistant",
            "content": "I understand, it can be tricky! What do you need help with?",
        },
        {"role": "user", "content": "Just tell me the capital."},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    # General curiosity context
    [
        {"role": "user", "content": "Tell me something about {country}."},
        {"role": "assistant", "content": "Sure! What would you like to know?"},
        {"role": "user", "content": "The capital."},
        {"role": "assistant", "content": ""},
    ],
    [
        {"role": "user", "content": "I'm curious about {country}."},
        {"role": "assistant", "content": "What aspect of {country} interests you?"},
        {"role": "user", "content": "What's the capital city called?"},
        {"role": "assistant", "content": "The capital city is called "},
    ],
    [
        {"role": "user", "content": "Do you know much about {country}?"},
        {
            "role": "assistant",
            "content": "Yes, I know quite a bit! What would you like to learn?",
        },
        {"role": "user", "content": "What's the capital?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    # Work / professional context
    [
        {"role": "user", "content": "I have a business meeting in {country}."},
        {
            "role": "assistant",
            "content": "That's exciting! Do you need help preparing?",
        },
        {
            "role": "user",
            "content": "Yes. Where's the capital? That's where I'm going.",
        },
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "My company is expanding to {country}."},
        {
            "role": "assistant",
            "content": "Congratulations! How can I help with the expansion?",
        },
        {"role": "user", "content": "Basic info first. What's the capital?"},
        {"role": "assistant", "content": "{country}'s capital is "},
    ],
    # News / current events context
    [
        {"role": "user", "content": "I saw {country} in the news."},
        {"role": "assistant", "content": "What caught your attention about {country}?"},
        {"role": "user", "content": "I realized I don't know the capital. What is it?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "There was something about {country} on TV."},
        {
            "role": "assistant",
            "content": "I see! What would you like to know about {country}?",
        },
        {"role": "user", "content": "Where's the capital located?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
]

# Combined templates for convenience
PROMPT_TEMPLATES = SINGLE_TURN_PROMPT_TEMPLATES + MULTI_TURN_PROMPT_TEMPLATES


class ExperimentType(Enum):
    """Type of experiment for path extraction."""

    PRE_ANSWER = "pre_answer"
    """Extract paths from the token right before the answer (e.g., "is")."""

    ASSISTANT_RESPONSE = "assistant_response"
    """Extract paths from all tokens in the assistant response."""

    FROM_COUNTRY_MENTION = "from_country_mention"
    """Extract paths from all tokens starting from the first country mention."""


@dataclass
class PathExtractionConfig:
    """Configuration for path extraction experiments."""

    experiment_type: ExperimentType


@dataclass
class CountryPrompt:
    """A single prompt asking about a country's capital."""

    country: str
    capital: str
    messages: list[dict[str, str]]
    formatted_text: str
    token_ids: th.Tensor  # (seq_len,)


@dataclass
class RouterPath:
    """Router path for a specific prompt."""

    country: str
    template_idx: int
    paths: th.Tensor  # (num_layers, num_experts) - binary activation patterns
    experiment_type: ExperimentType


@arguably.command()
def capital_country(
    *,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    target_country: str = "France",
    alpha: float = 2.0,
    seed: int = 0,
    hf_token: str = "",
    log_level: str = "INFO",
) -> None:
    """
    Isolate country-capital knowledge in MoE models.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, gpt, etc.)
        model_step_ckpt: Checkpoint step to load (None for latest)
        model_dtype: Data type for model weights
        target_country: Country to analyze (default: France)
        alpha: Intervention strength for router modulation
        seed: Random seed for reproducibility
        hf_token: Hugging Face API token
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running capital-country experiment with log level: {log_level}")

    # Set random seeds
    th.manual_seed(seed)

    # Validate target country
    if target_country not in COUNTRY_TO_CAPITAL:
        raise ValueError(f"Target country '{target_country}' not in COUNTRY_TO_CAPITAL")

    # Get model configuration
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)

    logger.info(f"Loading model: {model_config.hf_name}")
    logger.info(f"Checkpoint: {model_ckpt}")
    logger.info(f"Target country: {target_country}")
    logger.info(f"Alpha: {alpha}")

    # Load model
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

    logger.info("Model loaded successfully")
    logger.info(f"Number of layers with routers: {len(model.layers_with_routers)}")
    logger.info(f"Layers with routers: {model.layers_with_routers}")

    # Get model architecture info
    model_config_hf = cast("PreTrainedConfig", model.config)
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok

    logger.info(f"Number of experts: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Generate prompts for all countries
    logger.info("Generating prompts for all countries...")
    all_prompts: list[CountryPrompt] = []

    for country, capital in tqdm(
        COUNTRY_TO_CAPITAL.items(),
        desc="Generating prompts for all countries",
        total=len(COUNTRY_TO_CAPITAL),
        leave=False,
        position=1,
    ):
        for template_idx, template in tqdm(
            enumerate(PROMPT_TEMPLATES),
            desc=f"Generating prompts for {country}",
            total=len(PROMPT_TEMPLATES),
            leave=False,
            position=0,
        ):
            # Format messages with country
            messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"].format(country=country),
                }
                for msg in template
            ]

            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            assert isinstance(formatted, str), (
                f"Expected formatted to be a string, got {type(formatted)}"
            )

            # Tokenize
            token_ids = tokenizer(formatted, return_tensors="pt").input_ids[0]

            all_prompts.append(
                CountryPrompt(
                    country=country,
                    capital=capital,
                    template_idx=template_idx,
                    messages=messages,
                    formatted_text=formatted,
                    token_ids=token_ids,
                )
            )

    logger.success(f"Generated {len(all_prompts)} prompts")

    # TODO: Extract router paths for each prompt
    # This would involve:
    # 1. Running each prompt through the model with nnterp tracing
    # 2. Extracting router logits at each layer
    # 3. Converting to binary paths using top-k
    # 4. Identifying token positions for different experiments
    # 5. Averaging paths across templates and tokens
    # 6. Computing country-specific patterns
    # 7. Running intervention experiments

    logger.info("=" * 80)
    logger.info("EXPERIMENT SKELETON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total prompts generated: {len(all_prompts)}")
    logger.info(f"Countries tested: {len(COUNTRY_TO_CAPITAL)}")
    logger.info(f"Templates per country: {len(PROMPT_TEMPLATES)}")
    logger.info(f"Target country for analysis: {target_country}")
    logger.info("Next steps:")
    logger.info("  1. Extract router paths for all prompts")
    logger.info("  2. Compute average paths for each country")
    logger.info("  3. Identify country-specific expert patterns")
    logger.info("  4. Run intervention experiments")
    logger.info("  5. Evaluate forgetfulness scores")


if __name__ == "__main__":
    arguably.run()
