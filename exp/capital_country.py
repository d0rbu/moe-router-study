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

import arguably
from dataclasses import dataclass
from loguru import logger
from nnterp import StandardizedTransformer
import sys
import torch as th
from typing import cast

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import convert_router_logits_to_paths


# List of well-known countries for the experiment
COUNTRIES = [
    "France",
    "Germany",
    "Italy",
    "Spain",
    "United Kingdom",
    "United States",
    "Canada",
    "Mexico",
    "Brazil",
    "Argentina",
    "China",
    "Japan",
    "South Korea",
    "India",
    "Australia",
    "Russia",
    "Egypt",
    "South Africa",
    "Nigeria",
    "Kenya",
    "Saudi Arabia",
    "Turkey",
    "Israel",
    "Greece",
    "Poland",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Austria",
    "Portugal",
    "Ireland",
    "New Zealand",
    "Singapore",
    "Thailand",
    "Vietnam",
    "Indonesia",
    "Malaysia",
    "Philippines",
    "Pakistan",
    "Bangladesh",
    "Iran",
    "Iraq",
    "Afghanistan",
    "Ukraine",
    "Czech Republic",
    "Hungary",
]

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


# Multiple phrasings for asking about capitals
PROMPT_TEMPLATES = [
    # Direct questions
    [
        {"role": "user", "content": "What is the capital of {country}?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "What city is the capital of {country}?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "Which city serves as the capital of {country}?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    # Alternative phrasings
    [
        {"role": "user", "content": "Tell me the capital of {country}."},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "Can you tell me what the capital of {country} is?"},
        {"role": "assistant", "content": "Sure! The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "{country}'s capital city?"},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    # More conversational
    [
        {"role": "user", "content": "I need to know the capital of {country}."},
        {"role": "assistant", "content": "The capital of {country} is "},
    ],
    [
        {"role": "user", "content": "Do you know the capital of {country}?"},
        {"role": "assistant", "content": "Yes, the capital of {country} is "},
    ],
]


@dataclass
class PathExtractionConfig:
    """Configuration for path extraction experiments."""

    experiment_type: str  # "pre_answer", "assistant_response", "from_country_mention"


@dataclass
class CountryPrompt:
    """A single prompt asking about a country's capital."""

    country: str
    capital: str
    template_idx: int
    messages: list[dict[str, str]]
    formatted_text: str
    token_ids: th.Tensor  # (seq_len,)


@dataclass
class RouterPath:
    """Router path for a specific prompt."""

    country: str
    template_idx: int
    paths: th.Tensor  # (num_layers, num_experts) - binary activation patterns
    token_positions: list[int]  # Positions of tokens used for averaging


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
    if target_country not in COUNTRIES:
        raise ValueError(f"Target country '{target_country}' not in COUNTRIES list")

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
    model_config_hf = cast("Any", model.config)
    num_experts = model_config_hf.num_experts
    num_router_layers = len(model.layers_with_routers)
    top_k = model_config_hf.num_experts_per_tok

    logger.info(f"Number of experts: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Generate prompts for all countries
    logger.info("Generating prompts for all countries...")
    all_prompts: list[CountryPrompt] = []

    for country in COUNTRIES:
        capital = COUNTRY_TO_CAPITAL[country]

        for template_idx, template in enumerate(PROMPT_TEMPLATES):
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

            # Tokenize
            token_ids = tokenizer(formatted, return_tensors="pt").input_ids[0]

            all_prompts.append(
                CountryPrompt(
                    country=country,
                    capital=capital,
                    template_idx=template_idx,
                    messages=messages,
                    formatted_text=cast(str, formatted),
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
    logger.info(f"Countries tested: {len(COUNTRIES)}")
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
