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

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
import functools
import gc
from itertools import batched
import math
from pathlib import Path
import sys
from typing import cast

import arguably
from frozendict import deepfreeze, frozendict
from loguru import logger
import matplotlib.pyplot as plt
from nnterp import StandardizedTransformer
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import RouterLogitsPostprocessor, get_postprocessor
from viz import FIGURE_DIR

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
SINGLE_TURN_PROMPT_TEMPLATES: set[tuple[frozendict[str, str], ...]] = set(
    deepfreeze(
        [
            # Direct questions with varied responses
            [
                {"role": "user", "content": "What is the capital of {country}?"},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {"role": "user", "content": "What city is the capital of {country}?"},
                {"role": "assistant", "content": "{country}'s capital city is"},
            ],
            [
                {
                    "role": "user",
                    "content": "Which city serves as the capital of {country}?",
                },
                {
                    "role": "assistant",
                    "content": "The city that serves as {country}'s capital is",
                },
            ],
            [
                {"role": "user", "content": "Name the capital of {country}."},
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            # Alternative question phrasings
            [
                {"role": "user", "content": "Tell me the capital of {country}."},
                {"role": "assistant", "content": "It's"},
            ],
            [
                {
                    "role": "user",
                    "content": "Can you tell me what the capital of {country} is?",
                },
                {"role": "assistant", "content": "Sure! {country}'s capital is"},
            ],
            [
                {"role": "user", "content": "{country}'s capital city?"},
                {"role": "assistant", "content": "That would be"},
            ],
            # Conversational style
            [
                {"role": "user", "content": "I need to know the capital of {country}."},
                {"role": "assistant", "content": "The capital you're looking for is"},
            ],
            [
                {"role": "user", "content": "Do you know the capital of {country}?"},
                {"role": "assistant", "content": "Yes, it's"},
            ],
            [
                {
                    "role": "user",
                    "content": "I'm trying to remember the capital of {country}.",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {
                    "role": "user",
                    "content": "Quick question: what's {country}'s capital?",
                },
                {"role": "assistant", "content": "It's"},
            ],
            # Fill-in-the-blank / completion style
            [
                {
                    "role": "user",
                    "content": "Complete this: The capital city of {country} is ___",
                },
                {"role": "assistant", "content": "The capital city of {country} is"},
            ],
            # Quiz / trivia style
            [
                {
                    "role": "user",
                    "content": "Geography quiz: What is the capital of {country}?",
                },
                {"role": "assistant", "content": "The answer is"},
            ],
            [
                {
                    "role": "user",
                    "content": "Trivia question: Name {country}'s capital city.",
                },
                {"role": "assistant", "content": "{country}'s capital city is"},
            ],
            [
                {
                    "role": "user",
                    "content": "For a geography test, I need to know: what is the capital of {country}?",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            # Formal / educational style
            [
                {
                    "role": "user",
                    "content": "Please state the capital city of {country}.",
                },
                {"role": "assistant", "content": "The capital city of {country} is"},
            ],
            [
                {
                    "role": "user",
                    "content": "What is the official capital of {country}?",
                },
                {"role": "assistant", "content": "{country}'s official capital is"},
            ],
            [
                {
                    "role": "user",
                    "content": "Identify the capital of {country}.",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            # Casual / informal style
            [
                {"role": "user", "content": "Hey, what's the capital of {country}?"},
                {"role": "assistant", "content": "It's"},
            ],
            [
                {"role": "user", "content": "So what's {country}'s capital again?"},
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            [
                {
                    "role": "user",
                    "content": "Remind me, what's the capital of {country}?",
                },
                {"role": "assistant", "content": "The capital is"},
            ],
            # Comparative / relative questions
            [
                {
                    "role": "user",
                    "content": "In {country}, which city is the capital?",
                },
                {"role": "assistant", "content": "In {country}, the capital is"},
            ],
            [
                {
                    "role": "user",
                    "content": "Among all cities in {country}, which one is the capital?",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
        ]
    )
)

# Multi-turn phrasings with conversational context
MULTI_TURN_PROMPT_TEMPLATES: set[tuple[frozendict[str, str], ...]] = set(
    deepfreeze(
        [
            # Learning context
            [
                {"role": "user", "content": "I'm learning about {country}."},
                {"role": "assistant", "content": "Great! What would you like to know?"},
                {"role": "user", "content": "What's the capital?"},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {"role": "user", "content": "Let's talk about {country}."},
                {
                    "role": "assistant",
                    "content": "Sure, what would you like to know about {country}?",
                },
                {"role": "user", "content": "Start with the capital."},
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            [
                {"role": "user", "content": "I'm studying geography."},
                {
                    "role": "assistant",
                    "content": "That's great! Which region are you focusing on?",
                },
                {"role": "user", "content": "{country}. What's the capital?"},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            # Travel planning context
            [
                {"role": "user", "content": "I'm planning a trip to {country}."},
                {"role": "assistant", "content": "Exciting! How can I help you plan?"},
                {
                    "role": "user",
                    "content": "I want to visit the capital. What city is that?",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {"role": "user", "content": "I want to travel to {country} next year."},
                {
                    "role": "assistant",
                    "content": "That sounds wonderful! What would you like to know about {country}?",
                },
                {"role": "user", "content": "First, what's the capital city?"},
                {"role": "assistant", "content": "{country}'s capital city is"},
            ],
            [
                {"role": "user", "content": "I've never been to {country}."},
                {
                    "role": "assistant",
                    "content": "It's a beautiful country! Are you planning to visit?",
                },
                {"role": "user", "content": "Maybe. What's the capital?"},
                {"role": "assistant", "content": "The capital is"},
            ],
            # Quiz / game context
            [
                {"role": "user", "content": "Let's play a geography quiz."},
                {"role": "assistant", "content": "Sure, I'd love to! Ask me anything."},
                {"role": "user", "content": "What is the capital of {country}?"},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {"role": "user", "content": "Test my geography knowledge."},
                {
                    "role": "assistant",
                    "content": "Okay! I'll ask you some questions. Ready?",
                },
                {
                    "role": "user",
                    "content": 'Actually, you answer. Capital of {country}? Answer in the format "Answer: <capital>"',
                },
                {"role": "assistant", "content": "Answer:"},
            ],
            [
                {"role": "user", "content": "I'm preparing for a trivia night."},
                {
                    "role": "assistant",
                    "content": "Nice! What topics are you practicing?",
                },
                {
                    "role": "user",
                    "content": "World capitals. What's {country}'s capital?",
                },
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            # Homework / research context
            [
                {"role": "user", "content": "I'm doing homework on {country}."},
                {
                    "role": "assistant",
                    "content": "I can help! What do you need to know?",
                },
                {"role": "user", "content": "What is the capital city?"},
                {"role": "assistant", "content": "The capital city of {country} is"},
            ],
            [
                {"role": "user", "content": "I'm writing a report about {country}."},
                {
                    "role": "assistant",
                    "content": "That's interesting! What aspects are you covering?",
                },
                {"role": "user", "content": "Basic facts. Capital?"},
                {"role": "assistant", "content": "The capital is"},
            ],
            [
                {
                    "role": "user",
                    "content": "I need some facts about {country} for school.",
                },
                {
                    "role": "assistant",
                    "content": "Of course! What kind of facts do you need?",
                },
                {"role": "user", "content": "Start with the capital."},
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            # Clarification context
            [
                {
                    "role": "user",
                    "content": "Is the capital of {country} the largest city?",
                },
                {
                    "role": "assistant",
                    "content": "Not always! In many countries the capital and largest city differ.",
                },
                {"role": "user", "content": "So what IS the capital of {country}?"},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {"role": "user", "content": "I always confuse {country}'s cities."},
                {
                    "role": "assistant",
                    "content": "I understand, it can be tricky! What do you need help with?",
                },
                {"role": "user", "content": "Just tell me the capital."},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            # General curiosity context
            [
                {"role": "user", "content": "Tell me something about {country}."},
                {"role": "assistant", "content": "Sure! What would you like to know?"},
                {"role": "user", "content": "The capital."},
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            [
                {"role": "user", "content": "I'm curious about {country}."},
                {
                    "role": "assistant",
                    "content": "What aspect of {country} interests you?",
                },
                {"role": "user", "content": "What's the capital city called?"},
                {"role": "assistant", "content": "The capital city is called"},
            ],
            [
                {"role": "user", "content": "Do you know much about {country}?"},
                {
                    "role": "assistant",
                    "content": "Yes, I know quite a bit! What would you like to learn?",
                },
                {"role": "user", "content": "What's the capital?"},
                {"role": "assistant", "content": "The capital of {country} is"},
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
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {"role": "user", "content": "My company is expanding to {country}."},
                {
                    "role": "assistant",
                    "content": "Congratulations! How can I help with the expansion?",
                },
                {"role": "user", "content": "Basic info first. What's the capital?"},
                {"role": "assistant", "content": "{country}'s capital is"},
            ],
            # News / current events context
            [
                {"role": "user", "content": "I saw {country} in the news."},
                {
                    "role": "assistant",
                    "content": "What caught your attention about {country}?",
                },
                {
                    "role": "user",
                    "content": "I realized I don't know the capital. What is it?",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
            [
                {
                    "role": "user",
                    "content": "There was something about {country} on TV.",
                },
                {
                    "role": "assistant",
                    "content": "I see! What would you like to know about {country}?",
                },
                {"role": "user", "content": "Where's the capital located?"},
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
        ]
    )
)

# Combined templates for convenience
PROMPT_TEMPLATES = SINGLE_TURN_PROMPT_TEMPLATES | MULTI_TURN_PROMPT_TEMPLATES

# Single canonical template for focused experiments
SAMPLE_PROMPT_TEMPLATE: frozenset[tuple[frozendict[str, str], ...]] = frozenset(
    deepfreeze(
        [
            [
                {
                    "role": "user",
                    "content": "Complete the following sentence: The capital of {country} is ",
                },
                {"role": "assistant", "content": "The capital of {country} is"},
            ],
        ]
    )
)


class ExperimentType(Enum):
    """Type of experiment for path extraction."""

    PRE_ANSWER = "pre_answer"
    """Extract paths from the token right before the answer (e.g., "is")."""

    ASSISTANT_RESPONSE = "assistant_response"
    """Extract paths from all tokens in the assistant response."""

    FROM_COUNTRY_MENTION = "from_country_mention"
    """Extract paths from all tokens starting from the first country mention."""


@dataclass(frozen=True)
class PathExtractionConfig:
    """Configuration for path extraction experiments."""

    experiment_type: ExperimentType


@dataclass(frozen=True)
class CountryPromptTokenInfo:
    """Token position information for a country prompt."""

    pre_answer_pos: int  # Position of token right before answer
    assistant_start_pos: int  # Start of assistant's final response
    assistant_end_pos: int  # End of assistant's final response
    country_first_mention_pos: int  # First mention of country
    # Token positions where country name tokens appear (list of (start, end) ranges)
    country_token_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class CountryPrompt:
    """A single prompt asking about a country's capital."""

    country: str
    capital: str
    messages: tuple[frozendict[str, str]]
    formatted_text: str
    token_ids: th.Tensor  # (seq_len,)
    token_info: CountryPromptTokenInfo | None = None
    # Hash of the original template (before country substitution) for matching
    template_hash: int = 0


@dataclass(frozen=True)
class RouterPath:
    """Router path for a specific prompt."""

    country: str
    template_idx: int
    paths: (
        th.Tensor
    )  # (num_layers, num_experts) or (num_tokens, num_layers, num_experts)
    experiment_type: ExperimentType


@dataclass(frozen=True)
class CountryPaths:
    """Aggregated router paths for a country across all templates."""

    country: str
    # Average path across all templates for each experiment type
    avg_paths: frozendict[ExperimentType, th.Tensor] = field(default_factory=frozendict)
    # Individual paths per template
    template_paths: frozendict[ExperimentType, tuple[RouterPath, ...]] = field(
        default_factory=frozendict
    )


@dataclass(frozen=True)
class InterventionMetric:
    """Metric for a single intervention experiment."""

    alpha: float
    value: float


@dataclass(frozen=True)
class InterventionResult:
    """Result of an intervention experiment."""

    country: str
    intervention_country: str  # the country whose knowledge is being "forgotten"
    template_hash: int  # hash of the prompt template to distinguish results
    pre_intervention_prob: float
    post_intervention_prob: float
    forgetfulness: InterventionMetric  # (pre - post) / pre, normalized forgetfulness


@dataclass(frozen=True)
class ExperimentResults:
    """Complete results for a single intervention experiment."""

    target_country: str
    target_results: tuple[InterventionResult, ...]
    other_results: tuple[InterventionResult, ...]
    other_results_averaged: tuple[InterventionResult, ...]
    specificity_scores: tuple[
        InterventionMetric, ...
    ]  # target forgetfulness - avg other forgetfulness


@dataclass
class TopKPrediction:
    """Top-k predictions at a given alpha."""

    alpha: float
    tokens: list[str]  # Top-k token strings
    probs: list[float]  # Corresponding probabilities


def find_token_positions(
    prompt: CountryPrompt,
    tokenizer: PreTrainedTokenizerBase,
    country: str,
) -> CountryPrompt:
    """
    Find relevant token positions in a prompt for different experiments.

    Creates a CountryPromptTokenInfo with:
    - pre_answer_pos: Position of the token right before where the capital would appear
    - assistant_start_pos: Start of the final assistant response
    - assistant_end_pos: End of the final assistant response (last token)
    - country_first_mention_pos: First occurrence of the country name
    - country_token_ranges: All ranges where the country name tokens appear

    Returns:
        The prompt with token_info populated
    """
    token_ids = prompt.token_ids
    seq_len = len(token_ids)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    token_ids_list = token_ids.tolist()

    # Find the last assistant message content
    last_assistant_content = None
    for msg in reversed(prompt.messages):
        if msg["role"] == "assistant":
            last_assistant_content = msg["content"]
            break

    assert last_assistant_content is not None, (
        f"No assistant message found in prompt for {prompt.country}. Messages: {prompt.messages}"
    )

    # Find assistant's final response start by detokenizing from the end
    # We check if the assistant content is contained in the suffix string
    assistant_start = None

    for num_tokens in range(1, seq_len + 1):
        # Get the last num_tokens tokens
        suffix_tokens = tokens[-num_tokens:]
        suffix_str = tokenizer.convert_tokens_to_string(suffix_tokens)

        # Check if the assistant content is contained in this suffix
        if last_assistant_content in suffix_str:
            # The assistant response starts at position (seq_len - num_tokens)
            assistant_start = seq_len - num_tokens
            break

    assert assistant_start is not None, (
        f"Could not find assistant response start for {prompt.country}. "
        f"Looking for: {last_assistant_content!r}"
    )

    assistant_end = seq_len  # Last token
    pre_answer = seq_len - 1  # Token right before where answer would be

    # Find first mention of country by detokenizing from the start
    # We check if the country name is contained in the prefix string
    country_first_pos = None

    for num_tokens in range(1, seq_len + 1):
        # Get the first num_tokens tokens
        prefix_tokens = tokens[:num_tokens]
        prefix_str = tokenizer.convert_tokens_to_string(prefix_tokens)

        # Check if the country name is contained in this prefix
        if country.lower() in prefix_str.lower():
            # The first mention is at position (num_tokens - 1)
            country_first_pos = num_tokens - 1
            break

    assert country_first_pos is not None, (
        f"Could not find first mention of country '{country}' in prompt for {prompt.country}. "
        f"Looking for: {country!r}"
    )

    # Find all country token ranges by tokenizing the country name and searching
    # We tokenize with a space prefix to handle subword tokenization properly
    country_with_space = f" {country}"
    country_tokens = tokenizer.encode(country_with_space, add_special_tokens=False)

    # Also try without space prefix for cases where country starts a segment
    country_tokens_no_space = tokenizer.encode(country, add_special_tokens=False)

    country_token_ranges: list[tuple[int, int]] = []

    # Search for country token sequences in the prompt
    i = 0
    while i < seq_len:
        # Try matching with space-prefixed tokens first
        match_len = 0
        for ct_list in [country_tokens, country_tokens_no_space]:
            if (
                i + len(ct_list) <= seq_len
                and token_ids_list[i : i + len(ct_list)] == ct_list
            ):
                match_len = len(ct_list)
                break

        if match_len > 0:
            country_token_ranges.append((i, i + match_len))
            i += match_len
        else:
            i += 1

    # Create and attach the token info
    prompt_with_token_info = replace(
        prompt,
        token_info=CountryPromptTokenInfo(
            pre_answer_pos=pre_answer,
            assistant_start_pos=assistant_start,
            assistant_end_pos=assistant_end,
            country_first_mention_pos=country_first_pos,
            country_token_ranges=tuple(country_token_ranges),
        ),
    )

    return prompt_with_token_info


@functools.cache
def generate_prompts(
    tokenizer: PreTrainedTokenizerBase,
    templates: frozenset[tuple[frozendict[str, str], ...]] | None = None,
) -> set[CountryPrompt]:
    """
    Generate country-capital prompts with token info populated.

    Args:
        tokenizer: The tokenizer to use for formatting and tokenization
        templates: Set of prompt templates to use. Defaults to PROMPT_TEMPLATES.

    Returns:
        Set of CountryPrompt objects with token_info populated
    """
    if templates is None:
        templates = frozenset(PROMPT_TEMPLATES)

    logger.info(
        f"Generating prompts for all countries with {len(templates)} templates..."
    )
    all_prompts: set[CountryPrompt] = set()

    for country, capital in tqdm(
        COUNTRY_TO_CAPITAL.items(),
        desc="Generating prompts for all countries",
        total=len(COUNTRY_TO_CAPITAL),
        leave=False,
        position=1,
    ):
        for template in tqdm(
            templates,
            desc=f"Generating prompts for {country}",
            total=len(templates),
            leave=False,
            position=0,
        ):
            # Compute template hash from the original template (before substitution)
            template_hash_val = hash(template)

            # Format messages with country
            messages: tuple[frozendict[str, str], ...] = deepfreeze(
                [
                    {
                        "role": msg["role"],
                        "content": msg["content"].format(country=country),
                    }
                    for msg in template
                ]
            )

            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )

            assert isinstance(formatted, str), (
                f"Expected formatted to be a string, got {type(formatted)}"
            )

            # Tokenize
            token_ids = tokenizer(formatted, return_tensors="pt").input_ids[0]

            prompt = CountryPrompt(
                country=country,
                capital=capital,
                messages=messages,
                formatted_text=formatted,
                token_ids=token_ids,
                template_hash=template_hash_val,
            )

            # Populate token info (including country token ranges)
            prompt_with_token_info = find_token_positions(prompt, tokenizer, country)

            all_prompts.add(prompt_with_token_info)

    logger.success(f"Generated {len(all_prompts)} prompts")
    return all_prompts


@dataclass(frozen=True)
class PerTokenRouterPath:
    """Router path for relevant token positions of a specific prompt."""

    prompt: CountryPrompt  # The specific prompt this path is for
    experiment_type: ExperimentType  # Which experiment type this path is for
    # Range of token positions this path covers (relative to prompt's token_ids)
    start_pos: int
    end_pos: int  # exclusive
    # Router path for the relevant token range only
    # Shape: (end_pos - start_pos, L, E) - compact, no zeros
    paths: th.Tensor


@th.no_grad()
def extract_router_paths(
    model: StandardizedTransformer,
    top_k: int,
    batch_size: int = 64,
    postprocessor: RouterLogitsPostprocessor = RouterLogitsPostprocessor.MASKS,
    templates: frozenset[tuple[frozendict[str, str], ...]] | None = None,
) -> set[PerTokenRouterPath]:
    """
    Extract router paths for all prompts in batches, keeping per-token paths.

    Each prompt gets its own PerTokenRouterPath for each experiment type.
    Paths are stored with full token dimension (T, L, E) matching the prompt's
    sequence length. Only positions relevant to the experiment type have non-zero values.

    Args:
        model: The MoE model
        top_k: Number of top experts
        batch_size: Number of prompts to process at once
        postprocessor: How to process router logits
        templates: Set of prompt templates to use. Defaults to PROMPT_TEMPLATES.

    Returns:
        Set of PerTokenRouterPath objects, one per (prompt, experiment_type) pair.
    """
    prompts = generate_prompts(model.tokenizer, templates)

    postprocessor_fn = get_postprocessor(postprocessor)

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    assert all(p.token_info is not None for p in prompts), (
        "All prompts must have token info"
    )

    all_paths: set[PerTokenRouterPath] = set()

    # Sort prompts by sequence length for more efficient batching
    sorted_prompts = sorted(prompts, key=lambda p: len(p.token_ids))

    batches = list(batched(sorted_prompts, batch_size))

    for batch_prompts in tqdm(
        batches, desc="Extracting router paths", total=len(batches)
    ):
        # Get sequence lengths and find max length in batch
        seq_lengths = [len(p.token_ids) for p in batch_prompts]
        max_seq_len = max(seq_lengths)

        attn_mask = th.ones(
            (len(batch_prompts), max_seq_len),
            dtype=th.bool,
            device=batch_prompts[0].token_ids.device,
        )

        # Left pad sequences to max length and stack into batch
        padded_tokens = []
        for sample_idx, prompt in enumerate(batch_prompts):
            tokens = prompt.token_ids
            padding_amt = max_seq_len - len(tokens)

            if padding_amt > 0:
                padding = th.full(
                    (padding_amt,),
                    pad_token_id,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = th.cat([padding, tokens])
                attn_mask[sample_idx, :padding_amt] = False

            padded_tokens.append(tokens)

        # Stack into (B, T)
        batch_token_ids = th.stack(padded_tokens, dim=0)
        batch = {
            "input_ids": batch_token_ids,
            "attention_mask": attn_mask,
        }

        router_logits_list = []

        with model.trace(batch):
            for layer_idx in model.layers_with_routers:
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats
                if isinstance(router_output, tuple):
                    if len(router_output) == 2:
                        router_scores, _router_indices = router_output
                    else:
                        raise ValueError(
                            f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                        )
                else:
                    router_scores = router_output

                logits = router_scores.save()
                router_logits_list.append(logits.reshape(*batch_token_ids.shape, -1))

        # Stack into (B, T, L, E)
        router_logits = th.stack(router_logits_list, dim=2)

        # Apply postprocessor to get paths
        router_paths_batch = postprocessor_fn(router_logits, top_k)  # (B, T, L, E)
        router_paths_batch = router_paths_batch.cpu()

        # Process each prompt in the batch
        for prompt_idx, (seq_len, prompt) in enumerate(
            zip(seq_lengths, batch_prompts, strict=False)
        ):
            token_info = prompt.token_info

            # Get paths for this prompt (excluding batch padding, keeping only actual tokens)
            # Shape: (seq_len, L, E) - matches this prompt's sequence length
            router_paths = router_paths_batch[prompt_idx, -seq_len:]

            # For each experiment type, store ONLY the relevant token range (compact storage)

            # 1. Pre-answer token (last token only)
            pre_answer_start = token_info.pre_answer_pos
            pre_answer_end = token_info.pre_answer_pos + 1
            all_paths.add(
                PerTokenRouterPath(
                    prompt=prompt,
                    experiment_type=ExperimentType.PRE_ANSWER,
                    start_pos=pre_answer_start,
                    end_pos=pre_answer_end,
                    paths=router_paths[pre_answer_start:pre_answer_end].clone(),
                )
            )

            # 2. All tokens in assistant response
            asst_start = token_info.assistant_start_pos
            asst_end = token_info.assistant_end_pos
            all_paths.add(
                PerTokenRouterPath(
                    prompt=prompt,
                    experiment_type=ExperimentType.ASSISTANT_RESPONSE,
                    start_pos=asst_start,
                    end_pos=asst_end,
                    paths=router_paths[asst_start:asst_end].clone(),
                )
            )

            # 3. All tokens from first country mention onwards
            from_country_start = token_info.country_first_mention_pos
            from_country_end = seq_len
            all_paths.add(
                PerTokenRouterPath(
                    prompt=prompt,
                    experiment_type=ExperimentType.FROM_COUNTRY_MENTION,
                    start_pos=from_country_start,
                    end_pos=from_country_end,
                    paths=router_paths[from_country_start:from_country_end].clone(),
                )
            )

        # Clean up batch tensors
        del router_logits, router_paths_batch
        gc.collect()
        th.cuda.empty_cache()

    return all_paths


def _align_path_by_token_ids(
    target_prompt: CountryPrompt,
    target_start: int,
    target_end: int,
    other_prompt: CountryPrompt,
    other_path: PerTokenRouterPath,
) -> th.Tensor:
    """
    Align other_path to match target_prompt's token positions using token ID matching.

    This function handles different country name token lengths by:
    1. Matching token IDs before the first country mention (should be identical)
    2. For country tokens: average the other country's tokens and broadcast
    3. Matching token IDs after the last country mention (should be identical with offset)

    All indexing is relative to the compact path ranges:
    - result indices are relative to target_start
    - other_path.paths indices are relative to other_path.start_pos

    Args:
        target_prompt: The target prompt
        target_start: Start position of relevant range in target (absolute)
        target_end: End position of relevant range in target (absolute)
        other_prompt: The other prompt to align from
        other_path: The path from other_prompt

    Returns:
        Aligned path tensor of shape (target_end - target_start, L, E)
    """
    target_country_ranges = target_prompt.token_info.country_token_ranges
    other_country_ranges = other_prompt.token_info.country_token_ranges

    assert len(target_country_ranges) == len(other_country_ranges), (
        f"Target country ranges count ({len(target_country_ranges)}) "
        f"does not match other country ranges count ({len(other_country_ranges)})!"
    )
    assert len(target_country_ranges) > 0, "Country ranges are empty!"

    num_layers = other_path.paths.shape[1]
    num_experts = other_path.paths.shape[2]

    result_len = target_end - target_start
    result = th.empty(
        result_len,
        num_layers,
        num_experts,
        dtype=other_path.paths.dtype,
        device=other_path.paths.device,
    )

    # Track positions in the COMPACT tensors (relative to start positions)
    # last_target_rel: last processed position in result (relative to 0)
    # last_other_rel: last processed position in other_path.paths (relative to 0)
    last_target_rel = 0
    last_other_rel = 0

    # Compute cumulative offset from country token length differences
    cumulative_offset = 0

    for target_range, other_range in zip(
        target_country_ranges, other_country_ranges, strict=False
    ):
        target_country_start_abs, target_country_end_abs = target_range
        other_country_start_abs, other_country_end_abs = other_range

        target_country_len = target_country_end_abs - target_country_start_abs
        other_country_len = other_country_end_abs - other_country_start_abs

        # Skip country ranges that are entirely outside our intervention range
        if target_country_end_abs <= target_start:
            cumulative_offset += target_country_len - other_country_len
            continue
        if target_country_start_abs >= target_end:
            break

        # Convert to relative positions for our compact tensors
        # Clamp to the intervention range
        target_country_start_rel = max(0, target_country_start_abs - target_start)
        target_country_end_rel = min(result_len, target_country_end_abs - target_start)

        other_country_start_rel = max(0, other_country_start_abs - other_path.start_pos)
        other_country_end_rel = min(
            other_path.paths.shape[0], other_country_end_abs - other_path.start_pos
        )

        # Copy tokens BEFORE this country mention (non-country tokens match 1:1)
        if target_country_start_rel > last_target_rel:
            copy_len = target_country_start_rel - last_target_rel
            result[last_target_rel:target_country_start_rel] = other_path.paths[
                last_other_rel : last_other_rel + copy_len
            ]
            last_other_rel += copy_len

        # Handle country tokens: average and broadcast
        if other_country_end_rel > other_country_start_rel:
            other_country_paths = other_path.paths[
                other_country_start_rel:other_country_end_rel
            ]
            other_country_avg = other_country_paths.mean(dim=0, keepdim=True)

            # Broadcast to target country length
            target_country_actual_len = (
                target_country_end_rel - target_country_start_rel
            )
            result[target_country_start_rel:target_country_end_rel] = (
                other_country_avg.expand(target_country_actual_len, -1, -1)
            )

        last_target_rel = target_country_end_rel
        last_other_rel = other_country_end_rel
        cumulative_offset += target_country_len - other_country_len

    # Copy remaining tokens after the last country mention
    if last_target_rel < result_len:
        remaining_len = result_len - last_target_rel
        result[last_target_rel:] = other_path.paths[
            last_other_rel : last_other_rel + remaining_len
        ]

    return result


def compute_country_interventions(
    all_paths: set[PerTokenRouterPath],
) -> dict[str, dict[ExperimentType, th.Tensor]]:
    """
    Compute country-level interventions (not per-prompt).

    For each country, computes: mean(target_country) - mean(other_countries)
    where each path is first averaged over the token dimension.

    This is the original intervention approach: one (L, E) intervention per country,
    which gets expanded when applied to specific prompts.

    Args:
        all_paths: Set of all PerTokenRouterPath objects

    Returns:
        Dictionary mapping country -> experiment_type -> intervention tensor (L, E)
    """
    # Group paths by country and experiment type
    # country -> experiment_type -> list of (L, E) averaged paths
    country_paths: dict[str, dict[ExperimentType, list[th.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path_obj in all_paths:
        country = path_obj.prompt.country
        exp_type = path_obj.experiment_type
        # Average over token dimension: (T, L, E) -> (L, E)
        path_avg = path_obj.paths.mean(dim=0)
        country_paths[country][exp_type].append(path_avg)

    # Compute average path for each country and experiment type
    # country -> experiment_type -> (L, E)
    country_avgs: dict[str, dict[ExperimentType, th.Tensor]] = {}
    for country, exp_dict in country_paths.items():
        country_avgs[country] = {}
        for exp_type, paths in exp_dict.items():
            # Average across prompts: (N, L, E) -> (L, E)
            country_avgs[country][exp_type] = th.stack(paths, dim=0).mean(dim=0)

    # Compute interventions: mean(target_country) - mean(others)
    interventions: dict[str, dict[ExperimentType, th.Tensor]] = defaultdict(dict)
    all_countries = set(country_avgs.keys())

    for target_country in all_countries:
        for exp_type in country_avgs[target_country]:
            # Collect averages from other countries
            other_avgs = [
                country_avgs[c][exp_type]
                for c in all_countries
                if c != target_country and exp_type in country_avgs[c]
            ]

            if not other_avgs:
                continue

            # Average across other countries: (N, L, E) -> (L, E)
            others_mean = th.stack(other_avgs, dim=0).mean(dim=0)

            # Intervention: target - others (what's unique to target country)
            target_mean = country_avgs[target_country][exp_type]
            interventions[target_country][exp_type] = target_mean - others_mean

    return dict(interventions)


def compute_prompt_specific_intervention(
    target_path: PerTokenRouterPath,
    all_paths: set[PerTokenRouterPath],
) -> th.Tensor:
    """
    Compute the intervention path for a specific prompt using same-template comparison.

    Compares against paths from the same template but different countries,
    using token-ID alignment to handle different country name token lengths.

    Args:
        target_path: The PerTokenRouterPath for the target prompt
        all_paths: Set of all PerTokenRouterPath objects

    Returns:
        Tensor of shape (end_pos - start_pos, L, E) representing the per-token
        intervention path for the relevant token range of this prompt.
    """
    target_prompt = target_path.prompt
    target_country = target_prompt.country
    target_exp_type = target_path.experiment_type
    target_template_hash = target_prompt.template_hash

    # Collect paths from the SAME TEMPLATE but DIFFERENT COUNTRIES
    other_paths: list[PerTokenRouterPath] = [
        p
        for p in all_paths
        if (
            p.prompt.template_hash == target_template_hash
            and p.prompt.country != target_country
            and p.experiment_type == target_exp_type
        )
    ]

    assert other_paths, (
        f"No other countries found for experiment type {target_exp_type}"
    )

    # Align each other path to target's token positions
    aligned_other_paths: list[th.Tensor] = []

    for other_path in other_paths:
        aligned = _align_path_by_token_ids(
            target_prompt=target_prompt,
            target_start=target_path.start_pos,
            target_end=target_path.end_pos,
            other_prompt=other_path.prompt,
            other_path=other_path,
        )
        aligned_other_paths.append(aligned)

    # Average across other countries: (N, T_range, L, E) -> (T_range, L, E)
    other_avg = th.stack(aligned_other_paths, dim=0).mean(dim=0)

    # Compute per-token intervention: target - average(others)
    intervention = target_path.paths - other_avg

    return intervention


@th.no_grad()
def run_intervention(
    prompts: list[CountryPrompt],
    model: StandardizedTransformer,
    intervention_paths: dict[
        CountryPrompt, dict[ExperimentType, tuple[int, int, th.Tensor]]
    ],  # prompt -> experiment_type -> (start_pos, end_pos, compact intervention path)
    alpha: float,
    top_k: int,
) -> tuple[list[float], dict[ExperimentType, list[float]]]:
    """
    Run the model with and without intervention for multiple prompts.

    Each prompt has its own compact per-token intervention path for each experiment type.
    The intervention path is (end_pos - start_pos, L, E) covering only the relevant tokens.
    Zeros are at positions where no intervention should be applied.

    The intervention works by:
    1. Getting the original router logits
    2. Expanding compact paths to full sequence length
    3. Subtracting the scaled per-token intervention path directly
    4. Re-normalizing and applying top-k selection

    Args:
        prompts: List of prompts to run
        model: The model
        intervention_paths: Mapping from prompt -> experiment_type -> (start, end, path)
            where path has shape (end - start, L, E)
        alpha: Scaling factor for the intervention
        top_k: Number of top experts to select

    Returns:
        Tuple of (pre_intervention_probs, dictionary mapping experiment type to post_intervention_probs)
        where each list has one probability per prompt in the same order as input prompts
    """
    # Get experiment types from the first prompt's intervention paths
    first_prompt = prompts[0]
    experiments = list(intervention_paths[first_prompt].keys())
    num_experiments = len(experiments) + 1  # + 1 for the control
    num_prompts = len(prompts)

    # Get the token ID for the capital (first token) for each prompt
    capital_token_ids: th.Tensor = th.empty(num_prompts, dtype=th.long)
    for prompt_idx, prompt in enumerate(prompts):
        capital_tokens = model.tokenizer(
            f" {prompt.capital}", add_special_tokens=False
        ).input_ids
        assert isinstance(capital_tokens, list) and capital_tokens, (
            f"Capital '{prompt.capital}' not found in tokenizer"
        )
        capital_first_token_id = capital_tokens[0]
        assert isinstance(capital_first_token_id, int), (
            f"Capital '{prompt.capital}' first token ID is not an integer"
        )
        capital_token_ids[prompt_idx] = capital_first_token_id

    # Pad prompts to the same length (left padding)
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    seq_lengths = [len(p.token_ids) for p in prompts]
    max_seq_len = max(seq_lengths)

    # Left-pad sequences and create attention mask
    padded_tokens: list[th.Tensor] = []
    prompt_attn_mask = th.ones(  # (P, N, T)
        (num_prompts, num_experiments, max_seq_len),
        dtype=th.bool,
        device=prompts[0].token_ids.device,
    )

    for prompt_idx, prompt in enumerate(prompts):
        tokens = prompt.token_ids
        padding_amt = max_seq_len - len(tokens)

        if padding_amt > 0:
            padding = th.full(
                (padding_amt,),
                pad_token_id,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = th.cat([padding, tokens])
            prompt_attn_mask[prompt_idx, :, :padding_amt] = False

        padded_tokens.append(tokens)

    # Stack into (P, T)
    prompt_token_ids_single = th.stack(padded_tokens, dim=0)

    # Expand for all experiments: (P, T) -> (P, N, T)
    prompt_token_ids = prompt_token_ids_single.unsqueeze(1).repeat(
        1, num_experiments, 1
    )  # (P, N, T)

    # Build the per-prompt, per-experiment intervention tensor: (P, N, T, L, E)
    # Each prompt's compact intervention path is (end_pos - start_pos, L, E)
    # We need to expand it to full sequence length and left-pad to max_seq_len
    _, _, first_intervention = intervention_paths[first_prompt][experiments[0]]
    num_layers = first_intervention.shape[1]
    num_experts = first_intervention.shape[2]

    prompt_intervention_paths_tensor = th.zeros(
        (num_prompts, num_experiments, max_seq_len, num_layers, num_experts),
        dtype=th.float32,
        device=prompts[0].token_ids.device,
    )

    for prompt_idx, prompt in enumerate(prompts):
        seq_len = seq_lengths[prompt_idx]
        padding_amt = max_seq_len - seq_len

        for exp_idx, exp_type in enumerate(experiments):
            # Get the compact per-token intervention path for this prompt and experiment
            start_pos, end_pos, compact_path = intervention_paths[prompt][
                exp_type
            ]  # (end-start, L, E)

            # Calculate the position in the padded tensor
            # Original position in unpadded: [start_pos, end_pos)
            # Position in padded: [padding_amt + start_pos, padding_amt + end_pos)
            padded_start = padding_amt + start_pos
            padded_end = padding_amt + end_pos

            # Place the compact path at the correct position
            prompt_intervention_paths_tensor[
                prompt_idx, exp_idx, padded_start:padded_end
            ] = compact_path.to(
                device=prompt_intervention_paths_tensor.device, dtype=th.float32
            )

        # Control experiment (last one) stays all zeros

    # Run with interventions using nnterp's router_probabilities mechanism
    layers_with_routers = list(model.layers_with_routers)

    token_ids = prompt_token_ids.view(-1, max_seq_len)  # (P*N, T)
    attn_mask = prompt_attn_mask.view(-1, max_seq_len)  # (P*N, T)
    batch = {
        "input_ids": token_ids,
        "attention_mask": attn_mask,
    }
    # Reshape to (P*N, T, L, E)
    intervention_paths_tensor = prompt_intervention_paths_tensor.view(
        -1, max_seq_len, num_layers, num_experts
    )

    with model.trace(batch):
        for i, layer_idx in enumerate(layers_with_routers):
            router_output = model.routers_output[layer_idx]

            # Handle different router output formats
            router_output_is_tuple = isinstance(router_output, tuple)
            router_output_len = len(router_output)
            if router_output_is_tuple:
                if len(router_output) == 2:
                    router_scores, _router_indices = router_output
                    raise ValueError(
                        "Cannot run this experiment on a model whose routers do not return raw logits"
                    )
                elif len(router_output) == 3:
                    (
                        original_router_logits,
                        original_router_weights,
                        original_router_indices,
                    ) = router_output

                    router_logits = cast("th.Tensor", original_router_logits.save())
                    router_logits = router_logits.reshape(*token_ids.shape, -1)
                    router_weights = cast("th.Tensor", original_router_weights.save())
                    router_weights = router_weights.reshape(*token_ids.shape, -1)
                    router_indices = cast("th.Tensor", original_router_indices.save())
                    router_indices = router_indices.reshape(*token_ids.shape, -1)

                    assert router_logits.shape[-1] > top_k, (
                        f"Expected router logits to have shape (P*N, T, >{top_k}), got {router_logits.shape}"
                    )
                    assert router_weights.shape[-1] == top_k, (
                        f"Expected router weights to have shape (P*N, T, {top_k}), got {router_weights.shape}"
                    )
                    assert router_indices.shape[-1] == top_k, (
                        f"Expected router indices to have shape (P*N, T, {top_k}), got {router_indices.shape}"
                    )
                    assert th.allclose(router_weights.sum(dim=-1), 1.0), (
                        "Router weights must sum to 1.0"
                    )
                    assert th.all(router_weights >= 0), (
                        "Router weights must be non-negative"
                    )
                else:
                    raise ValueError(
                        f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                    )
            else:
                router_scores = cast("th.Tensor", router_output.save())
                router_logits = router_scores.reshape(
                    *token_ids.shape, -1
                )  # (P*N, T, E)

            # Apply per-token intervention directly (no mask needed)
            # intervention_paths_tensor is (P*N, T, L, E), we need layer i: (P*N, T, E)
            intervention_paths_tensor = intervention_paths_tensor.to(
                device=router_logits.device
            )
            layer_intervention = intervention_paths_tensor[:, :, i, :]  # (P*N, T, E)

            modified_logits = router_logits.clone()
            modified_logits -= alpha * layer_intervention

            if router_output_is_tuple:
                if router_output_len == 3:
                    # Recompute topk for all positions
                    all_new_weights, all_new_indices = th.topk(
                        modified_logits, k=top_k, dim=-1
                    )  # (P*N, T, top_k)
                    all_new_weights = F.softmax(
                        all_new_weights, dim=-1, dtype=all_new_weights.dtype
                    )

                    # Update weights and indices where intervention is non-zero
                    # Since intervention_paths already has zeros at non-intervention positions,
                    # we can check if any expert has non-zero intervention to create a mask
                    intervention_nonzero = (
                        layer_intervention.abs().sum(dim=-1) > 0
                    )  # (P*N, T)
                    mask_topk = intervention_nonzero.unsqueeze(-1).expand_as(
                        router_weights
                    )  # (P*N, T, top_k)

                    modified_weights = th.where(
                        mask_topk, all_new_weights, router_weights
                    )
                    modified_indices = th.where(
                        mask_topk, all_new_indices, router_indices
                    )

                    model.routers_output[layer_idx] = (
                        modified_logits.reshape(-1, modified_logits.shape[-1]),
                        modified_weights.reshape(-1, modified_weights.shape[-1]),
                        modified_indices.reshape(-1, modified_indices.shape[-1]),
                    )
                else:
                    raise ValueError(
                        f"Found tuple of length {len(router_output)} for router output at layer {layer_idx}"
                    )
            else:
                model.routers_output[layer_idx] = modified_logits.reshape(
                    -1, modified_logits.shape[-1]
                )

        final_logits = model.lm_head.output.save()  # (P*N, T, vocab_size)

    # Extract control probabilities (last experiment, i.e., every num_experiments-th element)
    final_prompt_logits = final_logits.view(
        num_prompts, num_experiments, -1, final_logits.shape[-1]
    )  # (P, N, T, vocab_size)
    final_prompt_probs = F.softmax(final_prompt_logits.float(), dim=-1)

    # Extract pre-intervention probabilities for each prompt's capital
    capital_token_ids = capital_token_ids.to(
        device=final_prompt_probs.device
    ).unsqueeze(-1)
    pre_probs_tensor = final_prompt_probs[:, -1, -1].gather(
        dim=-1,
        index=capital_token_ids,
    )
    pre_probs: list[float] = pre_probs_tensor.cpu().squeeze(-1).tolist()  # (P,)

    # Extract post-intervention probabilities for each experiment type
    experiment_post_probs: dict[ExperimentType, list[float]] = {}

    for experiment_idx, experiment_type in enumerate(experiments):
        post_probs_tensor = final_prompt_probs[:, experiment_idx, -1].gather(
            dim=-1,
            index=capital_token_ids,
        )
        post_probs: list[float] = post_probs_tensor.cpu().squeeze(-1).tolist()  # (P,)
        experiment_post_probs[experiment_type] = post_probs

    return pre_probs, experiment_post_probs


def run_intervention_experiment(
    model: StandardizedTransformer,
    all_paths: set[PerTokenRouterPath],
    alphas: set[float],
    top_k: int,
    batch_size: int = 64,
    templates: frozenset[tuple[frozendict[str, str], ...]] | None = None,
    m: int = 0,
    use_same_template_only: bool = True,
) -> frozendict[ExperimentType, tuple[ExperimentResults, ...]]:
    """
    Run the full intervention experiment across multiple alpha values.

    Two modes are supported:
    1. use_same_template_only=True (per-prompt): Each prompt gets its own per-token
       intervention path computed by comparing against same-template, different-country
       paths with token-ID alignment.
    2. use_same_template_only=False (per-country): One intervention per country,
       computed as mean(target_country) - mean(others), then expanded to each prompt.

    Args:
        model: The MoE model
        all_paths: Set of PerTokenRouterPath objects for all prompts
        alphas: Set of alpha values to test
        top_k: Number of top experts to select
        batch_size: Number of prompts to process at once
        templates: Set of prompt templates to use. Defaults to PROMPT_TEMPLATES.
        m: Number of top experts to keep in intervention path (0 = use all experts)
        use_same_template_only: If True, use per-prompt same-template comparison.
            If False, use per-country averaged comparison.

    Returns:
        Dictionary mapping experiment type to set of ExperimentResults
    """
    prompts = generate_prompts(model.tokenizer, templates)
    num_templates = len(PROMPT_TEMPLATES) if templates is None else len(templates)

    # Pre-compute intervention paths for all prompts
    # Organized as: prompt -> experiment_type -> (start_pos, end_pos, intervention tensor)
    prompt_interventions: dict[
        CountryPrompt, dict[ExperimentType, tuple[int, int, th.Tensor]]
    ] = defaultdict(dict)

    if use_same_template_only:
        # Per-prompt mode: compute intervention for each prompt individually
        logger.info(
            "Pre-computing per-prompt intervention paths (same-template mode)..."
        )
        for path_obj in tqdm(
            all_paths, desc="Computing interventions", total=len(all_paths)
        ):
            intervention = compute_prompt_specific_intervention(path_obj, all_paths)

            # Apply top-m filtering if requested
            if m > 0:
                intervention = apply_top_m_filtering(intervention, m)

            # Store with position info for later expansion
            prompt_interventions[path_obj.prompt][path_obj.experiment_type] = (
                path_obj.start_pos,
                path_obj.end_pos,
                intervention,
            )
    else:
        # Per-country mode: compute one intervention per country, expand to prompts
        logger.info("Pre-computing country-level intervention paths...")
        country_interventions = compute_country_interventions(all_paths)

        # Apply top-m filtering to country interventions if requested
        if m > 0:
            for country in country_interventions:
                for exp_type in country_interventions[country]:
                    country_interventions[country][exp_type] = apply_top_m_filtering(
                        country_interventions[country][exp_type], m
                    )

        # Expand country interventions to each prompt
        logger.info("Expanding country interventions to prompts...")
        for path_obj in tqdm(
            all_paths, desc="Expanding to prompts", total=len(all_paths)
        ):
            country = path_obj.prompt.country
            exp_type = path_obj.experiment_type

            if country not in country_interventions:
                continue
            if exp_type not in country_interventions[country]:
                continue

            # Get country-level intervention: (L, E)
            country_intervention = country_interventions[country][exp_type]

            # Expand to prompt's token range: (L, E) -> (T_range, L, E)
            range_len = path_obj.end_pos - path_obj.start_pos
            expanded = country_intervention.unsqueeze(0).expand(range_len, -1, -1)

            prompt_interventions[path_obj.prompt][exp_type] = (
                path_obj.start_pos,
                path_obj.end_pos,
                expanded,
            )

    all_results: dict[ExperimentType, set[InterventionResult]] = defaultdict(set)

    for alpha in tqdm(
        alphas, desc="Testing alpha values", total=len(alphas), position=3
    ):
        # Process prompts in batches
        prompts_list = list(prompts)
        prompt_batches = list(batched(prompts_list, batch_size))

        for batch_prompts in tqdm(
            prompt_batches,
            desc="Testing prompt batches",
            total=len(prompt_batches),
            position=1,
            leave=False,
        ):
            batch_prompts_list = list(batch_prompts)

            # Build intervention paths dict for this batch
            batch_intervention_paths: dict[
                CountryPrompt, dict[ExperimentType, tuple[int, int, th.Tensor]]
            ] = {prompt: prompt_interventions[prompt] for prompt in batch_prompts_list}

            pre_probs, post_probs_dict = run_intervention(
                batch_prompts_list, model, batch_intervention_paths, alpha, top_k
            )

            # Process results for each prompt in the batch
            for prompt_idx, prompt in enumerate(batch_prompts_list):
                pre_prob = pre_probs[prompt_idx]
                for experiment_type, post_probs in post_probs_dict.items():
                    post_prob = post_probs[prompt_idx]
                    # Calculate normalized forgetfulness: (pre - post) / pre
                    # 1.0 = full forgetting, 0.0 = no change, -1.0 = doubling
                    normalized_forgetfulness = (
                        (pre_prob - post_prob) / pre_prob if pre_prob > 0 else 0.0
                    )
                    # For per-prompt interventions, the "intervention_country" is the prompt's own country
                    all_results[experiment_type].add(
                        InterventionResult(
                            country=prompt.country,
                            intervention_country=prompt.country,
                            template_hash=prompt.template_hash,
                            pre_intervention_prob=pre_prob,
                            post_intervention_prob=post_prob,
                            forgetfulness=InterventionMetric(
                                alpha=alpha, value=normalized_forgetfulness
                            ),
                        )
                    )

    # Structure results by country
    # In the per-prompt intervention approach, each prompt intervenes on itself
    # So we group by country and compute average forgetfulness across templates
    structured_results: dict[ExperimentType, set[ExperimentResults]] = defaultdict(set)

    for experiment_type, results in tqdm(
        all_results.items(),
        desc="Structuring results by experiment type",
        total=len(all_results),
        position=2,
    ):
        for target_country in tqdm(
            COUNTRY_TO_CAPITAL,
            desc=f"Structuring results by target country for {experiment_type.value}",
            total=len(COUNTRY_TO_CAPITAL),
            position=1,
            leave=False,
        ):
            # Get all results for this country (each prompt intervenes on itself)
            country_results = {
                result for result in results if result.country == target_country
            }

            # Get results for other countries (for comparison/specificity)
            other_country_results = {
                result for result in results if result.country != target_country
            }

            target_results_averaged = set()
            other_results_averaged = set()
            specificity_scores = set()

            for alpha in alphas:
                # Target country results for this alpha
                target_results_for_alpha = {
                    result
                    for result in country_results
                    if result.forgetfulness.alpha == alpha
                }

                assert len(target_results_for_alpha) == num_templates, (
                    f"Expected {num_templates} target results for alpha {alpha}, "
                    f"got {len(target_results_for_alpha)}"
                )

                avg_target_pre = sum(
                    r.pre_intervention_prob for r in target_results_for_alpha
                ) / len(target_results_for_alpha)
                avg_target_post = sum(
                    r.post_intervention_prob for r in target_results_for_alpha
                ) / len(target_results_for_alpha)
                avg_target_forgetfulness = (
                    (avg_target_pre - avg_target_post) / avg_target_pre
                    if avg_target_pre > 0
                    else 0.0
                )

                target_results_averaged.add(
                    InterventionResult(
                        country=target_country,
                        intervention_country=target_country,
                        template_hash=0,  # Averaged result, no specific template
                        pre_intervention_prob=avg_target_pre,
                        post_intervention_prob=avg_target_post,
                        forgetfulness=InterventionMetric(
                            alpha=alpha, value=avg_target_forgetfulness
                        ),
                    )
                )

                # Other countries' results for this alpha (for specificity comparison)
                other_results_for_alpha = {
                    result
                    for result in other_country_results
                    if result.forgetfulness.alpha == alpha
                }

                avg_other_pre = sum(
                    r.pre_intervention_prob for r in other_results_for_alpha
                ) / len(other_results_for_alpha)
                avg_other_post = sum(
                    r.post_intervention_prob for r in other_results_for_alpha
                ) / len(other_results_for_alpha)
                avg_other_forgetfulness = (
                    (avg_other_pre - avg_other_post) / avg_other_pre
                    if avg_other_pre > 0
                    else 0.0
                )

                other_results_averaged.add(
                    InterventionResult(
                        country="avg_others",
                        intervention_country="self",
                        template_hash=0,  # Averaged result, no specific template
                        pre_intervention_prob=avg_other_pre,
                        post_intervention_prob=avg_other_post,
                        forgetfulness=InterventionMetric(
                            alpha=alpha, value=avg_other_forgetfulness
                        ),
                    )
                )

                # Specificity: how much more does target country forget vs others
                specificity_scores.add(
                    InterventionMetric(
                        alpha=alpha,
                        value=avg_target_forgetfulness - avg_other_forgetfulness,
                    )
                )

            structured_results[experiment_type].add(
                ExperimentResults(
                    target_country=target_country,
                    target_results=tuple(target_results_averaged),
                    other_results=tuple(other_country_results),
                    other_results_averaged=tuple(other_results_averaged),
                    specificity_scores=tuple(specificity_scores),
                )
            )

    return deepfreeze(structured_results)


def _plot_single_country_results(
    results: ExperimentResults,
    experiment_type: ExperimentType,
    output_path: Path,
) -> None:
    """Create visualization for a single country's intervention results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract alphas and sort results by alpha
    alphas = sorted(score.alpha for score in results.specificity_scores)

    # Build forgetfulness values sorted by alpha
    target_forgetfulnesses: list[float] = []
    other_forgetfulnesses: list[float] = []
    specificity_scores: list[float] = []

    for alpha in alphas:
        # Get target forgetfulness for this alpha
        targets_for_alpha = {
            result
            for result in results.target_results
            if result.forgetfulness.alpha == alpha
        }
        assert len(targets_for_alpha) == 1, (
            f"Expected 1 target result for alpha {alpha}, got {len(targets_for_alpha)}"
        )
        target_forgetfulness = next(iter(targets_for_alpha)).forgetfulness.value
        target_forgetfulnesses.append(target_forgetfulness)

        # Get averaged other forgetfulness for this alpha
        other_results_averaged_for_alpha = [
            result
            for result in results.other_results_averaged
            if result.forgetfulness.alpha == alpha
        ]
        assert len(other_results_averaged_for_alpha) == 1, (
            f"Expected 1 other result averaged for alpha {alpha}, got {len(other_results_averaged_for_alpha)}"
        )
        other_forgetfulness = next(
            iter(other_results_averaged_for_alpha)
        ).forgetfulness.value
        other_forgetfulnesses.append(other_forgetfulness)

        # Get specificity score for this alpha
        specificity_scores_for_alpha = {
            result for result in results.specificity_scores if result.alpha == alpha
        }
        assert len(specificity_scores_for_alpha) == 1, (
            f"Expected 1 specificity score for alpha {alpha}, got {len(specificity_scores_for_alpha)}"
        )
        specificity_score = next(iter(specificity_scores_for_alpha)).value
        specificity_scores.append(specificity_score)

    target_capital = COUNTRY_TO_CAPITAL[results.target_country]

    # Plot 1: Forgetfulness by alpha
    ax1 = axes[0]
    ax1.plot(
        alphas,
        target_forgetfulnesses,
        "b-o",
        label=f"{results.target_country} (target)",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        alphas,
        other_forgetfulnesses,
        "r-s",
        label="Other countries (avg)",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Alpha (intervention strength)", fontsize=12)
    ax1.set_ylabel("Forgetfulness (normalized: (pre-post)/pre)", fontsize=12)
    ax1.set_title(
        f"Forgetfulness vs Intervention Strength\n({experiment_type.value})",
        fontsize=14,
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 2: Specificity score
    ax2 = axes[1]
    ax2.plot(alphas, specificity_scores, "g-^", linewidth=2, markersize=8)
    ax2.set_xlabel("Alpha (intervention strength)", fontsize=12)
    ax2.set_ylabel("Specificity (target - other forgetfulness)", fontsize=12)
    ax2.set_title(
        f"Specificity of {results.target_country}-{target_capital} Knowledge\n({experiment_type.value})",
        fontsize=14,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.fill_between(alphas, specificity_scores, alpha=0.3, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_average_results(
    all_results: set[ExperimentResults],
    experiment_type: ExperimentType,
    output_path: Path,
) -> None:
    """Create visualization of average intervention results across all countries."""
    if not all_results:
        logger.warning(f"No results to plot for {experiment_type.value}")
        return

    # Get alphas from the first result
    first_result = next(iter(all_results))
    alphas = sorted(score.alpha for score in first_result.specificity_scores)

    # Aggregate forgetfulness and specificity across all countries
    avg_target_forgetfulnesses: list[float] = []
    avg_other_forgetfulnesses: list[float] = []
    avg_specificity_scores: list[float] = []

    for alpha in alphas:
        target_forgetfulnesses: list[float] = []
        other_forgetfulnesses: list[float] = []
        specificity_scores: list[float] = []

        for result in all_results:
            # Target forgetfulness
            targets_for_alpha = {
                result
                for result in result.target_results
                if result.forgetfulness.alpha == alpha
            }
            assert len(targets_for_alpha) == 1, (
                f"Expected 1 target result for alpha {alpha}, got {len(targets_for_alpha)}"
            )
            target_forgetfulness = next(iter(targets_for_alpha)).forgetfulness.value
            target_forgetfulnesses.append(target_forgetfulness)

            # Other forgetfulness (averaged)
            other_results_averaged_for_alpha = [
                result
                for result in result.other_results_averaged
                if result.forgetfulness.alpha == alpha
            ]
            assert len(other_results_averaged_for_alpha) == 1, (
                f"Expected 1 other result averaged for alpha {alpha}, got {len(other_results_averaged_for_alpha)}"
            )
            other_forgetfulness = next(
                iter(other_results_averaged_for_alpha)
            ).forgetfulness.value
            other_forgetfulnesses.append(other_forgetfulness)

            # Specificity
            specificity_scores_for_alpha = {
                specificity_score
                for specificity_score in result.specificity_scores
                if specificity_score.alpha == alpha
            }
            assert len(specificity_scores_for_alpha) == 1, (
                f"Expected 1 specificity score for alpha {alpha}, got {len(specificity_scores_for_alpha)}"
            )
            specificity_score = next(iter(specificity_scores_for_alpha)).value
            specificity_scores.append(specificity_score)

        avg_target_forgetfulnesses.append(
            sum(target_forgetfulnesses) / len(target_forgetfulnesses)
        )
        avg_other_forgetfulnesses.append(
            sum(other_forgetfulnesses) / len(other_forgetfulnesses)
        )
        avg_specificity_scores.append(sum(specificity_scores) / len(specificity_scores))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average forgetfulness by alpha
    ax1 = axes[0]
    ax1.plot(
        alphas,
        avg_target_forgetfulnesses,
        "b-o",
        label="Target country (avg)",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        alphas,
        avg_other_forgetfulnesses,
        "r-s",
        label="Other countries (avg)",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Alpha (intervention strength)", fontsize=12)
    ax1.set_ylabel("Forgetfulness (normalized: (pre-post)/pre)", fontsize=12)
    ax1.set_title(
        f"Average Forgetfulness Across All Countries\n({experiment_type.value})",
        fontsize=14,
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 2: Average specificity score
    ax2 = axes[1]
    ax2.plot(alphas, avg_specificity_scores, "g-^", linewidth=2, markersize=8)
    ax2.set_xlabel("Alpha (intervention strength)", fontsize=12)
    ax2.set_ylabel("Specificity (target - other forgetfulness)", fontsize=12)
    ax2.set_title(
        f"Average Specificity Across All Countries\n({experiment_type.value})",
        fontsize=14,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.fill_between(alphas, avg_specificity_scores, alpha=0.3, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_results(
    results: dict[ExperimentType, set[ExperimentResults]],
    output_dir: Path,
) -> None:
    """
    Create visualizations for all intervention results.

    Generates:
    - A plot for each country and experiment type
    - An average plot across all countries for each experiment type
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment_type, experiment_results in tqdm(
        results.items(),
        desc="Plotting results by experiment type",
        total=len(results),
    ):
        exp_dir = output_dir / experiment_type.value
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Plot individual country results
        for country_result in tqdm(
            experiment_results,
            desc=f"Plotting {experiment_type.value} results",
            total=len(experiment_results),
            leave=False,
        ):
            country_file = (
                exp_dir
                / f"{country_result.target_country.lower().replace(' ', '_')}.png"
            )
            _plot_single_country_results(country_result, experiment_type, country_file)

        # Plot average results
        avg_file = exp_dir / "average.png"
        _plot_average_results(experiment_results, experiment_type, avg_file)

    logger.info(f"Saved all plots to {output_dir}")


def apply_top_m_filtering(
    intervention_path: th.Tensor,  # (T, L, E) or (L, E)
    m: int,
) -> th.Tensor:
    """
    Apply top-m filtering to an intervention path.

    At m=0, returns the original path unchanged.
    At m>0, keeps only the top m experts overall (across all tokens and layers,
    by absolute value) and zeros the rest.

    Args:
        intervention_path: The intervention path to filter (T, L, E) or (L, E)
        m: Number of top experts to keep overall. If 0, returns original path.

    Returns:
        Filtered intervention path with same shape as input
    """
    if m == 0:
        return intervention_path

    original_shape = intervention_path.shape

    # Flatten the path to select top m experts overall
    flattened_path = intervention_path.flatten()
    total_elements = flattened_path.numel()

    # Get top m by absolute value
    k = min(m, total_elements)
    _, top_indices = th.topk(th.abs(flattened_path), k=k, dim=-1)

    # Create a mask: True for top m, False for others
    mask = th.zeros(total_elements, dtype=th.bool, device=intervention_path.device)
    mask[top_indices] = True

    # Apply mask and reshape back to original shape
    filtered_flattened = flattened_path * mask.float()
    filtered_path = filtered_flattened.reshape(original_shape)

    return filtered_path


@th.no_grad()
def extract_topk_predictions_across_alphas(
    model: StandardizedTransformer,
    prompt: CountryPrompt,
    intervention_path_info: tuple[int, int, th.Tensor],  # (start, end, compact path)
    alphas: list[float],
    num_top_tokens: int,
    router_top_k: int,
    m: int = 0,
) -> list[TopKPrediction]:
    """
    Extract top-k token predictions at each alpha level for a single prompt.

    The intervention path is compact (end - start, L, E) and covers only
    the relevant token positions [start, end).

    Args:
        model: The MoE model
        prompt: The prompt to run
        intervention_path_info: Tuple of (start_pos, end_pos, compact_path)
            where compact_path has shape (end - start, L, E)
        alphas: List of alpha values to test
        num_top_tokens: Number of top tokens to extract
        router_top_k: Number of top experts per layer
        m: Number of top experts to keep in intervention path (0 = use all experts)

    Returns:
        List of TopKPrediction for each alpha
    """
    layers_with_routers = list(model.layers_with_routers)
    device = next(model.parameters()).device

    # Unpack compact path info
    start_pos, end_pos, compact_path = intervention_path_info

    logger.trace(f"Prompt: {_sanitize_string_for_display(prompt.formatted_text)}")
    logger.trace(f"Intervention path: {compact_path.shape} at [{start_pos}, {end_pos})")
    logger.trace(f"Alphas: {alphas}")
    logger.trace(f"Top-m filtering: m={m}")

    # Prepare input
    token_ids = prompt.token_ids.unsqueeze(0).to(device)  # (1, T)
    seq_len = token_ids.shape[1]

    # Apply top-m filtering to compact intervention path
    filtered_compact_path = apply_top_m_filtering(compact_path, m)

    # Expand compact path to full sequence length
    num_layers = filtered_compact_path.shape[1]
    num_experts = filtered_compact_path.shape[2]
    full_intervention_path = th.zeros(
        (seq_len, num_layers, num_experts),
        dtype=th.float32,
        device=device,
    )
    full_intervention_path[start_pos:end_pos] = filtered_compact_path.to(
        device=device, dtype=th.float32
    )

    # Move intervention path to device: (T, L, E)
    intervention_path_device = full_intervention_path

    predictions: list[TopKPrediction] = []

    for alpha in alphas:
        batch = {"input_ids": token_ids}

        with model.trace(batch):
            for i, layer_idx in enumerate(layers_with_routers):
                router_output = model.routers_output[layer_idx]

                # Handle different router output formats
                router_output_is_tuple = isinstance(router_output, tuple)
                router_output_len = len(router_output) if router_output_is_tuple else 0

                if router_output_is_tuple:
                    if router_output_len == 3:
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
                        raise ValueError(
                            f"Unexpected router output tuple length {router_output_len}"
                        )
                else:
                    router_scores = cast("th.Tensor", router_output.save())
                    router_logits = router_scores.reshape(1, seq_len, -1)

                # Apply per-token intervention directly (no mask needed)
                # intervention_path_device is (T, L, E), we need layer i: (T, E)
                layer_intervention = intervention_path_device[:, i, :]  # (T, E)

                modified_logits = router_logits.clone()
                # Apply intervention: (1, T, E) -= alpha * (T, E) -> broadcast works
                modified_logits -= alpha * layer_intervention.unsqueeze(0)

                if router_output_is_tuple and router_output_len == 3:
                    # Recompute topk for all positions
                    all_new_weights, all_new_indices = th.topk(
                        modified_logits, k=router_top_k, dim=-1
                    )  # (1, T, top_k)
                    all_new_weights = F.softmax(all_new_weights, dim=-1)

                    # Update weights and indices where intervention is non-zero
                    intervention_nonzero = (
                        layer_intervention.abs().sum(dim=-1) > 0
                    )  # (T,)
                    mask_topk = (
                        intervention_nonzero.unsqueeze(0)
                        .unsqueeze(-1)
                        .expand_as(router_weights)
                    )  # (1, T, top_k)

                    modified_weights = th.where(
                        mask_topk, all_new_weights, router_weights
                    )
                    modified_indices = th.where(
                        mask_topk, all_new_indices, router_indices
                    )

                    model.routers_output[layer_idx] = (
                        modified_logits.reshape(-1, modified_logits.shape[-1]),
                        modified_weights.reshape(-1, modified_weights.shape[-1]),
                        modified_indices.reshape(-1, modified_indices.shape[-1]),
                    )
                else:
                    model.routers_output[layer_idx] = modified_logits.reshape(
                        -1, modified_logits.shape[-1]
                    )

            final_logits = model.lm_head.output.save()  # (1, T, vocab_size)

        # Get top-k predictions from the last token
        last_token_logits = final_logits[0, -1, :]  # (vocab_size,)
        probs = F.softmax(last_token_logits.float(), dim=-1)

        top_probs, top_indices = th.topk(probs, k=num_top_tokens, dim=-1)
        top_tokens = [model.tokenizer.decode([idx.item()]) for idx in top_indices]

        predictions.append(
            TopKPrediction(
                alpha=alpha,
                tokens=top_tokens,
                probs=top_probs.cpu().tolist(),
            )
        )

    return predictions


def _sanitize_string_for_display(string: str) -> str:
    """
    Sanitize a string for display in matplotlib.

    Handles:
    - Control characters and non-printable characters
    - Unicode characters that may not render in all fonts
    - Special tokenizer markers
    - Whitespace characters (made visible)

    Args:
        string: Raw string to sanitize

    Returns:
        A display-safe version of the token
    """
    # Replace common special tokens and tokenizer markers first
    replacements = {
        "Ġ": " ",  # GPT-2 style space marker
        "▁": " ",  # SentencePiece space marker
        "Ċ": "\n",  # GPT-2 style newline
        "\x00": "",  # Null byte
        "\ufffd": "<?>",  # Unicode replacement character (diamond with ?)
    }

    result = string
    for old, new in replacements.items():
        result = result.replace(old, new)

    # Now process character by character
    sanitized = []
    for char in result:
        # Check for specific whitespace/control characters
        if char == " ":
            sanitized.append("␣")  # Unicode symbol for space (visible)
        elif char == "\n":
            sanitized.append("<nl>")
        elif char == "\r":
            sanitized.append("<cr>")
        elif char == "\t":
            sanitized.append("<tab>")
        elif char.isspace():
            # Other whitespace (non-breaking space, etc.)
            sanitized.append(f"<U+{ord(char):04X}>")
        elif char.isprintable():
            # Check if it's in a safe ASCII-like range or common chars
            if ord(char) < 128 or char.isalnum():
                sanitized.append(char)
            else:
                # Extended Unicode - might not render, show codepoint
                # But first try to keep common punctuation and symbols
                try:
                    # Test if it's a "safe" character by checking category
                    import unicodedata

                    cat = unicodedata.category(char)
                    # Keep letters, numbers, punctuation, symbols, currency
                    if cat.startswith(("L", "N", "P", "S")):
                        sanitized.append(char)
                    else:
                        sanitized.append(f"<U+{ord(char):04X}>")
                except Exception:
                    sanitized.append(f"<U+{ord(char):04X}>")
        else:
            # Non-printable character
            sanitized.append(f"<U+{ord(char):04X}>")

    final = "".join(sanitized)

    # If result is empty or only whitespace markers, show placeholder
    if not final or final.isspace():
        return "<empty>"

    return final


def plot_topk_grid(
    predictions: list[TopKPrediction],
    target_country: str,
    correct_capital: str,
    prompt_text: str,
    output_path: Path,
) -> None:
    """
    Plot a grid showing top-k token predictions at each alpha level.

    The grid has num_top_tokens rows and num_alphas columns.
    Each cell shows the token and is shaded by probability.
    The correct capital is highlighted in green.

    Args:
        predictions: List of TopKPrediction for each alpha
        target_country: Name of target country
        correct_capital: The correct capital city
        prompt_text: The prompt text being completed
        output_path: Path to save the figure
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # DejaVu Sans is bundled with matplotlib and has excellent Unicode support
    plt.rcParams["font.family"] = "DejaVu Sans"

    num_alphas = len(predictions)
    num_tokens = len(predictions[0].tokens) if predictions else 0

    if num_alphas == 0 or num_tokens == 0:
        logger.warning("No predictions to plot")
        return

    # Create figure with appropriate size
    fig_width = max(12, num_alphas * 1.5)
    fig_height = max(4, num_tokens * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Build the data grid
    cell_texts = [[]] * num_tokens
    cell_colors = th.zeros((num_tokens, num_alphas))

    for alpha_idx, pred in enumerate(predictions):
        for token_idx, (token, prob) in enumerate(
            zip(pred.tokens, pred.probs, strict=True)
        ):
            if alpha_idx == 0:
                cell_texts[token_idx] = [""] * num_alphas
            # Sanitize token for display
            cell_texts[token_idx][alpha_idx] = _sanitize_string_for_display(token)
            cell_colors[token_idx, alpha_idx] = prob

    # Create custom colormap (white to blue, with green highlight for correct)
    base_cmap = plt.cm.Blues

    # Plot the heatmap
    im = ax.imshow(cell_colors.numpy(), aspect="auto", cmap=base_cmap, vmin=0, vmax=1)

    # Add text annotations
    for token_idx in range(num_tokens):
        for alpha_idx in range(num_alphas):
            token = cell_texts[token_idx][alpha_idx]
            prob = cell_colors[token_idx, alpha_idx]

            # Determine text color based on background
            text_color = "white" if prob > 0.5 else "black"

            token_sanitized = (
                token.lower()
                .replace("␣", "")
                .replace("<nl>", "")
                .replace("<cr>", "")
                .replace("<tab>", "")
            )

            # Check if this is the correct capital (case-insensitive, handle tokenization variations)
            is_correct = (
                correct_capital.lower().startswith(token_sanitized)
                and len(token_sanitized) > 0
            )
            if is_correct:
                # Add green background for correct capital
                ax.add_patch(
                    plt.Rectangle(
                        (alpha_idx - 0.5, token_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="green",
                        linewidth=3,
                    )
                )

            # Add text
            ax.text(
                alpha_idx,
                token_idx,
                f"{token}\n{prob:.2%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold" if is_correct else "normal",
            )

    # Set axis labels
    ax.set_xticks(range(num_alphas))
    ax.set_xticklabels(
        [f"a={pred.alpha:.2f}" for pred in predictions], rotation=45, ha="right"
    )
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels([f"Rank {i + 1}" for i in range(num_tokens)])

    ax.set_xlabel("Intervention Strength (alpha)", fontsize=12)
    ax.set_ylabel("Token Rank", fontsize=12)
    ax.set_title(
        f"Top-{num_tokens} Token Predictions by Alpha\n"
        f"{target_country} (correct: {correct_capital})",
        fontsize=14,
    )

    # Add prompt text below the plot
    # Sanitize and truncate if too long
    max_prompt_len = 300
    sanitized_prompt = _sanitize_string_for_display(prompt_text)
    display_prompt = (
        sanitized_prompt
        if len(sanitized_prompt) <= max_prompt_len
        else sanitized_prompt[:max_prompt_len] + "..."
    )
    fig.text(
        0.5,
        0.02,
        f"Prompt: {display_prompt}",
        ha="center",
        fontsize=9,
        style="italic",
        wrap=True,
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Probability")

    # Adjust layout to make room for prompt text at bottom
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved top-k grid to {output_path}")


@arguably.command()
def capital_country(
    *,
    model_name: str = "olmoe-i",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    alpha_min: float = 0.0,
    alpha_max: float = 20.0,
    alpha_steps: int = 11,
    postprocessor: str = "masks",
    router_path_batch_size: int = 128,
    intervention_batch_size: int = 8,
    sample_only: bool = False,
    topk_only: bool = False,
    topk_num_tokens: int = 10,
    m_min: int = 0,
    m_max: int = 16 * 128,
    m_steps: int = 1,
    use_same_template_only: bool = True,
    seed: int = 0,
    hf_token: str = "",
    output_dir: str = "out/capital_country",
    log_level: str = "INFO",
) -> None:
    """
    Isolate country-capital knowledge in MoE models.

    Args:
        model_name: Name of the model to use (olmoe-i, q3, gpt, etc.)
        model_step_ckpt: Checkpoint step to load (None for latest)
        model_dtype: Data type for model weights
        alpha_min: Minimum alpha value for intervention sweep
        alpha_max: Maximum alpha value for intervention sweep
        alpha_steps: Number of alpha values to test
        postprocessor: Router logits postprocessor (masks, identity, softmax, etc.)
        router_path_batch_size: Batch size for router path extraction
        intervention_batch_size: Batch size for intervention experiments
        sample_only: If True (default), use only the sample prompt template. If False, use all templates.
        topk_only: If True, skip steps 1-4 and only generate top-k prediction visualizations. Default False.
        topk_num_tokens: Number of top tokens to show in top-k visualization
        m_min: Minimum value of m for top-m experiments. At m=0, use original intervention path.
        m_max: Maximum value of m for top-m experiments. At m>0, keep only top m experts overall.
        m_steps: Number of m values to test (discretized to whole numbers).
        use_same_template_only: If True, compare only against paths from the same template.
            If False, compare against all other countries' paths (averaged).
        seed: Random seed for reproducibility
        hf_token: Hugging Face API token
        output_dir: Directory to save results
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running capital-country experiment with log level: {log_level}")

    # Set random seeds
    th.manual_seed(seed)

    # Parse postprocessor
    postprocessor_enum = RouterLogitsPostprocessor(postprocessor)
    logger.info(f"Postprocessor: {postprocessor_enum}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)

    logger.info(f"Loading model: {model_config.hf_name}")
    logger.info(f"Checkpoint: {model_ckpt}")

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
    model_config_hf = model.config
    num_experts = model_config_hf.num_experts
    top_k = model_config_hf.num_experts_per_tok

    logger.info(f"Number of experts: {num_experts}")
    logger.info(f"Top-k: {top_k}")

    # Determine which templates to use
    if sample_only:
        templates = SAMPLE_PROMPT_TEMPLATE
        logger.info("Using sample prompt template only")
    else:
        templates = frozenset(PROMPT_TEMPLATES)
        logger.info(f"Using all {len(templates)} prompt templates")

    alphas = frozenset(th.linspace(alpha_min, alpha_max, alpha_steps).tolist())

    assert m_min >= 0, "m_min must be non-negative"
    assert m_max > m_min, "m_max must be greater than m_min"
    assert m_steps < (m_max - m_min), "m_steps must be less than m_max - m_min"

    # Generate m values (discretized to whole numbers)
    m_values_linear = th.linspace(m_min, m_max, m_steps + 1)[:-1].round().int().tolist()
    m_values_linear = [int(m) for m in m_values_linear]  # Ensure they're Python ints

    if m_min > 0:
        first_nonzero = m_values_linear[0]
    elif len(m_values_linear) > 1:
        first_nonzero = m_values_linear[1]
    else:
        first_nonzero = m_max

    # Generate powers of 2 from 1 up to (but not including) first_nonzero
    exponent_to_reach = math.ceil(math.log2(first_nonzero)) - 1

    m_values_exp = [2**i for i in range(exponent_to_reach)]

    # Combine linear and exponential values, remove duplicates, and sort
    m_values = sorted(set(m_values_linear + m_values_exp))

    # Step 1: Extract router paths for all prompts
    # (prompts are generated and cached internally by generate_prompts)
    logger.info("=" * 80)
    logger.info("STEP 1: Extracting per-token router paths")
    logger.info("=" * 80)

    all_paths = extract_router_paths(
        model,
        top_k=top_k,
        batch_size=router_path_batch_size,
        postprocessor=postprocessor_enum,
        templates=templates,
    )
    logger.info(f"Extracted {len(all_paths)} per-token router paths")

    if not topk_only:
        # Step 2: Run intervention experiments
        logger.info("=" * 80)
        logger.info("STEP 2: Running intervention experiments")
        logger.info("=" * 80)
        logger.info(f"Testing alphas: {alphas}")
        logger.info(f"Testing m values: {m_values}")

        # Iterate over m values
        for m in tqdm(
            m_values,
            desc="Testing m values",
            total=len(m_values),
            position=4,
        ):
            logger.info(f"Running intervention experiments with m={m}")
            results = run_intervention_experiment(
                model,
                all_paths,
                alphas,
                top_k,
                batch_size=intervention_batch_size,
                templates=templates,
                m=m,
                use_same_template_only=use_same_template_only,
            )

            # Save results
            for experiment_type, all_results in results.items():
                for country in COUNTRY_TO_CAPITAL:
                    all_country_results = {
                        result
                        for result in all_results
                        if result.target_country == country
                    }
                    results_dir = output_path / country.lower()
                    results_dir.mkdir(parents=True, exist_ok=True)
                    results_file = results_dir / f"{experiment_type.value}_m-{m}.yaml"

                    assert len(all_country_results) == 1, (
                        f"Expected 1 country result for {country}, got {len(all_country_results)}"
                    )
                    country_results = next(iter(all_country_results))

                    assert isinstance(country_results, ExperimentResults), (
                        f"Expected ExperimentResults for {country}, got {type(country_results)}"
                    )

                    country_alphas = sorted(
                        specificity_score.alpha
                        for specificity_score in country_results.specificity_scores
                    )

                    target_forgetfulness: list[float] = []
                    other_forgetfulness: list[float] = []
                    other_average_forgetfulness: list[float] = []
                    specificity_scores: list[float] = []
                    for alpha in country_alphas:
                        target_results_for_alpha = {
                            result
                            for result in country_results.target_results
                            if result.forgetfulness.alpha == alpha
                        }
                        other_results_for_alpha = {
                            result
                            for result in country_results.other_results
                            if result.forgetfulness.alpha == alpha
                        }
                        other_results_averaged_for_alpha = {
                            result
                            for result in country_results.other_results_averaged
                            if result.forgetfulness.alpha == alpha
                        }
                        specificity_scores_for_alpha = {
                            specificity_score
                            for specificity_score in country_results.specificity_scores
                            if specificity_score.alpha == alpha
                        }

                        assert len(target_results_for_alpha) == 1, (
                            f"Expected 1 target result (averaged) for alpha {alpha}, got {len(target_results_for_alpha)}"
                        )
                        assert len(other_results_averaged_for_alpha) == 1, (
                            f"Expected 1 other result averaged for alpha {alpha}, got {len(other_results_averaged_for_alpha)}"
                        )
                        assert len(specificity_scores_for_alpha) == 1, (
                            f"Expected 1 specificity score for alpha {alpha}, got {len(specificity_scores_for_alpha)}"
                        )

                        target_forgetfulness.append(
                            next(iter(target_results_for_alpha)).forgetfulness.value
                        )
                        other_average_forgetfulness.append(
                            next(
                                iter(other_results_averaged_for_alpha)
                            ).forgetfulness.value
                        )
                        specificity_scores.append(
                            next(iter(specificity_scores_for_alpha)).value
                        )

                        other_forgetfulness.append(
                            next(iter(other_results_for_alpha)).forgetfulness.value
                        )

                    results_dict = {
                        "target_country": country,
                        "target_capital": COUNTRY_TO_CAPITAL[country],
                        "alphas": country_alphas,
                        "target_forgetfulness": target_forgetfulness,
                        "other_average_forgetfulness": other_average_forgetfulness,
                        "specificity_scores": specificity_scores,
                    }
                    with open(results_file, "w") as f:
                        yaml.dump(results_dict, f)

                    logger.debug(f"Saved results to {results_file}")

            # Step 4: Create visualization for this m value
            logger.info(f"Creating visualization for m={m}")
            plot_results(results, Path(FIGURE_DIR) / "capital_country" / f"m-{m}")
    else:
        logger.info("Skipping steps 1-4 (--topk-only mode)")

    # Step 3: Generate top-k prediction visualizations
    logger.info("=" * 80)
    logger.info("STEP 3: Generating top-k prediction visualizations")
    logger.info("=" * 80)

    alphas_list = sorted(alphas)

    # Group paths by prompt for efficient lookup
    paths_by_prompt: dict[CountryPrompt, dict[ExperimentType, PerTokenRouterPath]] = (
        defaultdict(dict)
    )
    for path_obj in all_paths:
        paths_by_prompt[path_obj.prompt][path_obj.experiment_type] = path_obj

    # Pre-compute country interventions if using per-country mode
    country_interventions_cache: dict[str, dict[ExperimentType, th.Tensor]] = {}
    if not use_same_template_only:
        logger.info("Pre-computing country-level interventions for top-k viz...")
        country_interventions_cache = compute_country_interventions(all_paths)

    for target_country in tqdm(
        COUNTRY_TO_CAPITAL.keys(),
        desc="Generating top-k grids",
        total=len(COUNTRY_TO_CAPITAL),
    ):
        # Get paths for this country's prompts
        country_path_objs = [
            path_obj
            for path_obj in all_paths
            if path_obj.prompt.country == target_country
        ]
        if not country_path_objs:
            logger.warning(f"No paths found for {target_country}, skipping top-k viz")
            continue

        # Create country directory
        country_slug = target_country.lower().replace(" ", "_")
        country_topk_dir = Path(FIGURE_DIR) / "capital_country" / "topk" / country_slug
        country_topk_dir.mkdir(parents=True, exist_ok=True)

        # Group by prompt
        prompts_for_country = list({path_obj.prompt for path_obj in country_path_objs})

        # Iterate over m values
        for m in m_values:
            for prompt_idx, prompt in enumerate(
                tqdm(
                    prompts_for_country,
                    desc=f"Processing {target_country} prompts (m={m})",
                    total=len(prompts_for_country),
                    leave=False,
                )
            ):
                # Get paths for this prompt
                prompt_paths = paths_by_prompt[prompt]

                for experiment_type, path_obj in prompt_paths.items():
                    # Compute intervention based on mode
                    if use_same_template_only:
                        # Per-prompt mode: compute for this specific prompt
                        intervention_path = compute_prompt_specific_intervention(
                            path_obj, all_paths
                        )
                    else:
                        # Per-country mode: get from pre-computed country interventions
                        country = path_obj.prompt.country
                        if country not in country_interventions_cache:
                            continue
                        if experiment_type not in country_interventions_cache[country]:
                            continue

                        # Get country intervention (L, E) and expand to token range
                        country_interv = country_interventions_cache[country][
                            experiment_type
                        ]
                        range_len = path_obj.end_pos - path_obj.start_pos
                        intervention_path = country_interv.unsqueeze(0).expand(
                            range_len, -1, -1
                        )

                    # Bundle position info with the intervention path
                    intervention_path_info = (
                        path_obj.start_pos,
                        path_obj.end_pos,
                        intervention_path,
                    )

                    # Extract top-k predictions at each alpha
                    predictions = extract_topk_predictions_across_alphas(
                        model=model,
                        prompt=prompt,
                        intervention_path_info=intervention_path_info,
                        alphas=alphas_list,
                        num_top_tokens=topk_num_tokens,
                        router_top_k=top_k,
                        m=m,
                    )

                    # Plot the grid
                    topk_output_path = (
                        country_topk_dir
                        / f"prompt-{prompt_idx:03d}_{experiment_type.value.replace('_', '-')}_m-{m}.png"
                    )
                    plot_topk_grid(
                        predictions=predictions,
                        target_country=target_country,
                        correct_capital=COUNTRY_TO_CAPITAL[target_country],
                        prompt_text=prompt.formatted_text,
                        output_path=topk_output_path,
                    )

    # Print summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    num_prompts = len(generate_prompts(tokenizer, templates))
    logger.info(f"Total prompts: {num_prompts}")
    logger.info(f"Total per-token paths: {len(all_paths)}")
    logger.info(f"Countries tested: {len(COUNTRY_TO_CAPITAL)}")
    logger.info(f"Templates per country: {len(templates)}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Figures saved to: {Path(FIGURE_DIR) / 'capital_country'}")
    topk_base_dir = Path(FIGURE_DIR) / "capital_country" / "topk"
    logger.info(f"Top-k grids saved to: {topk_base_dir}/<country>/")
    logger.info(
        f"Top-m experiments: m values from {m_min} to {m_max} ({m_steps} steps, values: {m_values})"
    )
    logger.info(
        f"Total top-k grids generated: {num_prompts * len(m_values) * 3}"
    )  # 3 experiment types


if __name__ == "__main__":
    arguably.run()
