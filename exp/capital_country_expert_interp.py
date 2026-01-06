"""
Expert interpretability experiment for capital-country knowledge in MoE models.

This experiment:
1. Finds the most important experts for a given prompt by measuring intervention effects
2. Runs incremental ablations (top-1, top-2, ... experts) on the last token
3. Captures binary routing patterns (layers x experts) for each intervention
4. Performs similarity search across stored activations to find similar routing patterns
5. Returns tokens/passages corresponding to similar patterns pre/post intervention

Usage:
    uv run python -m exp.capital_country_expert_interp \\
        --model-name "olmoe-i" \\
        --target-country "France" \\
        --max-experts-to-ablate 16 \\
        --top-k-similar 10
"""

import asyncio
from dataclasses import dataclass
from itertools import batched, product
from pathlib import Path
import sys
from typing import Literal

import arguably
from loguru import logger
from nnterp import StandardizedTransformer
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from core.moe import convert_router_logits_to_paths
from exp.activations import Activations, load_activations_and_init_dist
from exp.get_activations import ActivationKeys

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

PROMPT_TEMPLATE = [
    {"role": "user", "content": "What is the capital of {country}?"},
    {"role": "assistant", "content": "The capital of {country} is"},
]

NEGATIVE_INF = float("-inf")


@dataclass(frozen=True)
class ExpertLocation:
    """Location of an expert in the model."""

    token_idx: int
    layer_idx: int
    expert_idx: int

    def __hash__(self) -> int:
        return hash((self.token_idx, self.layer_idx, self.expert_idx))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpertLocation):
            return False
        return (
            self.token_idx == other.token_idx
            and self.layer_idx == other.layer_idx
            and self.expert_idx == other.expert_idx
        )


@dataclass(frozen=True)
class ExpertImportance:
    """Expert with its importance score."""

    location: ExpertLocation
    importance: float  # Negative = more important (reduces correct answer prob more)


@dataclass
class InterventionResult:
    """Result of a single intervention ablating N experts."""

    num_experts_ablated: int
    ablated_experts: list[ExpertLocation]
    pre_intervention_prob: float
    post_intervention_prob: float
    prob_change: float
    routing_pattern: th.Tensor  # (num_layers, num_experts) binary mask
    generated_token: str
    generated_token_prob: float


@dataclass
class SimilarActivation:
    """A similar activation pattern found in the dataset."""

    similarity_score: float
    routing_mask: th.Tensor  # (num_layers, num_experts) binary mask
    text_passage: str
    activating_sample_slice: tuple[int, int]

    def highlighted_text_passage(self) -> str:
        """Highlight the text passage with the activating sample slice."""
        start, end = self.activating_sample_slice
        highlighted_token = self.text_passage[start:end]
        prefix = self.text_passage[:start]
        suffix = self.text_passage[end:]

        return f"{prefix}<<{highlighted_token}>>{suffix}"


@dataclass
class InterpretabilityResult:
    """Complete result of interpretability analysis for an intervention."""

    intervention: InterventionResult
    similar_activations_pre: list[SimilarActivation]
    similar_activations_post: list[SimilarActivation]


@dataclass
class ExperimentOutput:
    """Complete output of the experiment."""

    target_country: str
    target_capital: str
    prompt: str
    baseline_prob: float
    baseline_routing_pattern: th.Tensor  # (num_layers, num_experts) binary mask
    expert_importances: list[ExpertImportance]
    intervention_results: list[InterpretabilityResult]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_correct_answer_tokens(tokenizer, capital: str) -> dict[str, int]:
    """Get token IDs for correct answer tokens."""
    tokens_cleaned = tuple(
        (token, token.replace("Ġ", " "), token_id)
        for token, token_id in tokenizer.get_vocab().items()
    )
    correct_answer_tokens = {
        token: token_id
        for token, token_cleaned, token_id in tokens_cleaned
        if f" {capital}".lower().startswith(token_cleaned.lower())
        and token_cleaned != " "
    }
    return correct_answer_tokens


@th.no_grad()
def compute_expert_importance_and_routing_masks(
    model: StandardizedTransformer,
    batch: dict[str, th.Tensor],
    correct_answer_tokens: dict[str, int],
    processing_batch_size: int = 32,
    top_k: int = 8,
    last_token_only: bool = True,
) -> tuple[list[ExpertImportance], th.Tensor, th.Tensor, th.Tensor]:
    """
    Compute importance scores for all experts by measuring intervention effects.

    Args:
        model: The MoE model
        batch: Tokenized input batch
        correct_answer_tokens: Dict mapping token strings to IDs for correct answer
        processing_batch_size: Batch size for processing interventions
        last_token_only: If True, only intervene on the last token

    Returns:
        Tuple of (sorted expert importances, baseline probs, baseline routing pattern, modified routing patterns)
    """
    layers_with_routers = model.layers_with_routers
    seq_len = batch["input_ids"].shape[-1]
    num_experts = len(model.layers[layers_with_routers[0]].mlp.experts)
    num_layers = len(layers_with_routers)

    # (T, L, E) binary mask of experts pre-intervention
    baseline_mask: th.Tensor = th.zeros(
        seq_len, num_layers, num_experts, dtype=th.bool, device=model.device
    )
    # Get baseline logits and routing pattern
    with model.trace(batch):
        # Capture routing pattern
        for layer_idx in layers_with_routers:
            router_output = model.routers_output[layer_idx].save()
            _topk_weights, topk_indices = th.topk(router_output, k=top_k, dim=-1)

            baseline_mask[:, layer_idx, :].scatter_(-1, topk_indices, True)

        baseline_logits = model.lm_head.output.save()  # (1, T, vocab_size)

    # (vocab_size,)
    baseline_probs = F.softmax(baseline_logits[0, -1, :].float(), dim=-1)

    th.cuda.empty_cache()

    # Compute intervention effects
    intervention_effects: dict[ExpertLocation, float] = {}

    # Determine which tokens to intervene on
    token_indices = [seq_len - 1] if last_token_only else range(seq_len)

    layer_token_expert_iterator = product(
        token_indices, layers_with_routers, range(num_experts)
    )
    batch_layer_token_expert_iterator = enumerate(layer_token_expert_iterator)
    batched_iterator = tuple(
        batched(batch_layer_token_expert_iterator, processing_batch_size)
    )

    # (T*L*E, T, L, E) binary mask of experts post-intervention
    intervention_router_masks: th.Tensor = th.zeros(
        seq_len * num_layers * num_experts,
        seq_len,
        num_layers,
        num_experts,
        dtype=th.bool,
    )

    for token_layer_expert_batch in tqdm(
        batched_iterator,
        desc="Computing expert importance",
        total=len(batched_iterator),
    ):
        current_batch = {
            "input_ids": batch["input_ids"].expand(processing_batch_size, -1)
        }

        # Create intervention mask: (B, T, L, E)
        intervention_mask = th.zeros(
            processing_batch_size, seq_len, num_layers, num_experts, dtype=th.bool
        )
        for relative_batch_idx, (
            _batch_idx,
            (token_idx, layer_idx, expert_idx),
        ) in enumerate(token_layer_expert_batch):
            intervention_mask[relative_batch_idx, token_idx, layer_idx, expert_idx] = (
                True
            )

        # (B, T, L, E) -> (B*T, L, E)
        intervention_mask_flat = intervention_mask.view(-1, num_layers, num_experts)
        # (B,)
        intervention_batch_indices = th.tensor(
            [batch_idx for batch_idx, _ in token_layer_expert_batch],
        )

        with model.trace(current_batch):
            for layer_idx in layers_with_routers:
                layer_intervention_mask = intervention_mask_flat[:, layer_idx, :]
                model.routers_output[layer_idx][layer_intervention_mask] = NEGATIVE_INF

                router_output = model.routers_output[layer_idx].save()
                _topk_weights, topk_indices = th.topk(router_output, k=top_k, dim=-1)

                flat_relevant_intervention_router_masks = intervention_router_masks[
                    intervention_batch_indices, :, layer_idx, :
                ].view(-1, num_experts)
                flat_relevant_intervention_router_masks[topk_indices] = True

            final_logits = model.lm_head.output.save()  # (B, T, vocab_size)

        final_probs = F.softmax(
            final_logits[:, -1, :].float(), dim=-1
        )  # (B, vocab_size)

        for (_batch_idx, (token_idx, layer_idx, expert_idx)), probs in zip(
            token_layer_expert_batch, final_probs, strict=False
        ):
            # Compute mean change across all correct answer tokens
            changes = [
                probs[token_idx] - baseline_probs[token_idx]
                for token_idx in correct_answer_tokens.values()
            ]

            mean_change = sum(changes) / len(changes)

            location = ExpertLocation(
                token_idx=token_idx,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
            )
            intervention_effects[location] = mean_change

        th.cuda.empty_cache()

    intervention_routing = intervention_router_masks.view(
        seq_len, num_layers, num_experts, seq_len, num_layers, num_experts
    )

    # Sort by importance (most negative = most important)
    sorted_experts = sorted(intervention_effects.items(), key=lambda x: x[1])

    expert_importances = [
        ExpertImportance(location=location, importance=effect)
        for location, effect in sorted_experts
    ]

    return expert_importances, baseline_probs, baseline_mask, intervention_routing


@th.no_grad()
def run_incremental_ablations(
    model: StandardizedTransformer,
    batch: dict[str, th.Tensor],
    expert_importances: list[ExpertImportance],
    correct_answer_tokens: dict[str, int],
    baseline_probs: th.Tensor,
    max_experts_to_ablate: int = 32,
    top_k: int = 8,
) -> list[InterventionResult]:
    """
    Run incremental ablations, ablating top-1, top-2, ..., top-N experts.

    Only intervenes on the last token position.

    Args:
        model: The MoE model
        batch: Tokenized input batch
        expert_importances: Sorted list of expert importances
        correct_answer_tokens: Dict mapping token strings to IDs
        baseline_probs: Baseline probability distribution
        max_experts_to_ablate: Maximum number of experts to ablate
        top_k: Number of top experts selected by router

    Returns:
        List of intervention results
    """
    layers_with_routers = list(model.layers_with_routers)
    num_layers = len(layers_with_routers)
    seq_len = batch["input_ids"].shape[-1]
    num_experts = len(model.layers[layers_with_routers[0]].mlp.experts)

    # Limit to available experts
    assert max_experts_to_ablate < len(expert_importances), (
        f"Ablating {max_experts_to_ablate} while we have {len(expert_importances)} expert importances"
    )

    results: list[InterventionResult] = []

    # Create batch for all ablation levels at once
    current_batch = {"input_ids": batch["input_ids"].expand(max_experts_to_ablate, -1)}

    # Build intervention masks for each ablation level
    # intervention_mask[i] ablates the top-(i+1) experts
    intervention_mask = th.zeros(
        max_experts_to_ablate,
        seq_len,
        num_layers,
        num_experts,
        dtype=th.bool,
        device=model.device,
    )  # (B, T, L, E)
    new_routing_pattern = intervention_mask.clone()

    for ablation_level in range(max_experts_to_ablate):
        expert = expert_importances[ablation_level]
        token_idx = expert.location.token_idx
        layer_idx = expert.location.layer_idx
        expert_idx = expert.location.expert_idx

        # Ablation level i ablates experts 0..i (inclusive)
        # So mask[i:] has this expert ablated
        intervention_mask[ablation_level:, token_idx, layer_idx, expert_idx] = True

    with model.trace(current_batch):
        for layer_idx in layers_with_routers:
            # Apply ablation mask
            local_mask = intervention_mask[:, :, layer_idx, :]  # (B, T, E)
            local_mask_flat = local_mask.view(-1, num_experts)

            model.routers_output[layer_idx][local_mask_flat] = NEGATIVE_INF

            router_output = (
                model.routers_output[layer_idx]
                .save()
                .view(max_experts_to_ablate, seq_len, num_experts)
            )

            # Compute new top-k indices after ablation
            _, new_indices = th.topk(
                router_output,
                k=top_k,
                dim=-1,
            )
            new_routing_pattern[:, :, layer_idx, :].scatter_(-1, new_indices, True)

        final_logits = model.lm_head.output.save()  # (B, T, vocab_size)

    # Process results
    final_probs = F.softmax(final_logits[:, -1, :].float(), dim=-1)  # (B, vocab_size)

    for ablation_level in range(max_experts_to_ablate):
        probs = final_probs[ablation_level]

        # Compute mean change across correct answer tokens
        pre_prob = 0.0
        post_prob = 0.0
        for token_id in correct_answer_tokens.values():
            pre_prob += baseline_probs[token_id].item()
            post_prob += probs[token_id].item()
        pre_prob /= len(correct_answer_tokens)
        post_prob /= len(correct_answer_tokens)

        # Get generated token
        generated_token_id = probs.argmax().item()
        generated_token = model.tokenizer.decode([generated_token_id])
        generated_token_prob = probs[generated_token_id].item()

        # Collect ablated experts for this level
        ablated_experts = [
            expert_importances[i].location for i in range(ablation_level + 1)
        ]

        results.append(
            InterventionResult(
                num_experts_ablated=ablation_level + 1,
                ablated_experts=ablated_experts,
                pre_intervention_prob=pre_prob,
                post_intervention_prob=post_prob,
                prob_change=post_prob - pre_prob,
                routing_pattern=new_routing_pattern[ablation_level],
                generated_token=generated_token,
                generated_token_prob=generated_token_prob,
            )
        )

    return results


def compute_routing_similarity_batched(
    target_patterns: th.Tensor,
    candidate_patterns: th.Tensor,
    method: Literal["jaccard", "cosine", "hamming"] = "jaccard",
) -> th.Tensor:
    """
    Compute similarity between batches of routing patterns.

    Args:
        target_patterns: Binary masks (N, L*E) flattened target patterns
        candidate_patterns: Binary masks (M, L*E) flattened candidate patterns
        method: Similarity method to use

    Returns:
        Similarity scores (N, M) where [i, j] is similarity between target i and candidate j
    """
    # target_patterns: (N, D) where D = L*E
    # candidate_patterns: (M, D)
    p1 = target_patterns.float()  # (N, D)
    p2 = candidate_patterns.float()  # (M, D)

    if method == "jaccard":
        # Compute pairwise intersection and union
        # intersection[i,j] = sum(p1[i] * p2[j])
        intersection = p1 @ p2.T  # (N, M)
        # For union: |A| + |B| - |A ∩ B|
        p1_sum = p1.sum(dim=1, keepdim=True)  # (N, 1)
        p2_sum = p2.sum(dim=1, keepdim=True)  # (M, 1)
        union = p1_sum + p2_sum.T - intersection  # (N, M)
        similarity = th.where(union > 0, intersection / union, th.tensor(0))
    elif method == "cosine":
        # Normalize and compute dot product
        p1_norm = p1.norm(dim=1, keepdim=True)  # (N, 1)
        p2_norm = p2.norm(dim=1, keepdim=True)  # (M, 1)
        # Avoid division by zero
        p1_normalized = p1 / (p1_norm + 1e-8)
        p2_normalized = p2 / (p2_norm + 1e-8)
        similarity = p1_normalized @ p2_normalized.T  # (N, M)
        # Zero out where norms were zero
        similarity = th.where((p1_norm > 0) & (p2_norm.T > 0), similarity, th.tensor(0))
    elif method == "hamming":
        # 1 - (hamming distance / total)
        # hamming distance = number of positions where they differ
        D = p1.shape[1]
        # XOR gives 1 where they differ
        diff = (p1.unsqueeze(1) != p2.unsqueeze(0)).float()  # (N, M, D)
        hamming_dist = diff.sum(dim=2)  # (N, M)
        similarity = 1.0 - hamming_dist / D
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    return similarity


def search_similar_activations_batched(
    target_patterns: list[th.Tensor],
    activations: Activations,
    tokenizer: PreTrainedTokenizer,
    top_k: int = 10,
    max_samples: int = 10_000_000,
    target_batch_size: int = 17,
    activation_batch_size: int = 4096,
    router_top_k: int = 8,
    activation_context_length: int = 32,
    similarity_method: Literal["jaccard", "cosine", "hamming"] = "jaccard",
) -> list[list[SimilarActivation]]:
    """
    Search for similar routing patterns in the activation dataset for multiple targets.

    Args:
        target_patterns: List of target binary routing masks, each (L, E)
        activations: Activations object to search through
        tokenizer: Tokenizer to decode text passages
        top_k: Number of similar patterns to return per target
        max_samples: Maximum number of samples to search
        batch_size: Batch size for loading activations
        router_top_k: Number of top experts selected by router
        activation_context_length: Context length for activations
        similarity_method: Method for computing similarity

    Returns:
        List of lists, where result[i] contains top-k similar activations for target_patterns[i]
    """
    num_targets = len(target_patterns)
    if num_targets == 0:
        return []

    context_expansion_amount = activation_context_length // 2

    # Batch targets
    target_batches = batched(target_patterns, target_batch_size)
    target_patterns_flat = [
        th.stack(
            [pattern.flatten().cpu() for pattern in target_batch],
            dim=0,
        )
        for target_batch in target_batches
    ]
    target_pattern_indices = tuple(batched(range(num_targets), target_batch_size))

    # Initialize per-target heaps to track top-k
    # Each entry is (top-k similarity scores, top-k similar activations)
    similar_activations: list[tuple[th.Tensor, list[SimilarActivation | None]]] = [
        (th.full((top_k,), -float("inf")), [None] * top_k) for _ in range(num_targets)
    ]

    samples_processed = 0

    logger.info(
        f"Searching for similar activations for {num_targets} targets (max {max_samples} samples)..."
    )

    for batch in tqdm(
        activations(batch_size=activation_batch_size, max_samples=max_samples),
        desc="Searching activations",
        total=max_samples // activation_batch_size,
    ):
        router_logits = batch.get(ActivationKeys.ROUTER_LOGITS)
        assert router_logits is not None, "No router logits found in batch"

        batch_tokens = batch.get("tokens")
        assert batch_tokens is not None, "No tokens found in batch"

        num_documents = len(batch_tokens)
        document_lengths = th.tensor(
            [len(tokens) for tokens in batch_tokens], dtype=th.long
        )
        total_document_length = document_lengths.sum()
        document_start_indices = th.cat(
            [th.tensor([0]), document_lengths[:-1].cumsum(dim=0)]
        )
        document_start_mask: th.Tensor = th.zeros(total_document_length, dtype=th.long)
        document_start_mask.scatter_(0, document_start_indices, 1)
        document_idx = document_start_mask.cumsum(dim=0) - 1

        document_idx_mask = th.zeros(
            num_documents, total_document_length, dtype=th.bool
        )
        document_idx_mask.scatter_(0, document_idx.unsqueeze(0), 1)

        relative_token_idx_per_document = (
            th.arange(document_lengths.sum()).unsqueeze(0).repeat(num_documents, 1)
        )
        relative_token_idx_per_document -= document_start_indices.unsqueeze(1)
        relative_token_idx_per_document[~document_idx_mask] = 0

        relative_token_idx = relative_token_idx_per_document.sum(dim=0)

        batch_size, num_layers, num_experts = router_logits.shape
        router_masks = convert_router_logits_to_paths(
            router_logits, router_top_k
        )  # (B, L, E)
        router_masks_flat = router_masks.view(batch_size, -1)  # (B, L*E)

        for target_indices, target_masks_flat in zip(
            target_pattern_indices, target_patterns_flat, strict=True
        ):
            # Compute similarities for all targets against all candidates: (N, M)
            similarities = compute_routing_similarity_batched(
                target_masks_flat, router_masks_flat, method=similarity_method
            )

            # Update top-k for each target
            for target_idx in target_indices:
                current_similarities, current_activations = similar_activations[
                    target_idx
                ]
                old_similarities = current_similarities.clone()
                old_activations = list(current_activations)
                sims = th.cat(
                    [old_similarities, similarities[target_idx]],
                    dim=0,
                )  # (top_k + M,)

                new_similarities, new_similarity_indices = th.topk(sims, k=top_k, dim=0)
                for top_k_idx, (new_similarity_idx, new_similarity_score) in enumerate(
                    zip(new_similarity_indices, new_similarities, strict=True),
                ):
                    batch_idx = new_similarity_idx - top_k
                    # if one of the new similarities made it into the top-k
                    if batch_idx >= 0:
                        sample_document_idx = document_idx[batch_idx].item()
                        sample_relative_token_idx = relative_token_idx[batch_idx].item()
                        context_start_idx = max(
                            0, sample_relative_token_idx - context_expansion_amount
                        )
                        context_end_idx = min(
                            document_lengths[sample_document_idx].item(),
                            sample_relative_token_idx + context_expansion_amount,
                        )

                        sample_tokens = batch_tokens[sample_document_idx][
                            context_start_idx:context_end_idx
                        ]
                        activating_token_idx = min(
                            context_expansion_amount,
                            sample_relative_token_idx,
                        )
                        sample_tokens_prefix = sample_tokens[:activating_token_idx]
                        sample_tokens_suffix = sample_tokens[activating_token_idx + 1 :]

                        text_passage = tokenizer.convert_tokens_to_string(sample_tokens)
                        text_passage_prefix = tokenizer.convert_tokens_to_string(
                            sample_tokens_prefix
                        )
                        text_passage_suffix = tokenizer.convert_tokens_to_string(
                            sample_tokens_suffix
                        )

                        current_similarities[top_k_idx] = new_similarity_score
                        # Reshape flat mask back to (L, E) for SimilarActivation
                        routing_mask_reshaped = router_masks[batch_idx].cpu()  # (L, E)
                        current_activations[top_k_idx] = SimilarActivation(
                            similarity_score=new_similarity_score.item(),
                            routing_mask=routing_mask_reshaped,
                            text_passage=text_passage,
                            activating_sample_slice=(
                                len(text_passage_prefix),
                                len(text_passage) - len(text_passage_suffix),
                            ),
                        )
                    else:
                        current_similarities[top_k_idx] = old_similarities[
                            new_similarity_idx
                        ]
                        current_activations[top_k_idx] = old_activations[
                            new_similarity_idx
                        ]
                similar_activations[target_idx] = (
                    new_similarities,
                    current_activations,
                )

        samples_processed += total_document_length.item()

        if max_samples > 0 and samples_processed >= max_samples:
            break

    logger.info(f"Searched {samples_processed} samples for {num_targets} targets")

    # Extract just the SimilarActivation objects
    output = [activations for _similarity_scores, activations in similar_activations]

    assert all(activations is not None for activations in output), (
        "Not all targets have activations"
    )
    return output


@arguably.command()
def capital_country_expert_interp(
    *,
    model_name: str = "olmoe-i",
    dataset_name: str = "lmsys",
    model_step_ckpt: int | None = None,
    model_dtype: str = "bf16",
    target_country: str = "France",
    max_experts_to_ablate: int = 16,
    top_k_similar: int = 10,
    max_samples_to_search: int = 100000,
    target_batch_size: int = 17,
    activation_batch_size: int = 4096,
    processing_batch_size: int = 32,
    similarity_method: str = "jaccard",
    context_length: int = 2048,
    tokens_per_file: int = 50_000,
    output_dir: str = "out/capital_country_expert_interp",
    log_level: str = "INFO",
    hf_token: str = "",
) -> ExperimentOutput:
    """
    Run expert interpretability experiment for capital-country knowledge.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        model_step_ckpt: Model checkpoint step (None for latest)
        model_dtype: Model dtype
        target_country: Country to analyze
        max_experts_to_ablate: Maximum number of experts to ablate
        top_k_similar: Number of similar patterns to find
        max_samples_to_search: Maximum samples to search for similarity
        activation_batch_size: Batch size for loading activations
        processing_batch_size: Batch size for computing expert importance
        similarity_method: Method for computing similarity (jaccard, cosine, hamming)
        context_length: Context length for activations
        tokens_per_file: Tokens per file for activations
        output_dir: Output directory
        log_level: Logging level
        hf_token: HuggingFace token

    Returns:
        ExperimentOutput with all results
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Running capital_country_expert_interp with log level: {log_level}")

    # Validate inputs
    if target_country not in COUNTRY_TO_CAPITAL:
        raise ValueError(f"Unknown country: {target_country}")

    target_capital = COUNTRY_TO_CAPITAL[target_country]
    logger.info(f"Target: {target_country} -> {target_capital}")

    # Load model
    logger.info("=" * 80)
    logger.info("Loading model")
    logger.info("=" * 80)

    model_config = get_model_config(model_name)
    model_ckpt = model_config.get_checkpoint_strict(step=model_step_ckpt)
    model_dtype_torch = get_dtype(model_dtype)

    model = StandardizedTransformer(
        model_config.hf_name,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        revision=str(model_ckpt),
        device_map="auto",
        torch_dtype=model_dtype_torch,
        token=hf_token,
        dispatch=True,
    )
    tokenizer = model.tokenizer

    router_top_k = model.config.num_experts_per_tok

    # Format prompt
    messages = [
        {
            "role": msg["role"],
            "content": msg["content"].format(country=target_country),
        }
        for msg in PROMPT_TEMPLATE
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )

    logger.info(f"Prompt: {formatted}")

    # Tokenize
    batch = tokenizer(formatted, return_tensors="pt", return_attention_mask=False)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # Get correct answer tokens
    correct_answer_tokens = get_correct_answer_tokens(tokenizer, target_capital)

    # Sort tokens by edit distance from target (including initial space)
    target_with_space = f" {target_capital}"
    token_items = list(correct_answer_tokens.items())

    # Compute edit distance for each token
    token_distances = []
    for token, token_id in token_items:
        token_cleaned = token.replace("Ġ", " ")
        distance = levenshtein_distance(token_cleaned, target_with_space)
        token_distances.append((token, token_id, distance))

    # Sort by edit distance (lowest = most correct)
    token_distances.sort(key=lambda x: x[2])

    # Create ordered dictionary
    correct_answer_tokens = {token: token_id for token, token_id, _ in token_distances}

    logger.info(
        f"Correct answer tokens (ordered by edit distance): {correct_answer_tokens}"
    )

    # Step 1: Compute expert importance
    logger.info("=" * 80)
    logger.info("Computing expert importance")
    logger.info("=" * 80)

    expert_importances, baseline_probs, baseline_routing, _intervention_routing = (
        compute_expert_importance_and_routing_masks(
            model=model,
            batch=batch,
            correct_answer_tokens=correct_answer_tokens,
            processing_batch_size=processing_batch_size,
            top_k=router_top_k,
            last_token_only=True,
        )
    )

    logger.info(f"Found {len(expert_importances)} experts")
    logger.info("Top 10 most important experts:")
    for i, exp in enumerate(expert_importances[:10]):
        logger.info(
            f"  {i + 1}. Layer {exp.location.layer_idx}, Expert {exp.location.expert_idx}, "
            f"Token {exp.location.token_idx}, Importance: {exp.importance:.6f}"
        )

    # Step 2: Run incremental ablations
    logger.info("=" * 80)
    logger.info("Running incremental ablations")
    logger.info("=" * 80)

    intervention_results = run_incremental_ablations(
        model=model,
        batch=batch,
        expert_importances=expert_importances,
        correct_answer_tokens=correct_answer_tokens,
        baseline_probs=baseline_probs,
        max_experts_to_ablate=max_experts_to_ablate,
        top_k=router_top_k,
    )

    logger.info("Ablation results:")
    for result in intervention_results:
        logger.info(
            f"  {result.num_experts_ablated} experts: "
            f"prob {result.pre_intervention_prob:.4f} -> {result.post_intervention_prob:.4f} "
            f"(change: {result.prob_change:+.4f}), "
            f"generated: '{result.generated_token}' (p={result.generated_token_prob:.4f})"
        )

    # Step 3: Similarity search
    interpretability_results: list[InterpretabilityResult] = []

    logger.info("=" * 80)
    logger.info("Loading activations for similarity search")
    logger.info("=" * 80)

    activations, _activation_dims = asyncio.run(
        load_activations_and_init_dist(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=0,  # no reshuffling
            submodule_names=[ActivationKeys.ROUTER_LOGITS],
            context_length=context_length,
        )
    )

    # Batch all patterns together: baseline (for pre) + all intervention patterns (for post)
    # We search for baseline once and reuse it for all interventions
    # Pattern order: [baseline, intervention_0, intervention_1, ..., intervention_N-1]
    all_patterns = [baseline_routing] + [
        result.routing_pattern for result in intervention_results
    ]
    final_token_patterns = [pattern[-1] for pattern in all_patterns]

    logger.info(
        f"Searching for similar patterns for {len(final_token_patterns)} targets in one pass..."
    )

    all_similar_results = search_similar_activations_batched(
        target_patterns=final_token_patterns,
        activations=activations,
        tokenizer=tokenizer,
        top_k=top_k_similar,
        max_samples=max_samples_to_search,
        target_batch_size=target_batch_size,
        activation_batch_size=activation_batch_size,
        router_top_k=router_top_k,
        similarity_method=similarity_method,
    )

    # First result is for baseline (used as similar_pre for all interventions)
    similar_pre = all_similar_results[0]

    # Remaining results are for each intervention's post pattern
    for result, similar_post in zip(
        intervention_results, all_similar_results[1:], strict=True
    ):
        interpretability_results.append(
            InterpretabilityResult(
                intervention=result,
                similar_activations_pre=similar_pre,
                similar_activations_post=similar_post,
            )
        )

        logger.info(
            f"  {result.num_experts_ablated} experts ablated: "
            f"found {len(similar_pre)} similar pre, {len(similar_post)} similar post"
        )

    # Build output
    output = ExperimentOutput(
        target_country=target_country,
        target_capital=target_capital,
        prompt=formatted,
        baseline_prob=sum(
            baseline_probs[tid].item() for tid in correct_answer_tokens.values()
        )
        / len(correct_answer_tokens),
        baseline_routing_pattern=baseline_routing,
        expert_importances=expert_importances,
        intervention_results=interpretability_results,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    country_slug = target_country.lower().replace(" ", "_")
    output_file = output_path / f"{country_slug}.yaml"

    # Convert to serializable format
    # baseline_routing_pattern is (T, L, E), extract last token pattern (L, E)
    baseline_routing_last_token = (
        output.baseline_routing_pattern[-1].cpu().numpy()
    )  # (L, E)

    save_data = {
        "target_country": output.target_country,
        "target_capital": output.target_capital,
        "prompt": output.prompt,
        "baseline_prob": output.baseline_prob,
        "baseline_routing_mask": baseline_routing_last_token.tolist(),
        "expert_importances": [
            {
                "token_idx": exp.location.token_idx,
                "layer_idx": exp.location.layer_idx,
                "expert_idx": exp.location.expert_idx,
                "importance": exp.importance,
            }
            for exp in output.expert_importances
        ],
        "intervention_results": [
            {
                "num_experts_ablated": ir.intervention.num_experts_ablated,
                "ablated_experts": [
                    {
                        "token_idx": loc.token_idx,
                        "layer_idx": loc.layer_idx,
                        "expert_idx": loc.expert_idx,
                    }
                    for loc in ir.intervention.ablated_experts
                ],
                "pre_intervention_prob": ir.intervention.pre_intervention_prob,
                "post_intervention_prob": ir.intervention.post_intervention_prob,
                "prob_change": ir.intervention.prob_change,
                # routing_pattern is (T, L, E) or (L, E), extract last token if needed
                "routing_mask": (
                    ir.intervention.routing_pattern[-1].cpu().numpy().tolist()
                    if ir.intervention.routing_pattern.ndim == 3
                    else ir.intervention.routing_pattern.cpu().numpy().tolist()
                ),
                "generated_token": ir.intervention.generated_token,
                "generated_token_prob": ir.intervention.generated_token_prob,
                "similar_pre": [
                    {
                        "similarity_score": float(sa.similarity_score),
                        "routing_mask": sa.routing_mask.cpu().numpy().tolist(),
                        "text_passage": sa.text_passage,
                        "activating_sample_slice": list(sa.activating_sample_slice),
                        "highlighted_text_passage": sa.highlighted_text_passage(),
                    }
                    for sa in ir.similar_activations_pre
                ],
                "similar_post": [
                    {
                        "similarity_score": float(sa.similarity_score),
                        "routing_mask": sa.routing_mask.cpu().numpy().tolist(),
                        "text_passage": sa.text_passage,
                        "activating_sample_slice": list(sa.activating_sample_slice),
                        "highlighted_text_passage": sa.highlighted_text_passage(),
                    }
                    for sa in ir.similar_activations_post
                ],
            }
            for ir in output.intervention_results
        ],
    }

    with open(output_file, "w") as f:
        yaml.dump(save_data, f)
    logger.info(f"Saved results to {output_file}")

    # Print summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target: {target_country} -> {target_capital}")
    logger.info(f"Baseline probability: {output.baseline_prob:.4f}")
    logger.info(f"Number of experts analyzed: {len(expert_importances)}")
    logger.info(f"Number of interventions: {len(interpretability_results)}")

    return output


if __name__ == "__main__":
    arguably.run()
