"""
SAEBench autointerp evaluation for raw model activations.

This script evaluates raw model activations from specified layers and activation types
by collecting activations directly from layers and running autointerp evaluation on them.
Based on exp/autointerp_saebench.py but adapted for raw activations instead of router paths.
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import os
import random
import sys
from typing import Any

from loguru import logger
from nnterp import StandardizedTransformer
from sae_bench.evals.autointerp import main as autointerp
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.eval_output import (
    EVAL_TYPE_ID_AUTOINTERP,
    AutoInterpEvalOutput,
    AutoInterpMetricCategories,
    AutoInterpMetrics,
)
from sae_bench.sae_bench_utils import (
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.activation_collection import get_bos_pad_eos_mask
from sae_bench.sae_bench_utils.dataset_utils import (
    load_and_tokenize_dataset,
)
from tabulate import tabulate
import torch as th
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from core.memory import clear_memory
from core.model import get_model_config
from core.type import assert_type
from exp import MODEL_DIRNAME
from exp.autointerp_saebench import Example
from exp.get_activations import ActivationKeys


@dataclass
class RawActivations:
    """Container for raw activation data."""

    data: th.Tensor  # (num_features, total_hidden_size)
    name: str
    metadata: dict[str, Any]


@dataclass
class RawActivationsWithSparsity(RawActivations):
    sparsity: th.Tensor  # (num_features)


def collect_raw_activations(
    tokenized_dataset: th.Tensor,
    model: StandardizedTransformer,
    activation_key: ActivationKeys,
    layers: list[int],
    llm_batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
    selected_features: list[int] | None = None,
    activation_dtype: th.dtype | None = None,
) -> th.Tensor:
    """Collects raw activations for a given set of tokens.
    
    Args:
        tokenized_dataset: Tokenized dataset (B, T)
        model: Model to extract activations from
        activation_key: Type of activation to extract
        layers: List of layer indices to extract from
        llm_batch_size: Batch size for processing
        mask_bos_pad_eos_tokens: Whether to mask BOS/PAD/EOS tokens
        selected_features: Optional list of feature indices to select
        activation_dtype: Optional dtype to cast activations to
        
    Returns:
        Tensor of shape (B, T, F) where F is the total activation dimension
    """
    raw_acts = []
    layers_sorted = sorted(layers)

    for batch_idx, tokens_BT in tqdm(
        enumerate(th.split(tokenized_dataset, llm_batch_size, dim=0)),
        total=tokenized_dataset.shape[0] // llm_batch_size,
        desc="Collecting raw activations",
        leave=False,
    ):
        layer_activations = []

        # Use trace context manager to capture activations
        with model.trace(tokens_BT):
            # Extract activations for each specified layer
            for layer_idx in tqdm(
                layers_sorted,
                desc=f"Batch {batch_idx}",
                total=len(layers_sorted),
                leave=False,
            ):
                # Extract activation based on type
                match activation_key:
                    case ActivationKeys.LAYER_OUTPUT:
                        activation = model.layers_output[layer_idx].save()
                    case ActivationKeys.MLP_OUTPUT:
                        activation = model.mlps_output[layer_idx].save()
                    case ActivationKeys.ATTN_OUTPUT:
                        activation = model.attentions_output[layer_idx].save()
                    case _:
                        raise ValueError(
                            f"Unsupported activation key: {activation_key}"
                        )

                layer_activations.append(activation)

        # Concatenate activations across layers: (B, T, sum(hidden_sizes))
        concat_activations = th.cat(layer_activations, dim=-1)

        if selected_features is not None:
            concat_activations = concat_activations[:, :, selected_features]

        if mask_bos_pad_eos_tokens:
            attn_mask_BT = get_bos_pad_eos_mask(tokens_BT, model.tokenizer)
        else:
            attn_mask_BT = th.ones_like(tokens_BT, dtype=th.bool)

        attn_mask_BT = attn_mask_BT.to(device=concat_activations.device)

        concat_activations = concat_activations * attn_mask_BT[:, :, None]

        if activation_dtype is not None:
            concat_activations = concat_activations.to(dtype=activation_dtype)

        raw_acts.append(concat_activations)

    all_raw_acts_BTF = th.cat(raw_acts, dim=0)
    return all_raw_acts_BTF


def get_feature_activation_sparsity(
    tokens: th.Tensor,  # dataset_size x seq_len
    model: StandardizedTransformer,
    activation_key: ActivationKeys,
    layers: list[int],
    batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
) -> th.Tensor:  # num_features
    """Get the activation sparsity for each feature dimension."""
    device = tokens.device
    
    # Get total feature dimension by doing a dummy forward pass
    dummy_tokens = tokens[:1, :1]
    with th.no_grad():
        with model.trace(dummy_tokens):
            layer_activations = []
            for layer_idx in sorted(layers):
                match activation_key:
                    case ActivationKeys.LAYER_OUTPUT:
                        activation = model.layers_output[layer_idx].save()
                    case ActivationKeys.MLP_OUTPUT:
                        activation = model.mlps_output[layer_idx].save()
                    case ActivationKeys.ATTN_OUTPUT:
                        activation = model.attentions_output[layer_idx].save()
                    case _:
                        raise ValueError(f"Unsupported activation key: {activation_key}")
                layer_activations.append(activation)
            concat_activations = th.cat(layer_activations, dim=-1)
            num_features = concat_activations.shape[-1]
    
    running_sum_F = th.zeros(num_features, dtype=th.float32, device=device)
    total_tokens = 0

    for batch_idx, tokens_BT in tqdm(
        enumerate(th.split(tokens, batch_size, dim=0)),
        total=tokens.shape[0] // batch_size,
        desc="Getting activation sparsity",
        leave=False,
    ):
        layer_activations = []

        with model.trace(tokens_BT):
            for layer_idx in tqdm(
                sorted(layers),
                desc=f"Batch {batch_idx}",
                total=len(layers),
                leave=False,
            ):
                match activation_key:
                    case ActivationKeys.LAYER_OUTPUT:
                        activation = model.layers_output[layer_idx].save()
                    case ActivationKeys.MLP_OUTPUT:
                        activation = model.mlps_output[layer_idx].save()
                    case ActivationKeys.ATTN_OUTPUT:
                        activation = model.attentions_output[layer_idx].save()
                    case _:
                        raise ValueError(f"Unsupported activation key: {activation_key}")
                layer_activations.append(activation)

        concat_activations_BTF = th.cat(layer_activations, dim=-1)

        if mask_bos_pad_eos_tokens:
            attn_mask_BT = get_bos_pad_eos_mask(tokens_BT, model.tokenizer)
        else:
            attn_mask_BT = th.ones_like(tokens_BT, dtype=th.bool)

        attn_mask_BT = attn_mask_BT.to(device=concat_activations_BTF.device)

        concat_activations_BTF = concat_activations_BTF * attn_mask_BT[:, :, None]
        total_tokens += attn_mask_BT.sum().item()

        # Count non-zero activations
        running_sum_F += (concat_activations_BTF != 0).float().sum(dim=(0, 1))

    return running_sum_F / total_tokens


class RawActivationsAutoInterp(autointerp.AutoInterp):
    """
    AutoInterp implementation for raw activations.
    
    This collects raw activations from the model and generates explanations
    for individual feature dimensions in the concatenated activation space.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: StandardizedTransformer,
        activation_key: ActivationKeys,
        layers: list[int],
        tokenized_dataset: th.Tensor,
        sparsity: th.Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.activation_key = activation_key
        self.layers = layers
        self.tokenized_dataset = tokenized_dataset
        self.device = device
        self.api_key = api_key
        
        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            assert self.cfg.n_latents is not None
            sparsity *= cfg.total_tokens
            alive_latents = (
                th.nonzero(sparsity > self.cfg.dead_latent_threshold)
                .squeeze(1)
                .tolist()
            )
            if len(alive_latents) < self.cfg.n_latents:
                self.latents = alive_latents
                print(
                    f"\n\n\nWARNING: Found only {len(alive_latents)} alive latents, which is less than {self.cfg.n_latents}\n\n\n"
                )
            else:
                self.latents = random.sample(alive_latents, k=self.cfg.n_latents)
        self.n_latents = len(self.latents)

    def gather_data(
        self,
    ) -> tuple[dict[int, autointerp.Examples], dict[int, autointerp.Examples]]:
        """
        Stores top acts / random seqs data, which is used for generation & scoring respectively.
        """
        dataset_size, seq_len = self.tokenized_dataset.shape

        # Collect raw activations (B, T, F)
        acts = collect_raw_activations(
            self.tokenized_dataset,
            self.model,
            self.activation_key,
            self.layers,
            self.cfg.llm_batch_size,
            mask_bos_pad_eos_tokens=True,
            selected_features=self.latents,
            activation_dtype=th.bfloat16,  # reduce memory usage
        )

        generation_examples = {}
        scoring_examples = {}

        for i, latent in tqdm(
            enumerate(self.latents),
            total=len(self.latents),
            desc="Collecting examples for LLM judge",
            leave=False,
        ):
            # (1/3) Get random examples (we don't need their values)
            rand_indices = th.stack(
                [
                    th.randint(0, dataset_size, (self.cfg.n_random_ex_for_scoring,)),
                    th.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = autointerp.index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer
            )

            # (2/3) Get top-scoring examples
            top_indices = autointerp.get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = autointerp.index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=self.cfg.buffer
            )
            top_values = autointerp.index_with_buffer(
                acts[..., i], top_indices, buffer=self.cfg.buffer
            )
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Get importance-weighted examples, using a threshold so they're disjoint from top examples
            # Also, if we don't have enough values, then we assume this is a dead feature & continue
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = th.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = autointerp.get_iw_sample_indices(
                acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
            )
            iw_toks = autointerp.index_with_buffer(
                self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer
            )
            iw_values = autointerp.index_with_buffer(
                acts[..., i], iw_indices, buffer=self.cfg.buffer
            )

            # Get random values to use for splitting
            rand_top_ex_split_indices = th.randperm(self.cfg.n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[
                : self.cfg.n_top_ex_for_generation
            ]
            top_scoring_indices = rand_top_ex_split_indices[
                self.cfg.n_top_ex_for_generation :
            ]
            rand_iw_split_indices = th.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[
                : self.cfg.n_iw_sampled_ex_for_generation
            ]
            iw_scoring_indices = rand_iw_split_indices[
                self.cfg.n_iw_sampled_ex_for_generation :
            ]

            def create_examples(
                all_toks: th.Tensor,
                all_acts: th.Tensor | None = None,
                act_threshold: float | None = None,
            ) -> list[autointerp.Example]:
                if all_acts is None:
                    all_acts = th.zeros_like(all_toks).float()
                # Use 0.0 as default threshold if None provided
                threshold = act_threshold if act_threshold is not None else 0.0
                return [
                    Example(
                        toks=toks,
                        str_toks=self.model.tokenizer.batch_decode(
                            toks, clean_up_tokenization_spaces=False
                        ),
                        acts=acts,
                        act_threshold=threshold,
                    )
                    for (toks, acts) in zip(
                        all_toks.tolist(), all_acts.tolist(), strict=False
                    )
                ]

            # Get the generation & scoring examples
            generation_examples[latent] = autointerp.Examples(
                create_examples(
                    top_toks[top_gen_indices],
                    top_values[top_gen_indices],
                    act_threshold,
                )
                + create_examples(
                    iw_toks[iw_gen_indices],
                    iw_values[iw_gen_indices],
                    act_threshold,
                ),
            )
            scoring_examples[latent] = autointerp.Examples(
                create_examples(
                    top_toks[top_scoring_indices],
                    top_values[top_scoring_indices],
                    act_threshold,
                )
                + create_examples(
                    iw_toks[iw_scoring_indices],
                    iw_values[iw_scoring_indices],
                    act_threshold,
                )
                + create_examples(rand_toks, act_threshold=act_threshold),
                shuffle=True,
            )

        return generation_examples, scoring_examples


def run_eval_raw_activations(
    config: AutoInterpEvalConfig,
    raw_activations: RawActivations | RawActivationsWithSparsity,
    activation_key: ActivationKeys,
    layers: list[int],
    model: StandardizedTransformer,
    device: str,
    artifacts_folder: str,
    api_key: str,
    sparsity: th.Tensor | None = None,
) -> dict[int, dict[str, Any]]:
    random.seed(config.random_seed)
    th.manual_seed(config.random_seed)
    th.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    tokens_filename = f"{autointerp.escape_slash(config.model_name)}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = th.load(tokens_path).to(device)
    else:
        tokenized_dataset = load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            assert_type(model.tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast),
        ).to(device)
        th.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if isinstance(raw_activations, RawActivations):
        sparsity = get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            activation_key,
            layers,
            config.llm_batch_size,
            mask_bos_pad_eos_tokens=True,
        )
        raw_activations = RawActivationsWithSparsity(
            data=raw_activations.data,
            name=raw_activations.name,
            metadata=raw_activations.metadata,
            sparsity=sparsity,
        )

    autointerp_runner = RawActivationsAutoInterp(
        cfg=config,
        model=model,
        activation_key=activation_key,
        layers=layers,
        tokenized_dataset=tokenized_dataset,
        sparsity=raw_activations.sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp_runner.run())

    return results


def run_eval(
    config: AutoInterpEvalConfig,
    activation_key: ActivationKeys,
    layers: list[int],
    device: str,
    api_key: str,
    output_path: str,
    force_rerun: bool = False,
    save_logs_path: str | None = None,
    artifacts_path: str = "artifacts",
    log_level: str = "INFO",
) -> dict[str, Any]:
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    logger.trace(f"Using config: {config}")

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Get model config
    model_config = get_model_config(config.model_name)

    hf_name = model_config.hf_name
    local_path = os.path.join(os.path.abspath(MODEL_DIRNAME), hf_name)
    path = local_path if os.path.exists(local_path) else hf_name

    logger.info(f"Using model from {path}")
    # Initialize model
    model: StandardizedTransformer = StandardizedTransformer(
        path,
        check_attn_probs_with_trace=False,
        check_renaming=False,
        device_map=device,
        torch_dtype=llm_dtype,
    )

    # Create RawActivations object
    layers_str = "_".join(map(str, sorted(layers)))
    name = f"raw_{activation_key}_layers_{layers_str}"
    
    # Get total feature dimension
    dummy_tokens = th.zeros((1, 1), dtype=th.long, device=device)
    with th.no_grad():
        with model.trace(dummy_tokens):
            layer_activations = []
            for layer_idx in sorted(layers):
                match activation_key:
                    case ActivationKeys.LAYER_OUTPUT:
                        activation = model.layers_output[layer_idx].save()
                    case ActivationKeys.MLP_OUTPUT:
                        activation = model.mlps_output[layer_idx].save()
                    case ActivationKeys.ATTN_OUTPUT:
                        activation = model.attentions_output[layer_idx].save()
                    case _:
                        raise ValueError(f"Unsupported activation key: {activation_key}")
                layer_activations.append(activation)
            concat_activations = th.cat(layer_activations, dim=-1)
            total_dim = concat_activations.shape[-1]
    
    raw_activations = RawActivations(
        data=th.eye(total_dim, device=device),  # Identity matrix for direct feature access
        name=name,
        metadata={
            "activation_key": str(activation_key),
            "layers": sorted(layers),
            "total_dim": total_dim,
        },
    )

    sae_result_path = os.path.join(output_path, f"{name}_eval_results.json")

    if os.path.exists(sae_result_path) and not force_rerun:
        print(f"Skipping {name} as results already exist")
        return results_dict

    artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_AUTOINTERP)

    raw_eval_result = run_eval_raw_activations(
        config, raw_activations, activation_key, layers, model, device, artifacts_folder, api_key, None
    )

    # Save nicely formatted logs to a text file, helpful for debugging.
    if save_logs_path is not None:
        # Get summary results for all latents, as well logs for the best and worst-scoring latents
        headers = [
            "latent",
            "explanation",
            "predictions",
            "correct seqs",
            "score",
        ]
        logs = "Summary table:\n" + tabulate(
            [
                [raw_eval_result[latent][h] for h in headers]
                for latent in raw_eval_result
            ],
            headers=headers,
            tablefmt="simple_outline",
        )
        worst_result = min(raw_eval_result.values(), key=lambda x: x["score"])
        best_result = max(raw_eval_result.values(), key=lambda x: x["score"])
        logs += f"\n\nWorst scoring idx {worst_result['latent']}, score = {worst_result['score']}\n{worst_result['logs']}"
        logs += f"\n\nBest scoring idx {best_result['latent']}, score = {best_result['score']}\n{best_result['logs']}"
        # Save the results to a file
        with open(save_logs_path, "a") as f:
            f.write(logs)

    # Put important results into the results dict
    all_scores = [r["score"] for r in raw_eval_result.values()]

    all_scores_tensor = th.tensor(all_scores)
    score = all_scores_tensor.mean().item()
    std_dev = all_scores_tensor.std().item()

    eval_output = AutoInterpEvalOutput(
        eval_config=config,
        eval_id=eval_instance_id,
        datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
        eval_result_metrics=AutoInterpMetricCategories(
            autointerp=AutoInterpMetrics(
                autointerp_score=score, autointerp_std_dev=std_dev
            )
        ),
    )
    eval_output.eval_result_details = []
    eval_output.eval_result_unstructured = raw_eval_result
    eval_output.sae_bench_commit_hash = sae_bench_commit_hash
    eval_output.sae_lens_id = "raw_activations"
    eval_output.sae_lens_release_id = name
    eval_output.sae_lens_version = sae_lens_version
    eval_output.sae_cfg_dict = raw_activations.metadata

    results_dict[name] = asdict(eval_output)

    eval_output.to_json_file(sae_result_path, indent=2)

    gc.collect()
    clear_memory()

    return results_dict
