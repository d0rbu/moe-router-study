import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
from itertools import batched
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
from tabulate import tabulate
import torch as th
from tqdm import tqdm

from core.model import get_model_config
from exp import MODEL_DIRNAME


@dataclass
class Paths:
    data: th.Tensor  # (num_centroids, num_layers * num_experts)
    top_k: int
    name: str
    metadata: dict[str, Any]

    @classmethod
    def expert_aligned_paths(
        cls,
        top_k: int,
        name: int,
        num_total_experts: int,
        metadata: dict[str, Any] | None = None,
    ) -> "Paths":
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "num_paths": num_total_experts,
                "top_k": top_k,
            },
        )
        return cls(
            data=th.eye(num_total_experts),
            top_k=top_k,
            name=name,
            metadata=metadata,
        )


@dataclass
class PathsWithSparsity(Paths):
    sparsity: th.Tensor  # (num_centroids)


def collect_path_activations(
    tokenized_dataset: th.Tensor,
    model: StandardizedTransformer,
    paths: Paths | PathsWithSparsity,
    top_k: int,
    llm_batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
    selected_paths: list[int] | None = None,
    activation_dtype: th.dtype | None = None,
) -> dict[int, th.Tensor]:
    """Collects path activations for a given set of tokens."""
    path_acts = []

    for batch_idx, tokens_BT in tqdm(
        enumerate(batched(tokenized_dataset, llm_batch_size)),
        total=tokenized_dataset.shape[0] // llm_batch_size,
        desc="Collecting path activations",
        leave=False,
    ):
        router_logits_set = []

        # use trace context manager to capture router outputs
        with model.trace(tokens_BT):
            # extract activations for each layer
            for layer_idx in tqdm(
                model.layers_with_routers,
                desc=f"Batch {batch_idx}",
                total=len(model.layers_with_routers),
                leave=False,
            ):
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

                router_logits_set.append(logits)

        # (B, T, L, E)
        router_paths = th.cat(router_logits_set, dim=-2)
        sparse_paths = th.topk(router_paths, k=top_k, dim=-1).indices

        router_paths.zero_()
        router_paths.scatter_(-1, sparse_paths, 1)

        del sparse_paths

        # (B, T, L, E) -> (B, T, L * E)
        router_paths_BTP = router_paths.view(*tokens_BT.shape, -1)

        # (B, T, L * E) @ (L * E, F) -> (B, T, F)
        router_paths_BTF = router_paths_BTP @ paths.data.T

        del router_paths, router_paths_BTP

        if selected_paths is not None:
            router_paths_BTF = router_paths_BTF[:, :, selected_paths]

        if mask_bos_pad_eos_tokens:
            attn_mask_BT = get_bos_pad_eos_mask(tokens_BT, model.tokenizer)
        else:
            attn_mask_BT = th.ones_like(tokens_BT, dtype=th.bool)

        attn_mask_BT = attn_mask_BT.to(device=router_paths_BTF.device)

        router_paths_BTF = router_paths_BTF * attn_mask_BT[:, :, None]

        if activation_dtype is not None:
            router_paths_BTF = router_paths_BTF.to(dtype=activation_dtype)

        path_acts.append(router_paths_BTF)

    all_path_acts_BTF = th.cat(path_acts, dim=0)
    return all_path_acts_BTF


def get_feature_activation_sparsity(
    tokens: th.Tensor,  # dataset_size x seq_len
    model: StandardizedTransformer,
    paths: th.Tensor,
    top_k: int,
    batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
) -> th.Tensor:  # num_paths
    """Get the activation sparsity for each path."""
    device = paths.device
    running_sum_F = th.zeros(paths.shape[0], dtype=th.float32, device=device)
    total_tokens = 0

    for batch_idx, tokens_BT in tqdm(
        enumerate(batched(tokens, batch_size)),
        total=tokens.shape[0] // batch_size,
        desc="Getting path activation sparsity",
        leave=False,
    ):
        router_logits_set = []

        with model.trace(tokens_BT):
            for layer_idx in tqdm(
                model.layers_with_routers,
                desc=f"Batch {batch_idx}",
                total=len(model.layers_with_routers),
                leave=False,
            ):
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

                router_logits_set.append(logits)

        router_paths = th.cat(router_logits_set, dim=-2)
        sparse_paths = th.topk(router_paths, k=top_k, dim=-1).indices

        router_paths.zero_()
        router_paths.scatter_(-1, sparse_paths, 1)

        del sparse_paths

        router_paths_BTP = router_paths.view(*tokens_BT.shape, -1)

        # one-hot encode the closest path to each token
        # (B, T, L * E) . (F, L * E) -> (B, T, F)
        distances = th.cdist(router_paths_BTP, paths.data, p=1)
        # (B, T, F) -> (B, T)
        closest_paths = th.argmin(distances, dim=-1)
        router_paths_BTF = th.zeros_like(distances)
        router_paths_BTF.scatter_(-1, closest_paths, 1)

        del distances, closest_paths

        if mask_bos_pad_eos_tokens:
            attn_mask_BT = get_bos_pad_eos_mask(tokens_BT, model.tokenizer)
        else:
            attn_mask_BT = th.ones_like(tokens_BT, dtype=th.bool)

        attn_mask_BT = attn_mask_BT.to(device=router_paths_BTF.device)

        router_paths_BTF = router_paths_BTF * attn_mask_BT[:, :, None]
        total_tokens += attn_mask_BT.sum().item()

        running_sum_F += th.sum(router_paths_BTF, dim=(0, 1))

    return running_sum_F / total_tokens


class PathAutoInterp(autointerp.AutoInterp):
    """
    This is a start-to-end class for generating explanations and optionally scores. It's easiest to implement it as a
    single class for the time being because there's data we'll need to fetch that'll be used in both the generation and
    scoring phases.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: StandardizedTransformer,
        paths: th.Tensor,
        top_k: int,
        tokenized_dataset: th.Tensor,
        sparsity: th.Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.paths = paths
        self.top_k = top_k
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

        # (B, L * E)
        acts = collect_path_activations(
            self.tokenized_dataset,
            self.model,
            self.paths,
            self.top_k,
            self.cfg.llm_batch_size,
            mask_bos_pad_eos_tokens=True,
            selected_paths=self.latents,
            activation_dtype=th.bfloat16,  # reduce memory usage, we don't need full precision when sampling activations
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
                return [
                    autointerp.Example(
                        toks=toks,
                        acts=acts,
                        act_threshold=act_threshold,
                        model=self.model,
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


def run_eval_paths(
    config: AutoInterpEvalConfig,
    paths: Paths | PathsWithSparsity,
    model: StandardizedTransformer,
    device: str,
    artifacts_folder: str,
    api_key: str,
    sparsity: th.Tensor | None = None,
) -> dict[str, float]:
    random.seed(config.random_seed)
    th.manual_seed(config.random_seed)
    th.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    tokens_filename = f"{autointerp.escape_slash(config.model_name)}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = th.load(tokens_path).to(device)
    else:
        from sae_bench.sae_bench_utils.dataset_utils import (
            load_and_tokenize_dataset,
        )

        tokenized_dataset = load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            model.tokenizer,
        ).to(device)
        th.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if isinstance(paths, Paths):
        sparsity = get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            paths.data,
            paths.top_k,
            config.llm_batch_size,
            mask_bos_pad_eos_tokens=True,
        )
        paths = PathsWithSparsity(
            data=paths.data,
            top_k=paths.top_k,
            name=paths.name,
            metadata=paths.metadata,
            sparsity=sparsity,
        )

    autointerp_runner = PathAutoInterp(
        cfg=config,
        model=model,
        paths=paths.data,
        top_k=paths.top_k,
        tokenized_dataset=tokenized_dataset,
        sparsity=paths.sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp_runner.run())

    return results


def run_eval(
    config: AutoInterpEvalConfig,
    selected_paths_set: list[Paths | PathsWithSparsity],
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
        device_map=device,
        dtype=llm_dtype,
    )

    for paths_with_metadata in tqdm(
        selected_paths_set,
        total=len(selected_paths_set),
        desc="Autointerp",
    ):
        sae_result_path = os.path.join(
            output_path, f"{paths_with_metadata.name}_eval_results.json"
        )

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {paths_with_metadata.name} as results already exist")
            continue

        artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_AUTOINTERP)

        paths_eval_result = run_eval_paths(
            config, paths_with_metadata, model, device, artifacts_folder, api_key, None
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
                    [paths_eval_result[latent][h] for h in headers]
                    for latent in paths_eval_result
                ],
                headers=headers,
                tablefmt="simple_outline",
            )
            worst_result = min(paths_eval_result.values(), key=lambda x: x["score"])
            best_result = max(paths_eval_result.values(), key=lambda x: x["score"])
            logs += f"\n\nWorst scoring idx {worst_result['latent']}, score = {worst_result['score']}\n{worst_result['logs']}"
            logs += f"\n\nBest scoring idx {best_result['latent']}, score = {best_result['score']}\n{best_result['logs']}"
            # Save the results to a file
            with open(save_logs_path, "a") as f:
                f.write(logs)

        # Put important results into the results dict
        all_scores = [r["score"] for r in paths_eval_result.values()]

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
            eval_result_details=[],
            eval_result_unstructured=paths_eval_result,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id="paths",
            sae_lens_release_id=paths_with_metadata.name,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=paths_with_metadata.metadata,
        )

        results_dict[f"{paths_with_metadata.name}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        th.cuda.empty_cache()

    return results_dict
