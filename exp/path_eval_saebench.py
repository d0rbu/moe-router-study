import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import os
import random
import sys
from typing import Any

import arguably
from dotenv import load_dotenv
from loguru import logger
from nnterp import StandardizedTransformer
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    output_folders as EVAL_DIRS,
)
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
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
from tabulate import tabulate
import torch as th
from tqdm import tqdm
import yaml

from core.dtype import get_dtype
from core.model import get_model_config
from exp import MODEL_DIRNAME, OUTPUT_DIR
from exp.kmeans import KMEANS_FILENAME, KMEANS_TYPE

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class Paths:
    data: th.Tensor  # (num_centroids, num_layers * num_experts)
    name: str
    metadata: dict[str, Any]


@dataclass
class PathsWithSparsity(Paths):
    sparsity: th.Tensor  # (num_centroids)


def collect_path_activations(
    tokenized_dataset: th.Tensor,
    model: StandardizedTransformer,
    paths: Paths | PathsWithSparsity,
    llm_batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
    selected_paths: list[int] | None = None,
    activation_dtype: th.dtype | None = None,
) -> dict[int, th.Tensor]:
    raise NotImplementedError("Not implemented. Implement knn inference here.")


def get_feature_activation_sparsity(
    tokens: th.Tensor,  # dataset_size x seq_len
    model: StandardizedTransformer,
    paths_data: th.Tensor,
    batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
) -> th.Tensor:  # d_paths
    raise NotImplementedError(
        "Not implemented. Implement feature activation sparsity here. Should be used in collect_path_activations."
    )


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
        tokenized_dataset: th.Tensor,
        sparsity: th.Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.paths = paths
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

        acts = collect_path_activations(
            self.tokenized_dataset,
            self.model,
            self.paths,
            self.cfg.llm_batch_size,
            mask_bos_pad_eos_tokens=True,
            selected_paths=self.latents,
            activation_dtype=th.bfloat16,  # reduce memory usage, we don't need full precision when sampling activations
        )

        generation_examples = {}
        scoring_examples = {}

        for i, latent in tqdm(
            enumerate(self.latents), desc="Collecting examples for LLM judge"
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
        tokenized_dataset = dataset_utils.load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            model.tokenizer,  # type: ignore
        ).to(device)
        th.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if isinstance(paths, Paths):
        sparsity = get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            paths.data,
            config.llm_batch_size,
            mask_bos_pad_eos_tokens=True,
        )
        paths = PathsWithSparsity(
            data=paths.data,
            name=paths.name,
            metadata=paths.metadata,
            sparsity=sparsity,
        )

    autointerp_runner = PathAutoInterp(
        cfg=config,
        model=model,
        paths=paths.data,
        tokenized_dataset=tokenized_dataset,
        sparsity=paths.sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp_runner.run())

    return results


def run_autointerp_eval(
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
        device_map="auto",
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


@arguably.command()
def main(
    *,
    experiment_dir: str,
    model_name: str = "olmoe-i",
    batchsize: int = 512,
    dtype: str = "float32",
    seed: int = 0,
    logs_path: str | None = None,
    log_level: str = "INFO",
) -> None:
    """
    Evaluate the paths on the given model.
    """

    th_dtype = get_dtype(dtype)
    str_dtype = th_dtype.__str__().split(".")[-1]

    device = general_utils.setup_environment()

    experiment_path = os.path.join(OUTPUT_DIR, experiment_dir)

    config_path = os.path.join(experiment_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["type"] == KMEANS_TYPE, (
        f"Experiment is not a kmeans experiment, type={config['type']}"
    )
    assert config["model_name"] == model_name, (
        f"Model name mismatch: {model_name} != {config['model_name']}"
    )

    paths_set = []
    with open(os.path.join(experiment_path, KMEANS_FILENAME)) as f:
        kmeans_data = th.load(f)

        # list of tensors of shape (num_centroids, num_layers * num_experts)
        centroid_sets = kmeans_data["centroids"]
        top_k = kmeans_data["top_k"]
        losses = kmeans_data["losses"]

        paths = Paths(
            data=centroid_sets,
            name=f"paths_{centroid_sets.shape[0]}",
            metadata={
                "num_paths": centroid_sets.shape[0],
                "top_k": top_k,
                "losses": losses.tolist(),
            },
        )
        paths_set.append(paths)

    # run autointerp
    autointerp_eval_dir = EVAL_DIRS["autointerp"]
    autointerp_eval_dir = os.path.join(OUTPUT_DIR, autointerp_eval_dir)
    run_autointerp_eval(
        config=AutoInterpEvalConfig(
            model_name=model_name,
            random_seed=seed,
            llm_batch_size=batchsize,
            llm_dtype=str_dtype,
        ),
        selected_paths_set=paths_set,
        device=device,
        api_key=OPENAI_API_KEY,
        output_path=autointerp_eval_dir,
        force_rerun=False,
        save_logs_path=logs_path,
        artifacts_path=os.path.join(experiment_path, "artifacts"),
        log_level=log_level,
    )

    raise NotImplementedError("Not implemented. Implement sparse probing here.")
