from dataclasses import asdict
from datetime import datetime
import gc
from itertools import batched
import os
import random
import shutil
import sys
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from transformers import AutoTokenizer

from loguru import logger
from nnterp import StandardizedTransformer
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.evals.sparse_probing.eval_output import (
    EVAL_TYPE_ID_SPARSE_PROBING,
    SparseProbingEvalOutput,
    SparseProbingLlmMetrics,
    SparseProbingMetricCategories,
    SparseProbingResultDetail,
    SparseProbingSaeMetrics,
)
from sae_bench.evals.sparse_probing.probe_training import train_probe_on_activations
from sae_bench.sae_bench_utils import (
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.activation_collection import (
    create_meaned_model_activations,
    get_bos_pad_eos_mask,
)
from sae_bench.sae_bench_utils.dataset_info import chosen_classes_per_dataset
from sae_bench.sae_bench_utils.dataset_utils import (
    filter_dataset,
    get_multi_label_train_test_data,
    tokenize_data_dictionary,
)
import torch as th
from tqdm import tqdm

from core.model import get_model_config
from exp import MODEL_DIRNAME
from exp.autointerp_saebench import Paths


@th.no_grad()
def get_llm_activations(
    tokens: th.Tensor,  # (B, T)
    model: StandardizedTransformer,
    batch_size: int,
    top_k: int,
    mask_bos_pad_eos_tokens: bool = False,
    show_progress: bool = True,
) -> th.Tensor:  # (B, T, P)
    """Collects activations for an LLM model from a given layer for a given set of tokens.
    VERY IMPORTANT NOTE: If mask_bos_pad_eos_tokens is True, we zero out activations for BOS, PAD, and EOS tokens.
    Later, we ignore zeroed activations."""

    all_acts_BTP = []

    for batch_idx, tokens_BT in tqdm(
        enumerate(batched(tokens, batch_size)),
        total=tokens.shape[0] // batch_size,
        desc="Collecting activations",
        disable=not show_progress,
    ):
        with model.trace(tokens_BT):
            acts_BTLE = [
                model.routers_output[layer_idx].save()
                for layer_idx in tqdm(
                    model.layers_with_routers,
                    desc=f"Batch {batch_idx}",
                    total=len(model.layers_with_routers),
                    leave=False,
                )
            ]

        acts_BTLE = th.stack(acts_BTLE, dim=-2)
        sparse_paths = th.topk(acts_BTLE, k=top_k, dim=-1).indices
        acts_BTLE.zero_()
        acts_BTLE.scatter_(-1, sparse_paths, 1)
        del sparse_paths

        acts_BTP = acts_BTLE.view(*tokens_BT.shape, -1)
        del acts_BTLE

        if mask_bos_pad_eos_tokens:
            attn_mask_BT = get_bos_pad_eos_mask(tokens_BT, model.tokenizer)
            acts_BTP *= attn_mask_BT[:, :, None]

        all_acts_BTP.append(acts_BTP)

    return th.cat(all_acts_BTP, dim=0)


@th.no_grad()
def get_all_llm_activations(
    tokenized_inputs_dict: dict[
        str, dict[str, th.Tensor]  # (B, T)
    ],
    model: StandardizedTransformer,
    batch_size: int,
    top_k: int,
    mask_bos_pad_eos_tokens: bool = False,
) -> dict[str, th.Tensor]:  # (B, T, P)
    """If we have a dictionary of tokenized inputs for different classes, this function collects activations for all classes.
    We assume that the tokenized inputs have both the input_ids and attention_mask keys.
    VERY IMPORTANT NOTE: We zero out masked token activations in this function. Later, we ignore zeroed activations."""
    all_classes_acts_BTP = {}

    for class_name in tokenized_inputs_dict:
        tokens = tokenized_inputs_dict[class_name]["input_ids"]

        acts_BTP = get_llm_activations(
            tokens,
            model=model,
            batch_size=batch_size,
            top_k=top_k,
            mask_bos_pad_eos_tokens=mask_bos_pad_eos_tokens,
        )

        all_classes_acts_BTP[class_name] = acts_BTP

    return all_classes_acts_BTP


def get_dataset_activations(
    dataset_name: str,
    config: SparseProbingEvalConfig,
    model: StandardizedTransformer,
    llm_batch_size: int,
    top_k: int,
    device: str,
) -> tuple[dict[str, th.Tensor], dict[str, th.Tensor]]:
    train_data, test_data = get_multi_label_train_test_data(
        dataset_name,
        config.probe_train_set_size,
        config.probe_test_set_size,
        config.random_seed,
    )

    chosen_classes = chosen_classes_per_dataset[dataset_name]

    train_data = filter_dataset(train_data, chosen_classes)
    test_data = filter_dataset(test_data, chosen_classes)

    train_data = tokenize_data_dictionary(
        train_data,
        cast("AutoTokenizer", model.tokenizer),
        config.context_length,
        device,
    )
    test_data = tokenize_data_dictionary(
        test_data,
        cast("AutoTokenizer", model.tokenizer),
        config.context_length,
        device,
    )

    all_train_acts_BTP = get_all_llm_activations(
        train_data,
        model=model,
        batch_size=llm_batch_size,
        top_k=top_k,
        mask_bos_pad_eos_tokens=True,
    )
    all_test_acts_BTP = get_all_llm_activations(
        test_data,
        model=model,
        batch_size=llm_batch_size,
        top_k=top_k,
        mask_bos_pad_eos_tokens=True,
    )

    return all_train_acts_BTP, all_test_acts_BTP


def get_paths_meaned_activations(
    all_llm_activations_BTP: dict[str, th.Tensor],
    paths: th.Tensor,  # (F, L * E)
    batch_size: int,
) -> dict[str, th.Tensor]:  # (B, F)
    """Encode LLM activations with an SAE and mean across the sequence length dimension for each class while ignoring padding tokens.
    VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    dtype = paths.dtype

    all_sae_activations_BF = {}
    for class_name, all_acts_BTP in all_llm_activations_BTP.items():
        all_acts_BF = []

        for _batch_idx, acts_BTP in enumerate(batched(all_acts_BTP, batch_size)):
            acts_BTF = acts_BTP @ paths.T

            acts_BT = th.sum(acts_BTF, dim=-2)
            nonzero_acts_BT = (acts_BT != 0.0).to(dtype=dtype)
            nonzero_acts_B = th.sum(nonzero_acts_BT, dim=-1)

            # TODO: check if *= can be used here or if it has to be acts_BTF = acts_BTF * nonzero_acts_BT[:, :, None]
            acts_BTF *= nonzero_acts_BT[:, :, None]
            acts_BF = th.sum(acts_BTF, dim=-2) / nonzero_acts_B[:, None]
            acts_BF = acts_BF.to(dtype=dtype)

            all_acts_BF.append(acts_BF)

        all_acts_BF = th.cat(all_acts_BF, dim=0)
        all_sae_activations_BF[class_name] = all_acts_BF

    return all_sae_activations_BF


def run_eval_single_dataset(
    dataset_name: str,
    config: SparseProbingEvalConfig,
    paths: Paths,
    model: StandardizedTransformer,
    device: str,
    artifacts_folder: str,
    save_activations: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility.
    """

    per_class_results_dict = {}

    activations_filename = f"{dataset_name}_activations.pt".replace("/", "_")

    activations_path = os.path.join(artifacts_folder, activations_filename)

    if not os.path.exists(activations_path):
        if config.lower_vram_usage:
            model = model.to(th.device(device))

        # Use default batch size of 32 if not specified
        batch_size = config.llm_batch_size if config.llm_batch_size is not None else 32
        all_train_acts_BTP, all_test_acts_BTP = get_dataset_activations(
            dataset_name,
            config,
            model,
            batch_size,
            paths.top_k,
            device,
        )
        if config.lower_vram_usage:
            model = model.to(th.device("cpu"))

        all_train_acts_BP = create_meaned_model_activations(all_train_acts_BTP)
        all_test_acts_BP = create_meaned_model_activations(all_test_acts_BTP)

        # We use GPU here as sklearn.fit is slow on large input dimensions, all other probe training is done with sklearn.fit
        _llm_probes, llm_test_accuracies = train_probe_on_activations(
            all_train_acts_BP,
            all_test_acts_BP,
            select_top_k=None,
            use_sklearn=False,
            batch_size=250,
            epochs=100,
            lr=1e-2,
        )

        llm_results = {"llm_test_accuracy": llm_test_accuracies}

        for k in config.k_values:
            _llm_top_k_probes, llm_top_k_test_accuracies = train_probe_on_activations(
                all_train_acts_BP,
                all_test_acts_BP,
                select_top_k=k,
            )
            llm_results[f"llm_top_{k}_test_accuracy"] = llm_top_k_test_accuracies

        acts = {
            "train": all_train_acts_BTP,
            "test": all_test_acts_BTP,
            "llm_results": llm_results,
        }

        if save_activations:
            th.save(acts, activations_path)
    else:
        if config.lower_vram_usage:
            model = model.to(th.device("cpu"))
        print(f"Loading activations from {activations_path}")
        acts = th.load(activations_path)
        all_train_acts_BTP = acts["train"]
        all_test_acts_BTP = acts["test"]
        llm_results = acts["llm_results"]

    all_sae_train_acts_BF = get_paths_meaned_activations(
        all_train_acts_BTP, paths.data, config.sae_batch_size
    )
    all_sae_test_acts_BF = get_paths_meaned_activations(
        all_test_acts_BTP, paths.data, config.sae_batch_size
    )

    for key in list(all_train_acts_BTP.keys()):
        del all_train_acts_BTP[key]
        del all_test_acts_BTP[key]

    if not config.lower_vram_usage:
        # This is optional, checking the accuracy of a probe trained on the entire SAE activations
        # We use GPU here as sklearn.fit is slow on large input dimensions, all other probe training is done with sklearn.fit
        _, sae_test_accuracies = train_probe_on_activations(
            all_sae_train_acts_BF,
            all_sae_test_acts_BF,
            select_top_k=None,
            use_sklearn=False,
            batch_size=250,
            epochs=100,
            lr=1e-2,
        )
        per_class_results_dict["sae_test_accuracy"] = sae_test_accuracies
    else:
        per_class_results_dict["sae_test_accuracy"] = {"-1": -1}

        for key in all_sae_train_acts_BF:
            all_sae_train_acts_BF[key] = all_sae_train_acts_BF[key].cpu()
            all_sae_test_acts_BF[key] = all_sae_test_acts_BF[key].cpu()

        th.cuda.empty_cache()
        gc.collect()

    per_class_results_dict.update(llm_results)

    for k in config.k_values:
        _sae_top_k_probes, sae_top_k_test_accuracies = train_probe_on_activations(
            all_sae_train_acts_BF,
            all_sae_test_acts_BF,
            select_top_k=k,
        )
        per_class_results_dict[f"sae_top_{k}_test_accuracy"] = sae_top_k_test_accuracies

    results_dict = {}
    for key, test_accuracies_dict in per_class_results_dict.items():
        average_test_acc = sum(test_accuracies_dict.values()) / len(
            test_accuracies_dict
        )
        results_dict[key] = average_test_acc

    return results_dict, per_class_results_dict


def run_eval_paths(
    config: SparseProbingEvalConfig,
    paths: Paths,
    model: StandardizedTransformer,
    device: str,
    artifacts_folder: str,
    save_activations: bool = True,
) -> tuple[dict[str, float | dict[str, float]], dict[str, dict[str, float]]]:
    """
    By default, we save activations for all datasets, and then reuse them for each set of paths.
    This is important to avoid recomputing activations for each set of paths, and to ensure that the same activations are used for all sets of paths.
    However, it can use 10s of GBs of disk space.
    """

    random.seed(config.random_seed)
    th.manual_seed(config.random_seed)
    os.makedirs(artifacts_folder, exist_ok=True)

    results_dict = {}

    dataset_results = {}
    per_class_dict = {}
    for dataset_name in config.dataset_names:
        results_key = f"{dataset_name}_results"
        (
            dataset_results[results_key],
            per_class_dict[results_key],
        ) = run_eval_single_dataset(
            dataset_name,
            config,
            paths,
            model,
            device,
            artifacts_folder,
            save_activations,
        )

    results_dict = general_utils.average_results_dictionaries(
        dataset_results, config.dataset_names
    )

    for dataset_name, dataset_result in dataset_results.items():
        results_dict[dataset_name] = dataset_result

    if config.lower_vram_usage:
        model = model.to(th.device(device))

    return results_dict, per_class_dict


def run_eval(
    config: SparseProbingEvalConfig,
    selected_paths_set: list[Paths],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
    artifacts_path: str = "artifacts",
    log_level: str = "INFO",
) -> dict[str, Any]:
    """
    If clean_up_activations is True, which means that the activations are deleted after the evaluation is done.
    You may want to use this because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of SAE name: evaluation results for that SAE.
    """
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    os.makedirs(output_path, exist_ok=True)

    artifacts_folder = None
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

        artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_SPARSE_PROBING)

        sparse_probing_results, per_class_dict = run_eval_paths(
            config,
            paths_with_metadata,
            model,
            device,
            artifacts_folder,
            save_activations=save_activations,
        )

        eval_output = SparseProbingEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=SparseProbingMetricCategories(
                llm=SparseProbingLlmMetrics(
                    **{
                        k: v
                        for k, v in sparse_probing_results.items()
                        if k.startswith("llm_") and not isinstance(v, dict)
                    }
                ),
                sae=SparseProbingSaeMetrics(
                    **{
                        k: v
                        for k, v in sparse_probing_results.items()
                        if k.startswith("sae_") and not isinstance(v, dict)
                    }
                ),
            ),
            eval_result_details=[
                SparseProbingResultDetail(
                    dataset_name=dataset_name,
                    **result,
                )
                for dataset_name, result in sparse_probing_results.items()
                if isinstance(result, dict)
            ],
            eval_result_unstructured=per_class_dict,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id="paths",
            sae_lens_release_id=paths_with_metadata.name,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=paths_with_metadata.metadata,
        )

        results_dict[paths_with_metadata.name] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        th.cuda.empty_cache()

    if (
        clean_up_activations
        and artifacts_folder is not None
        and os.path.exists(artifacts_folder)
    ):
        shutil.rmtree(artifacts_folder)

    return results_dict
