"""
Sparse probing evaluation for raw model activations.

This script evaluates raw model activations from specified layers and activation types
by collecting activations directly from layers and running sparse probing evaluation on them.
Based on exp/sparse_probing_saebench.py but adapted for raw activations instead of router paths.
"""

from dataclasses import asdict
from datetime import datetime
import gc
import os
import random
import shutil
import sys
from typing import Any

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
from sae_bench.sae_bench_utils.general_utils import (
    average_results_dictionaries,
    str_to_dtype,
)
import torch as th
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from core.memory import clear_memory
from core.model import get_model_config
from core.type import assert_type
from exp import MODEL_DIRNAME
from exp.autointerp_saebench import Paths
from exp.get_activations import ActivationKeys


@th.no_grad()
def get_raw_activations(
    tokens: th.Tensor,  # (B, T)
    model: StandardizedTransformer,
    activation_key: ActivationKeys,
    layers: list[int],
    batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
    show_progress: bool = True,
) -> th.Tensor:  # (B, T, F)
    """Collects raw activations from specified layers for a given set of tokens.
    
    VERY IMPORTANT NOTE: If mask_bos_pad_eos_tokens is True, we zero out activations 
    for BOS, PAD, and EOS tokens. Later, we ignore zeroed activations.
    """
    logger.trace(f"Collecting raw activations for model: {model}")
    logger.trace(f"Activation key: {activation_key}")
    logger.trace(f"Layers: {layers}")
    logger.trace(f"Batch size: {batch_size}")
    logger.trace(f"Mask bos pad eos tokens: {mask_bos_pad_eos_tokens}")
    logger.trace(f"Show progress: {show_progress}")
    logger.trace(f"Tokens shape: {tokens.shape}")
    logger.trace(f"Tokens dtype: {tokens.dtype}")
    logger.trace(f"Tokens device: {tokens.device}")

    all_acts_BTF = []
    layers_sorted = sorted(layers)

    for batch_idx, tokens_BT in tqdm(
        enumerate(th.split(tokens, batch_size, dim=0)),
        total=tokens.shape[0] // batch_size,
        desc="Collecting activations",
        disable=not show_progress,
    ):
        layer_acts_list: list[th.Tensor] = []
        with model.trace(tokens_BT):
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

                layer_acts_list.append(activation)

        # Concatenate across layers: (B, T, sum(hidden_sizes))
        acts_BTF = th.cat(layer_acts_list, dim=-1)

        if mask_bos_pad_eos_tokens:
            attn_mask_BT = get_bos_pad_eos_mask(tokens_BT, model.tokenizer)
            acts_BTF *= attn_mask_BT[:, :, None]

        all_acts_BTF.append(acts_BTF)

    return th.cat(all_acts_BTF, dim=0)


@th.no_grad()
def get_all_raw_activations(
    tokenized_inputs_dict: dict[
        str, dict[str, th.Tensor]  # (B, T)
    ],
    model: StandardizedTransformer,
    activation_key: ActivationKeys,
    layers: list[int],
    batch_size: int,
    mask_bos_pad_eos_tokens: bool = False,
) -> dict[str, th.Tensor]:  # (B, T, F)
    """If we have a dictionary of tokenized inputs for different classes, 
    this function collects activations for all classes.
    
    We assume that the tokenized inputs have both the input_ids and attention_mask keys.
    VERY IMPORTANT NOTE: We zero out masked token activations in this function. 
    Later, we ignore zeroed activations.
    """
    all_classes_acts_BTF = {}

    for class_name in tokenized_inputs_dict:
        tokens = tokenized_inputs_dict[class_name]["input_ids"]

        acts_BTF = get_raw_activations(
            tokens,
            model=model,
            activation_key=activation_key,
            layers=layers,
            batch_size=batch_size,
            mask_bos_pad_eos_tokens=mask_bos_pad_eos_tokens,
        )

        all_classes_acts_BTF[class_name] = acts_BTF

    return all_classes_acts_BTF


def get_dataset_activations(
    dataset_name: str,
    config: SparseProbingEvalConfig,
    model: StandardizedTransformer,
    activation_key: ActivationKeys,
    layers: list[int],
    llm_batch_size: int,
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

    tokenizer = assert_type(
        model.tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast
    )
    train_data = tokenize_data_dictionary(
        train_data,
        tokenizer,
        config.context_length,
        device,
    )
    test_data = tokenize_data_dictionary(
        test_data,
        tokenizer,
        config.context_length,
        device,
    )

    all_train_acts_BTF = get_all_raw_activations(
        train_data,
        model=model,
        activation_key=activation_key,
        layers=layers,
        batch_size=llm_batch_size,
        mask_bos_pad_eos_tokens=True,
    )
    all_test_acts_BTF = get_all_raw_activations(
        test_data,
        model=model,
        activation_key=activation_key,
        layers=layers,
        batch_size=llm_batch_size,
        mask_bos_pad_eos_tokens=True,
    )

    return all_train_acts_BTF, all_test_acts_BTF


DatasetResults = dict[str, int | float]

LLM_DEFAULT_BATCH_SIZE = 32


def run_eval_single_dataset(
    dataset_name: str,
    config: SparseProbingEvalConfig,
    activation_key: ActivationKeys,
    layers: list[int],
    model: StandardizedTransformer,
    device: str,
    artifacts_folder: str,
    save_activations: bool,
) -> tuple[DatasetResults, dict[str, DatasetResults]]:
    """
    config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility.
    """

    per_class_results_dict = {}

    layers_str = "_".join(map(str, sorted(layers)))
    activations_filename = f"{dataset_name}_{activation_key}_layers_{layers_str}_activations.pt".replace("/", "_")

    activations_path = os.path.join(artifacts_folder, activations_filename)

    if not os.path.exists(activations_path):
        if config.lower_vram_usage:
            model = model.to(th.device(device))

        batch_size = config.llm_batch_size or LLM_DEFAULT_BATCH_SIZE
        all_train_acts_BTF, all_test_acts_BTF = get_dataset_activations(
            dataset_name,
            config,
            model,
            activation_key,
            layers,
            batch_size,
            device,
        )
        if config.lower_vram_usage:
            model = model.to(th.device("cpu"))

        all_train_acts_BF = create_meaned_model_activations(all_train_acts_BTF)
        all_test_acts_BF = create_meaned_model_activations(all_test_acts_BTF)

        # Train probes on raw activations (no SAE encoding needed)
        # We use GPU here as sklearn.fit is slow on large input dimensions
        _probes, test_accuracies = train_probe_on_activations(
            all_train_acts_BF,
            all_test_acts_BF,
            select_top_k=None,
            use_sklearn=False,
            batch_size=250,
            epochs=100,
            lr=1e-2,
        )

        results = {"test_accuracy": test_accuracies}

        for k in config.k_values:
            _top_k_probes, top_k_test_accuracies = train_probe_on_activations(
                all_train_acts_BF,
                all_test_acts_BF,
                select_top_k=k,
            )
            results[f"top_{k}_test_accuracy"] = top_k_test_accuracies

        acts = {
            "train": all_train_acts_BTF,
            "test": all_test_acts_BTF,
            "results": results,
        }

        if save_activations:
            th.save(acts, activations_path)
    else:
        if config.lower_vram_usage:
            model = model.to(th.device("cpu"))
        print(f"Loading activations from {activations_path}")
        acts = th.load(activations_path)
        all_train_acts_BTF = acts["train"]
        all_test_acts_BTF = acts["test"]
        results = acts["results"]

    # Clean up
    for key in list(all_train_acts_BTF.keys()):
        del all_train_acts_BTF[key]
        del all_test_acts_BTF[key]

    if config.lower_vram_usage:
        clear_memory()
        gc.collect()

    per_class_results_dict.update(results)

    results_dict = {}
    for key, test_accuracies_dict in per_class_results_dict.items():
        average_test_acc = sum(test_accuracies_dict.values()) / len(
            test_accuracies_dict
        )
        results_dict[key] = average_test_acc

    return results_dict, per_class_results_dict


def run_eval_raw_activations(
    config: SparseProbingEvalConfig,
    activation_key: ActivationKeys,
    layers: list[int],
    model: StandardizedTransformer,
    device: str,
    artifacts_folder: str,
    save_activations: bool = True,
) -> tuple[dict[str, int | float | DatasetResults], dict[str, dict[str, DatasetResults]]]:
    """
    By default, we save activations for all datasets, and then reuse them for different evaluations.
    This is important to avoid recomputing activations, and to ensure that the same activations 
    are used consistently. However, it can use 10s of GBs of disk space.
    """

    random.seed(config.random_seed)
    th.manual_seed(config.random_seed)
    os.makedirs(artifacts_folder, exist_ok=True)

    results_dict: dict[str, int | float | DatasetResults] = {}

    dataset_results: dict[str, DatasetResults] = {}
    per_class_dict: dict[str, dict[str, DatasetResults]] = {}
    for dataset_name in config.dataset_names:
        results_key = f"{dataset_name}_results"
        (
            dataset_results[results_key],
            per_class_dict[results_key],
        ) = run_eval_single_dataset(
            dataset_name,
            config,
            activation_key,
            layers,
            model,
            device,
            artifacts_folder,
            save_activations,
        )

    averaged_results: DatasetResults = average_results_dictionaries(
        dataset_results, config.dataset_names
    )
    results_dict: dict[str, int | float | DatasetResults] = {}
    results_dict.update(averaged_results)

    for dataset_name, dataset_result in dataset_results.items():
        results_dict[dataset_name] = dataset_result

    if config.lower_vram_usage:
        model = model.to(th.device(device))

    return results_dict, per_class_dict


def run_eval(
    config: SparseProbingEvalConfig,
    activation_key: ActivationKeys,
    layers: list[int],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
    artifacts_path: str = "artifacts",
    log_level: str = "INFO",
) -> dict[str, Any]:
    """
    If clean_up_activations is True, the activations are deleted after the evaluation is done.
    You may want to use this because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of evaluation results.
    """
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    os.makedirs(output_path, exist_ok=True)

    artifacts_folder = None
    results_dict = {}

    logger.trace(f"Using config: {config}")

    llm_dtype = str_to_dtype(config.llm_dtype)

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
        check_renaming=False,
        device_map=device,
        torch_dtype=llm_dtype,
    )

    layers_str = "_".join(map(str, sorted(layers)))
    name = f"raw_{activation_key}_layers_{layers_str}"
    
    sae_result_path = os.path.join(
        output_path, f"{name}_eval_results.json"
    )

    if os.path.exists(sae_result_path) and not force_rerun:
        print(f"Skipping {name} as results already exist")
        return results_dict

    artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_SPARSE_PROBING)

    sparse_probing_results, per_class_dict = run_eval_raw_activations(
        config,
        activation_key,
        layers,
        model,
        device,
        artifacts_folder,
        save_activations=save_activations,
    )

    # Map results to appropriate metric categories
    # Since we're evaluating raw activations (not SAE), we treat them as LLM metrics
    eval_output = SparseProbingEvalOutput(
        eval_config=config,
        eval_id=eval_instance_id,
        datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
        eval_result_metrics=SparseProbingMetricCategories(
            llm=SparseProbingLlmMetrics(
                **{
                    k: v
                    for k, v in sparse_probing_results.items()
                    if not isinstance(v, dict)
                }
            ),
            sae=SparseProbingSaeMetrics(),  # Empty SAE metrics
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
        sae_lens_id="raw_activations",
        sae_lens_release_id=name,
        sae_lens_version=sae_lens_version,
        sae_cfg_dict={
            "activation_key": str(activation_key),
            "layers": sorted(layers),
        },
    )

    results_dict[name] = asdict(eval_output)

    eval_output.to_json_file(sae_result_path, indent=2)

    gc.collect()
    clear_memory()

    if (
        clean_up_activations
        and artifacts_folder is not None
        and os.path.exists(artifacts_folder)
    ):
        shutil.rmtree(artifacts_folder)

    return results_dict
