"""Experiments for MoE router study."""

import os
from typing import Any, TypeVar
import warnings

import yaml

# Base output directory
OUTPUT_DIR = "out"

# Constants for subdirectories
ROUTER_LOGITS_DIRNAME = "router_logits"
WEIGHT_DIRNAME = "weights"
CONFIG_FILENAME = "config.yaml"

# For backward compatibility
ROUTER_LOGITS_DIR = os.path.join(OUTPUT_DIR, ROUTER_LOGITS_DIRNAME)
WEIGHT_DIR = os.path.join(OUTPUT_DIR, WEIGHT_DIRNAME)

# Type definitions
T = TypeVar("T")


def get_experiment_dir(name: str) -> str:
    """Get the directory path for an experiment.

    Args:
        name: Name of the experiment

    Returns:
        Path to the experiment directory
    """
    experiment_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def get_experiment_name(model_name: str, dataset_name: str, **kwargs) -> str:
    """Generate a unique experiment name based on configuration parameters.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        **kwargs: Additional parameters to include in the name

    Returns:
        A unique experiment name
    """
    base_name = f"{model_name}_{dataset_name}"

    # Track which keys are being filtered out
    ignored_keys = {"device", "resume", "name"}
    filtered_keys = set()

    # Add any additional parameters that might affect the experiment
    param_items = []
    for k, v in sorted(kwargs.items()):
        if k in ignored_keys or k.startswith("_"):
            filtered_keys.add(k)
            continue
        param_items.append(f"{k}={v}")

    # Warn about filtered keys
    if filtered_keys:
        warnings.warn(
            f"The following keys were excluded from the experiment name: {filtered_keys}",
            stacklevel=2,
        )

    param_str = "_".join(param_items)

    if param_str:
        base_name = f"{base_name}_{param_str}"

    return base_name


def save_config(config: dict, experiment_dir: str) -> None:
    """Save experiment configuration to a YAML file.

    Args:
        config: Configuration dictionary
        experiment_dir: Directory to save the config file
    """
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(experiment_dir: str) -> dict | None:
    """Load experiment configuration from a YAML file.

    Args:
        experiment_dir: Directory containing the config file

    Returns:
        Configuration dictionary or None if file doesn't exist
    """
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        return yaml.safe_load(f)


def verify_config(
    config: dict, experiment_dir: str, keys_to_verify: set | None = None
) -> None:
    """Verify that the current configuration matches the saved one.

    Args:
        config: Current configuration dictionary
        experiment_dir: Directory containing the saved config file
        keys_to_verify: Set of keys to verify (if None, verify all keys)

    Raises:
        ValueError: If there are mismatches between the current and saved config
    """
    saved_config = load_config(experiment_dir)
    if saved_config is None:
        return

    # Check for mismatches
    mismatches = {}
    keys_to_check = keys_to_verify if keys_to_verify is not None else config.keys()

    for key in keys_to_check:
        current_value = config.get(key)
        saved_value = saved_config.get(key)
        if current_value != saved_value:
            mismatches[key] = (saved_value, current_value)

    if mismatches:
        mismatch_str = "\n".join(
            f"  - {key}: saved={saved} vs current={current}"
            for key, (saved, current) in mismatches.items()
        )
        raise ValueError(
            f"Configuration mismatch with existing experiment:\n{mismatch_str}"
        )
