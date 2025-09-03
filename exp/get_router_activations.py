"""Module for router activations experiment configuration."""

import os
import warnings

import yaml

# Constants
ROUTER_LOGITS_DIRNAME = "router_logits"
CONFIG_FILENAME = "config.yaml"

# Keys to verify in config
CONFIG_KEYS_TO_VERIFY = ["model_name", "dataset_name", "tokens_per_file", "batch_size"]


def get_experiment_name(model_name: str, dataset_name: str, **kwargs) -> str:
    """Get experiment name from model name, dataset name, and kwargs.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        **kwargs: Additional parameters to include in the experiment name

    Returns:
        Experiment name
    """
    # Start with base name
    experiment_name = f"{model_name}_{dataset_name}"

    # Filter out keys that should not be part of the experiment name
    filtered_keys = []
    for key in list(kwargs.keys()):
        if key == "device" or key == "resume" or key.startswith("_"):
            filtered_keys.append(key)
            kwargs.pop(key)

    # Warn about filtered keys
    if filtered_keys:
        warnings.warn(f"The following keys were excluded from the experiment name: {filtered_keys}", stacklevel=2)

    # Add remaining kwargs to experiment name
    if kwargs:
        # Sort keys for deterministic experiment names
        sorted_kwargs = sorted(kwargs.items())
        experiment_name += "_" + "_".join(f"{k}={v}" for k, v in sorted_kwargs)

    return experiment_name


def save_config(config: dict, experiment_dir: str) -> None:
    """Save config to experiment directory.

    Args:
        config: Configuration dictionary
        experiment_dir: Directory to save config to
    """
    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config to file
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def verify_config(config: dict, experiment_dir: str) -> None:
    """Verify that config matches saved config.

    Args:
        config: Configuration dictionary
        experiment_dir: Directory with saved config

    Raises:
        ValueError: If config does not match saved config
    """
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return

    # Load saved config
    with open(config_path) as f:
        saved_config = yaml.safe_load(f)

    # Check for mismatches in important keys
    mismatches = [
        f"{key}: {saved_config[key]} (saved) != {config[key]} (current)"
        for key in CONFIG_KEYS_TO_VERIFY
        if key in config and key in saved_config and config[key] != saved_config[key]
    ]

    # Raise error if mismatches found
    if mismatches:
        raise ValueError(
            "Config mismatch with saved experiment:\n" + "\n".join(mismatches)
        )
