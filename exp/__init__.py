"""Experiments for MoE router study."""
import os
from typing import Any, Dict, Optional
import warnings

import yaml

# Base directories
BASE_OUTPUT_DIR = "out"
ROUTER_LOGITS_DIRNAME = "router_logits"
WEIGHT_DIRNAME = "weights"
CONFIG_FILENAME = "config.yaml"


def get_experiment_dir(name: Optional[str] = None, **kwargs) -> str:
    """
    Get the experiment directory path.
    
    Args:
        name: Optional name for the experiment. If not provided, one will be generated
              based on the provided kwargs.
        **kwargs: Configuration parameters used to generate a name if one is not provided.
    
    Returns:
        Path to the experiment directory.
    """
    if name is None:
        # Generate a name based on config parameters
        if not kwargs:
            raise ValueError("Either name or config parameters must be provided")
        
        # Extract model_name and dataset_name if available
        model_name = kwargs.get("model_name", "unknown_model")
        dataset_name = kwargs.get("dataset_name", "unknown_dataset")
        
        # Track which keys are being filtered out
        ignored_keys = {"device", "resume", "model_name", "dataset_name"}
        filtered_keys = set()

        # Add any additional parameters that might affect the experiment
        param_items = []
        for k, v in sorted(kwargs.items()):
            if k in ignored_keys or k.startswith("_"):
                filtered_keys.add(k)
                continue
            param_items.append(f"{k}={v}")

        # Warn about filtered keys
        if filtered_keys and filtered_keys != ignored_keys:
            warnings.warn(
                f"The following keys were excluded from the experiment name: {filtered_keys}",
                stacklevel=2,
            )

        param_str = "_".join(param_items)
        name = f"{model_name}_{dataset_name}"
        
        if param_str:
            name = f"{name}_{param_str}"
    
    return os.path.join(BASE_OUTPUT_DIR, name)


def save_config(config: Dict[str, Any], experiment_dir: str) -> None:
    """
    Save experiment configuration to a YAML file.
    
    Args:
        config: Dictionary containing configuration parameters.
        experiment_dir: Path to the experiment directory.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def verify_config(config: Dict[str, Any], experiment_dir: str) -> None:
    """
    Verify that the current configuration matches the saved one.
    
    Args:
        config: Dictionary containing current configuration parameters.
        experiment_dir: Path to the experiment directory.
        
    Raises:
        ValueError: If there's a mismatch between current and saved configurations.
    """
    config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
    
    if not os.path.exists(config_path):
        return
    
    with open(config_path) as f:
        saved_config = yaml.safe_load(f)
    
    # Check for mismatches
    mismatches = {}
    for key, value in config.items():
        if key in saved_config and saved_config[key] != value:
            mismatches[key] = (saved_config[key], value)
    
    if mismatches:
        mismatch_str = "\n".join(
            f"  - {key}: saved={saved} vs current={current}"
            for key, (saved, current) in mismatches.items()
        )
        raise ValueError(
            f"Configuration mismatch with existing experiment:\n{mismatch_str}"
        )


def get_router_logits_dir(experiment_dir: str) -> str:
    """Get the router logits directory for an experiment."""
    return os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)


def get_weight_dir(experiment_dir: str) -> str:
    """Get the weights directory for an experiment."""
    return os.path.join(experiment_dir, WEIGHT_DIRNAME)

