from hashlib import sha256
import warnings


def get_experiment_name(model_name: str, dataset_name: str, **kwargs) -> str:
    """Generate a unique experiment name based on configuration parameters."""
    experiment_name = f"{model_name}_{dataset_name}"

    # Track which keys are being filtered out
    ignored_keys = {"device", "resume"}
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
        experiment_name = f"{experiment_name}_{param_str}"

    if len(experiment_name) > 255:
        # this is too long for the filesystem, so we hash it
        # the config should be stored in the experiment directory anyway
        experiment_name = sha256(experiment_name.encode()).hexdigest()

    return experiment_name


def parse_experiment_name(experiment_name: str) -> dict[str, str | int]:
    """Parse an experiment name back into its components.
    
    Args:
        experiment_name: Experiment name in format "model_dataset_param1=value1_param2=value2..."
        
    Returns:
        Dictionary with parsed components including model_name, dataset_name, and other parameters
    """
    # Handle hashed experiment names
    if len(experiment_name) == 64 and all(c in '0123456789abcdef' for c in experiment_name):
        raise ValueError(
            f"Cannot parse hashed experiment name: {experiment_name}. "
            "The original experiment name was too long and was hashed."
        )
    
    parts = experiment_name.split('_')
    
    # Find the first part that contains '=' to separate model_dataset from parameters
    param_start_idx = None
    for i, part in enumerate(parts):
        if '=' in part:
            param_start_idx = i
            break
    
    if param_start_idx is None:
        # No parameters, just model_dataset
        if len(parts) < 2:
            raise ValueError(f"Invalid experiment name format: {experiment_name}")
        return {
            "model_name": parts[0],
            "dataset_name": "_".join(parts[1:])
        }
    
    # Split into model_dataset and parameters
    model_dataset_parts = parts[:param_start_idx]
    param_parts = parts[param_start_idx:]
    
    if len(model_dataset_parts) < 2:
        raise ValueError(f"Invalid experiment name format: {experiment_name}")
    
    result = {
        "model_name": model_dataset_parts[0],
        "dataset_name": "_".join(model_dataset_parts[1:])
    }
    
    # Parse parameters
    for param_part in param_parts:
        if '=' not in param_part:
            raise ValueError(f"Invalid parameter format in experiment name: {param_part}")
        
        key, value = param_part.split('=', 1)
        
        # Try to convert to int if possible
        try:
            result[key] = int(value)
        except ValueError:
            result[key] = value
    
    return result
