from hashlib import sha256
import math
import warnings


def exponential_to_linear_save_steps(total_steps: int, save_every: int) -> set[int]:
    num_exponential_save_steps = math.ceil(math.log2(save_every))

    # exponential ramp and then linear
    # 0, 1, 2, 4, ..., save_every, save_every * 2, save_every * 3, ...
    save_steps = set(range(0, total_steps, save_every))
    save_steps += {2**i for i in range(num_exponential_save_steps)}
    save_steps.add(0)

    return save_steps


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
