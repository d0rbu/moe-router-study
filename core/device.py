"""Device abstraction layer for CUDA and Intel XPU support."""

from typing import Literal

import torch as th

# Type definition for supported device types
DeviceType = Literal["cuda", "xpu"]


def assert_device_type(device_type: str) -> DeviceType:
    """Assert that the device type is valid and return it as a DeviceType.
    
    Args:
        device_type: The device type string to validate
        
    Returns:
        The validated device type as a Literal["cuda", "xpu"]
        
    Raises:
        ValueError: If the device type is not "cuda" or "xpu"
    """
    if device_type not in ("cuda", "xpu"):
        raise ValueError(f"device_type must be 'cuda' or 'xpu', got '{device_type}'")
    return device_type  # type: ignore[return-value]


def is_available(device_type: DeviceType) -> bool:
    """Check if the specified device type is available.
    
    Args:
        device_type: The device type to check ("cuda" or "xpu")
        
    Returns:
        True if the device type is available, False otherwise
    """
    if device_type == "cuda":
        return th.cuda.is_available()
    elif device_type == "xpu":
        return th.xpu.is_available()
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")


def device_count(device_type: DeviceType) -> int:
    """Get the number of devices available for the specified device type.
    
    Args:
        device_type: The device type to check ("cuda" or "xpu")
        
    Returns:
        The number of available devices
    """
    if device_type == "cuda":
        return th.cuda.device_count()
    elif device_type == "xpu":
        return th.xpu.device_count()
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")


def empty_cache(device_type: DeviceType) -> None:
    """Release all unoccupied cached memory for the specified device type.
    
    Args:
        device_type: The device type to clear cache for ("cuda" or "xpu")
    """
    if device_type == "cuda":
        th.cuda.empty_cache()
    elif device_type == "xpu":
        th.xpu.empty_cache()
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")


def manual_seed(device_type: DeviceType, seed: int) -> None:
    """Set the random seed for the specified device type.
    
    Args:
        device_type: The device type to set seed for ("cuda" or "xpu")
        seed: The random seed value
    """
    if device_type == "cuda":
        th.cuda.manual_seed(seed)
    elif device_type == "xpu":
        th.xpu.manual_seed(seed)
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")


def manual_seed_all(device_type: DeviceType, seed: int) -> None:
    """Set the random seed for all devices of the specified device type.
    
    Args:
        device_type: The device type to set seed for ("cuda" or "xpu")
        seed: The random seed value
    """
    if device_type == "cuda":
        th.cuda.manual_seed_all(seed)
    elif device_type == "xpu":
        th.xpu.manual_seed_all(seed)
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")


def get_device(device_type: DeviceType, device_idx: int) -> th.device:
    """Create a torch device object for the specified device type and index.
    
    Args:
        device_type: The device type ("cuda" or "xpu")
        device_idx: The device index
        
    Returns:
        A torch.device object
    """
    return th.device(f"{device_type}:{device_idx}")

