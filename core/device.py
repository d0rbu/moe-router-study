"""Device abstraction layer for CUDA and Intel XPU support."""

from typing import Any, Literal

import torch as th

# Type definition for supported device types
DeviceType = Literal["cuda", "xpu"]

# Track if IPEX has been imported
_ipex_imported = False


def _ensure_ipex_imported() -> None:
    """Ensure Intel Extension for PyTorch is imported to register XPU backend.

    This must be called before any XPU operations to register the XPU device type
    with PyTorch. This is a no-op if IPEX has already been imported.
    """
    global _ipex_imported
    if not _ipex_imported:
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401

            _ipex_imported = True
        except ImportError:
            # IPEX not available, XPU operations will fail later with clear error
            pass


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


def get_backend(device_type: DeviceType) -> Any:
    """Get the backend module for the specified device type.

    Args:
        device_type: The device type ("cuda" or "xpu")

    Returns:
        The appropriate backend module (th.cuda or th.xpu)

    Raises:
        ValueError: If the device type is not supported
    """
    if device_type == "cuda":
        return th.cuda
    elif device_type == "xpu":
        # Ensure IPEX is imported before accessing th.xpu
        _ensure_ipex_imported()
        return th.xpu
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")


def get_device(device_type: DeviceType, device_idx: int = 0) -> th.device:
    """Create a torch device object for the specified device type and index.

    Args:
        device_type: The device type ("cuda" or "xpu")
        device_idx: The device index (defaults to 0)

    Returns:
        A torch.device object
    """
    # Ensure IPEX is imported for XPU devices
    if device_type == "xpu":
        _ensure_ipex_imported()
    return th.device(f"{device_type}:{device_idx}")


def get_distributed_backend(device_type: DeviceType) -> str:
    """Get the distributed backend name for the specified device type.

    Args:
        device_type: The device type ("cuda" or "xpu")

    Returns:
        The appropriate distributed backend name ("nccl" for CUDA, "xccl" for XPU)
    """
    if device_type == "cuda":
        return "nccl"
    elif device_type == "xpu":
        return "xccl"
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")
