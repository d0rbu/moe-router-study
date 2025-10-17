import gc

from core.device import DeviceType, get_backend


def clear_memory(device_type: DeviceType = "cuda") -> None:
    """Clear memory for the specified device type.

    Args:
        device_type: Device type ("cuda" or "xpu", defaults to "cuda")
    """
    backend = get_backend(device_type)
    if backend.is_available():
        backend.empty_cache()
    gc.collect()
