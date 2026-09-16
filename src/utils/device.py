"""Device discovery and management utilities."""

from typing import Any, Dict, Optional, Union
import torch


def is_cuda_available() -> bool:
    """Check if CUDA GPU acceleration is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if Apple Silicon MPS acceleration is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(preferred: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve and return appropriate PyTorch device.

    Args:
        preferred: Optional preferred device string ('cuda', 'cuda:0', 'mps', 'cpu').

    Returns:
        torch.device instance.
    """
    if preferred is not None:
        if isinstance(preferred, torch.device):
            return preferred
        pref_str = str(preferred).lower().strip()
        if pref_str.startswith("cuda") and is_cuda_available():
            return torch.device(pref_str)
        if pref_str == "mps" and is_mps_available():
            return torch.device("mps")
        if pref_str == "cpu":
            return torch.device("cpu")

    # Auto-detection
    if is_cuda_available():
        return torch.device("cuda:0")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: Optional[Union[str, torch.device]] = None) -> None:
    """Synchronize device execution stream for accurate timing."""
    dev = get_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def get_device_info(device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """Retrieve detailed hardware information for the selected device."""
    dev = get_device(device)
    info: Dict[str, Any] = {
        "device_type": dev.type,
        "device_index": dev.index if dev.index is not None else 0,
        "torch_version": torch.__version__,
        "cuda_available": is_cuda_available(),
    }

    if dev.type == "cuda" and is_cuda_available():
        idx = dev.index if dev.index is not None else 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_capability"] = torch.cuda.get_device_capability(idx)
        info["gpu_memory_total_mb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024), 2)
    elif dev.type == "cpu":
        import platform
        info["cpu_arch"] = platform.machine()
        info["cpu_processor"] = platform.processor()

    return info
