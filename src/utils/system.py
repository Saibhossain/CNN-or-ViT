"""System environment, hardware introspection, and reproducibility metadata."""

import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_git_info() -> Dict[str, Optional[str]]:
    """Retrieve current Git commit hash, branch, and dirty status.

    Returns:
        Dictionary containing git commit metadata.
    """
    git_info: Dict[str, Optional[str]] = {
        "commit_hash": None,
        "branch": None,
        "is_dirty": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_info["commit_hash"] = commit

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_info["branch"] = branch

        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_info["is_dirty"] = "true" if status else "false"
    except Exception:
        pass
    return git_info


def get_gpu_info() -> List[Dict[str, Any]]:
    """Retrieve CUDA GPU specifications and memory statistics.

    Returns:
        List of dictionaries with details for each available GPU.
    """
    gpus: List[Dict[str, Any]] = []
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_data = {
                "index": i,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024**2), 2),
                "major_capability": props.major,
                "minor_capability": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
            gpus.append(gpu_data)
    return gpus


def get_system_info() -> Dict[str, Any]:
    """Gather comprehensive system, software, and hardware metadata for reproducibility.

    Returns:
        Dictionary containing full system profiling metadata.
    """
    ram_gb = None
    cpu_count_logical = os.cpu_count()
    cpu_count_physical = None

    if HAS_PSUTIL:
        try:
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            cpu_count_physical = psutil.cpu_count(logical=False)
        except Exception:
            pass

    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform_architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": cpu_count_logical,
        "cpu_count_physical": cpu_count_physical,
        "ram_gb": ram_gb,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "git": get_git_info(),
        "pytorch": {
            "installed": HAS_TORCH,
            "version": torch.__version__ if HAS_TORCH else None,
            "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
            "cuda_version": torch.version.cuda if HAS_TORCH and torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if HAS_TORCH and torch.cuda.is_available() else None,
            "gpus": get_gpu_info(),
        },
    }
    return info


collect_system_fingerprint = get_system_info
get_system_metrics = get_system_info

