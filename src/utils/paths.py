"""Filesystem path helpers for CNN vs. ViT research framework."""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Return absolute path to repository root."""
    # Assuming this file is in src/utils/paths.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def get_data_dir(custom_path: Union[str, Path, None] = None) -> Path:
    """Return path to datasets directory."""
    if custom_path is not None:
        p = Path(custom_path)
        return p if p.is_absolute() else get_project_root() / p
    return get_project_root() / "DATA"


def get_results_dir(custom_path: Union[str, Path, None] = None) -> Path:
    """Return path to experiment results directory."""
    if custom_path is not None:
        p = Path(custom_path)
        return p if p.is_absolute() else get_project_root() / p
    return get_project_root() / "result"


def get_configs_dir() -> Path:
    """Return path to configuration directory."""
    return get_project_root() / "configs"


def get_logs_dir() -> Path:
    """Return path to execution logs directory."""
    p = get_project_root() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return resolved Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()
