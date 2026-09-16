"""Core utility package for CNN vs. ViT benchmarking framework."""

from src.utils.config import (
    deep_merge,
    load_config,
    load_yaml,
    validate_all_configs,
    validate_dataset_entry,
)
from src.utils.device import get_device, get_device_info, is_cuda_available, synchronize_device
from src.utils.logging import get_logger, log_step, setup_logging
from src.utils.paths import (
    ensure_dir,
    get_configs_dir,
    get_data_dir,
    get_logs_dir,
    get_project_root,
    get_results_dir,
)
from src.utils.seed import get_generator, set_seed, worker_init_fn
from src.utils.system import collect_system_fingerprint, get_system_metrics
from src.utils.validation import (
    validate_fraction,
    validate_model_output_shape,
    validate_seed,
)

__all__ = [
    "load_yaml",
    "deep_merge",
    "load_config",
    "validate_dataset_entry",
    "validate_all_configs",
    "set_seed",
    "get_generator",
    "worker_init_fn",
    "setup_logging",
    "get_logger",
    "log_step",
    "get_system_metrics",
    "collect_system_fingerprint",
    "get_device",
    "get_device_info",
    "is_cuda_available",
    "synchronize_device",
    "get_project_root",
    "get_data_dir",
    "get_results_dir",
    "get_configs_dir",
    "get_logs_dir",
    "ensure_dir",
    "validate_fraction",
    "validate_seed",
    "validate_model_output_shape",
]
