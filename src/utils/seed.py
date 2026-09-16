"""Reproducibility and Deterministic Seed Management.

Controls random seed initialization across Python, NumPy, PyTorch (CPU and CUDA),
cuDNN deterministic flags, and DataLoader multi-process workers.
"""

import os
import random
from typing import Any, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int, deterministic_cudnn: bool = True) -> int:
    """Set random seed across all libraries to ensure full experimental reproducibility.

    Args:
        seed: The integer seed to set.
        deterministic_cudnn: Whether to enforce deterministic cuDNN operations.

    Returns:
        The seed that was set.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Enable deterministic algorithms where supported
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader.

    Ensures that each worker has a unique, deterministic random seed based on
    the parent process initial seed.

    Args:
        worker_id: DataLoader worker index.
    """
    if HAS_TORCH:
        base_seed = torch.initial_seed() % 2**32
    else:
        base_seed = np.random.get_state()[1][0]
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def get_generator(seed: int) -> Optional[Any]:
    """Return a seeded PyTorch Generator if torch is available."""
    if HAS_TORCH:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen
    return None
