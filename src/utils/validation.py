"""Input validation and schema verification helpers."""

from typing import Any, Dict, List, Optional, Sequence, Union
import torch


def validate_fraction(fraction: Union[int, float]) -> int:
    """Validate and normalize data fraction (must be in 5, 10, 25, 50, 75, 100)."""
    valid_fractions = {5, 10, 25, 50, 75, 100}
    val = int(fraction)
    if val not in valid_fractions:
        raise ValueError(
            f"Invalid data fraction: {fraction}%. Allowed fractions are: {sorted(valid_fractions)}"
        )
    return val


def validate_seed(seed: int) -> int:
    """Validate random seed."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")
    return seed


def validate_model_output_shape(
    logits: torch.Tensor,
    batch_size: int,
    num_classes: int,
) -> None:
    """Ensure model output tensor matches [batch_size, num_classes]."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected model output to have 2 dimensions [B, C], got shape {logits.shape}"
        )
    if logits.shape[0] != batch_size:
        raise ValueError(
            f"Expected batch size {batch_size}, got {logits.shape[0]}"
        )
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} output logits, got {logits.shape[1]}"
        )
