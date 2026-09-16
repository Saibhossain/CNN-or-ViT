"""Compatibility wrapper delegating to src.models.cnn_family.efficientnet."""

from src.models.cnn_family.efficientnet import (
    EfficientNetModel,
    build_efficientnet_b0,
    build_efficientnetv2_s,
)

__all__ = [
    "EfficientNetModel",
    "build_efficientnetv2_s",
    "build_efficientnet_b0",
]
