"""Compatibility wrapper delegating to src.models.cnn_family.resnet."""

from src.models.cnn_family.resnet import (
    ResNetModel,
    build_resnet18,
    build_resnet50,
)

__all__ = ["ResNetModel", "build_resnet18", "build_resnet50"]
