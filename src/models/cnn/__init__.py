"""Compatibility wrapper for CNN architectures delegating to src.models.cnn_family."""

from src.models.cnn_family import (
    ConvNeXtModel,
    DenseNetModel,
    EfficientNetModel,
    ResNetModel,
    build_convnext_tiny,
    build_densenet121,
    build_efficientnet_b0,
    build_efficientnetv2_s,
    build_resnet18,
    build_resnet50,
)

__all__ = [
    "ResNetModel",
    "DenseNetModel",
    "EfficientNetModel",
    "ConvNeXtModel",
    "build_resnet18",
    "build_resnet50",
    "build_densenet121",
    "build_efficientnetv2_s",
    "build_efficientnet_b0",
    "build_convnext_tiny",
]
