"""CNN Family Architectures for CNN vs. ViT Benchmarking."""

from src.models.cnn_family.convnext import ConvNeXtModel, build_convnext_tiny
from src.models.cnn_family.densenet import DenseNetModel, build_densenet121
from src.models.cnn_family.efficientnet import (
    EfficientNetModel,
    build_efficientnet_b0,
    build_efficientnetv2_s,
)
from src.models.cnn_family.registry import CNN_REGISTRY
from src.models.cnn_family.resnet import (
    ResNetModel,
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
    "CNN_REGISTRY",
]
