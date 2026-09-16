"""CNN Family Registry mapping names and aliases to model builders."""

from typing import Any, Callable, Dict
import torch.nn as nn

from src.models.cnn_family.convnext import ConvNeXtModel, build_convnext_tiny
from src.models.cnn_family.densenet import DenseNetModel, build_densenet121
from src.models.cnn_family.efficientnet import (
    EfficientNetModel,
    build_efficientnet_b0,
    build_efficientnetv2_s,
)
from src.models.cnn_family.resnet import (
    ResNetModel,
    build_resnet18,
    build_resnet50,
)

CNN_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    # ResNet-18
    "resnet18": build_resnet18,
    "resnet-18": build_resnet18,
    "resnet_18": build_resnet18,
    "rasnet18": build_resnet18,
    "rasnet-18": build_resnet18,

    # ResNet-50
    "resnet50": build_resnet50,
    "resnet-50": build_resnet50,
    "resnet_50": build_resnet50,
    "rasnet50": build_resnet50,
    "rasnet-50": build_resnet50,

    # DenseNet-121
    "densenet121": build_densenet121,
    "densenet-121": build_densenet121,
    "densenet_121": build_densenet121,

    # EfficientNetV2-S
    "efficientnetv2_s": build_efficientnetv2_s,
    "efficientnetv2-s": build_efficientnetv2_s,
    "efficientnet_v2_s": build_efficientnetv2_s,
    "efficientnetv2": build_efficientnetv2_s,

    # EfficientNet-B0
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet-b0": build_efficientnet_b0,
    "efficientnetb0": build_efficientnet_b0,

    # ConvNeXt-Tiny
    "convnext_tiny": build_convnext_tiny,
    "convnext-tiny": build_convnext_tiny,
    "convnext_t": build_convnext_tiny,
    "convnext": build_convnext_tiny,
}
