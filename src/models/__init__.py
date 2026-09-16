"""Models package for CNN vs. ViT Benchmarking Suite."""

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
from src.models.common import count_parameters, estimate_model_flops, replace_classification_head
from src.models.factory import (
    calculate_model_complexity,
    create_model,
    get_model_family,
    list_models,
)
from src.models.vit_family import (
    SwinTransformerModel,
    VisionTransformerModel,
    build_deit_b16,
    build_deit_s16,
    build_swin_s,
    build_swin_t,
    build_vit_b16,
    build_vit_s16,
    build_vit_ti16,
)

__all__ = [
    "create_model",
    "list_models",
    "get_model_family",
    "count_parameters",
    "estimate_model_flops",
    "calculate_model_complexity",
    "replace_classification_head",
    "ResNetModel",
    "DenseNetModel",
    "EfficientNetModel",
    "ConvNeXtModel",
    "VisionTransformerModel",
    "SwinTransformerModel",
    "build_resnet18",
    "build_resnet50",
    "build_densenet121",
    "build_efficientnetv2_s",
    "build_efficientnet_b0",
    "build_convnext_tiny",
    "build_vit_b16",
    "build_vit_s16",
    "build_vit_ti16",
    "build_swin_t",
    "build_swin_s",
    "build_deit_s16",
    "build_deit_b16",
]
