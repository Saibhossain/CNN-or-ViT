"""Unified Model Factory and Architecture Registry for CNN and Vision Transformer Families."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.cnn_family import (
    CNN_REGISTRY,
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
from src.models.common import count_parameters, estimate_model_flops
from src.models.vit_family import (
    SwinTransformerModel,
    VIT_REGISTRY,
    VisionTransformerModel,
    build_deit_b16,
    build_deit_s16,
    build_swin_s,
    build_swin_t,
    build_vit_b16,
    build_vit_s16,
    build_vit_ti16,
)

# Combined canonical model registry
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    **CNN_REGISTRY,
    **VIT_REGISTRY,
}

# Family categorization mapping
MODEL_FAMILIES: Dict[str, str] = {
    # CNNs
    "resnet18": "cnn",
    "resnet-18": "cnn",
    "resnet_18": "cnn",
    "rasnet18": "cnn",
    "resnet50": "cnn",
    "resnet-50": "cnn",
    "resnet_50": "cnn",
    "rasnet50": "cnn",
    "densenet121": "cnn",
    "densenet-121": "cnn",
    "densenet_121": "cnn",
    "efficientnetv2_s": "cnn",
    "efficientnetv2-s": "cnn",
    "efficientnet_v2_s": "cnn",
    "efficientnetv2": "cnn",
    "efficientnet_b0": "cnn",
    "efficientnet-b0": "cnn",
    "efficientnetb0": "cnn",
    "convnext_tiny": "cnn",
    "convnext-tiny": "cnn",
    "convnext_t": "cnn",
    "convnext": "cnn",

    # Vision Transformers
    "vit_b16": "vit",
    "vit-b/16": "vit",
    "vit_b_16": "vit",
    "vit_base_patch16_224": "vit",
    "vit_base": "vit",
    "vit_s16": "vit",
    "vit-s/16": "vit",
    "vit_s_16": "vit",
    "vit_small_patch16_224": "vit",
    "vit_small": "vit",
    "vit_ti16": "vit",
    "vit-ti/16": "vit",
    "vit_ti_16": "vit",
    "vit_tiny_patch16_224": "vit",
    "vit_tiny": "vit",
    "swin_t": "vit",
    "swin-t": "vit",
    "swin_tiny": "vit",
    "swin_tiny_patch4_window7_224": "vit",
    "swin_s": "vit",
    "swin-s": "vit",
    "swin_small": "vit",
    "swin_small_patch4_window7_224": "vit",
    "deit_s16": "vit",
    "deit-s/16": "vit",
    "deit_small": "vit",
    "deit_small_patch16_224": "vit",
    "deit_b16": "vit",
    "deit-b/16": "vit",
    "deit_base": "vit",
    "deit_base_patch16_224": "vit",
}


def normalize_model_name(model_name: str) -> str:
    """Normalize input model name string into canonical identifier."""
    clean = model_name.lower().strip()
    return clean


def list_models(family: Optional[str] = None) -> List[str]:
    """List available model architectures.

    Args:
        family: Optional filter ('cnn', 'vit', or None for all).

    Returns:
        List of canonical model architecture names.
    """
    if family is None:
        # Return unique canonical names
        unique_canonical = [
            "resnet50", "densenet121", "efficientnetv2_s", "convnext_tiny",
            "resnet18", "efficientnet_b0",
            "vit_b16", "swin_t", "vit_s16", "vit_ti16", "swin_s", "deit_s16", "deit_b16",
        ]
        return unique_canonical
    fam = family.lower()
    return [name for name, f in MODEL_FAMILIES.items() if f == fam]


def get_model_family(model_name: str) -> str:
    """Return architecture family ('cnn' or 'vit') for given model name."""
    clean = normalize_model_name(model_name)
    if clean in MODEL_FAMILIES:
        return MODEL_FAMILIES[clean]
    clean_alt = clean.replace("-", "_").replace("/", "_")
    return MODEL_FAMILIES.get(clean_alt, "cnn")


def create_model(
    model_name: str,
    num_classes: int = 100,
    pretrained: bool = False,
    image_size: int = 224,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    """Unified Factory function to instantiate any CNN or Vision Transformer model.

    Args:
        model_name: Name of architecture (e.g. 'resnet50', 'densenet121', 'vit_b16', 'swin_t').
        num_classes: Target classification classes count.
        pretrained: If True, load ImageNet-1K pretrained weights.
        image_size: Input spatial resolution (default 224).
        in_channels: Input image channel count (default 3 for RGB).
        drop_rate: Dropout rate before classification head.

    Returns:
        Instantiated PyTorch nn.Module with output shape [batch_size, num_classes].
    """
    clean_name = normalize_model_name(model_name)
    if clean_name not in MODEL_REGISTRY:
        # Try fallback with underscores
        clean_alt = clean_name.replace("-", "_").replace("/", "_")
        if clean_alt in MODEL_REGISTRY:
            clean_name = clean_alt
        else:
            raise ValueError(
                f"Unknown model architecture '{model_name}'. Available: {list_models()}"
            )

    builder = MODEL_REGISTRY[clean_name]
    model = builder(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )
    return model


def calculate_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
) -> Dict[str, Any]:
    """Calculate parameter counts, estimated FLOPs, and model memory footprint.

    Args:
        model: PyTorch model.
        input_size: Input tensor dimensions (batch_size, channels, height, width).
        device: Device to profile on ('cpu' or 'cuda').

    Returns:
        Dictionary with complexity metrics.
    """
    params_dict = count_parameters(model)
    flops_g = estimate_model_flops(model, input_size=input_size, device=device)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return {
        "total_params": params_dict["total_params"],
        "trainable_params": params_dict["trainable_params"],
        "non_trainable_params": params_dict["non_trainable_params"],
        "param_size_mb": round(param_size_mb, 2),
        "flops_g": flops_g,
        "input_size": list(input_size),
    }
