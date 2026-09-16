"""Vision Transformer Family Architectures for CNN vs. ViT Benchmarking."""

from src.models.vit_family.registry import VIT_REGISTRY
from src.models.vit_family.swin import (
    SwinTransformerModel,
    build_swin_s,
    build_swin_t,
)
from src.models.vit_family.vit import (
    VisionTransformerModel,
    build_deit_b16,
    build_deit_s16,
    build_vit_b16,
    build_vit_s16,
    build_vit_ti16,
)

__all__ = [
    "VisionTransformerModel",
    "SwinTransformerModel",
    "build_vit_b16",
    "build_vit_s16",
    "build_vit_ti16",
    "build_swin_t",
    "build_swin_s",
    "build_deit_s16",
    "build_deit_b16",
    "VIT_REGISTRY",
]
