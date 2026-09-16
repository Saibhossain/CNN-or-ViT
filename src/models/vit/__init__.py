"""Compatibility wrapper for Vision Transformers delegating to src.models.vit_family."""

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
    "VisionTransformerModel",
    "SwinTransformerModel",
    "build_vit_b16",
    "build_vit_s16",
    "build_vit_ti16",
    "build_swin_t",
    "build_swin_s",
    "build_deit_s16",
    "build_deit_b16",
]
