"""Compatibility wrapper delegating to src.models.vit_family.vit."""

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
    "build_vit_b16",
    "build_vit_s16",
    "build_vit_ti16",
    "build_deit_s16",
    "build_deit_b16",
]
