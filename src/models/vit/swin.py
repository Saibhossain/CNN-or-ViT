"""Compatibility wrapper delegating to src.models.vit_family.swin."""

from src.models.vit_family.swin import (
    SwinTransformerModel,
    build_swin_s,
    build_swin_t,
)

__all__ = ["SwinTransformerModel", "build_swin_t", "build_swin_s"]
