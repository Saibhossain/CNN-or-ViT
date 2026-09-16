"""Compatibility wrapper delegating to src.models.cnn_family.convnext."""

from src.models.cnn_family.convnext import ConvNeXtModel, build_convnext_tiny

__all__ = ["ConvNeXtModel", "build_convnext_tiny"]
