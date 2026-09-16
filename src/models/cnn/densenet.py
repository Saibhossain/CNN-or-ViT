"""Compatibility wrapper delegating to src.models.cnn_family.densenet."""

from src.models.cnn_family.densenet import DenseNetModel, build_densenet121

__all__ = ["DenseNetModel", "build_densenet121"]
