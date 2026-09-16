"""DenseNet Architecture Implementation (DenseNet-121) for CNN vs. ViT Benchmark."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121 as tv_densenet121,
)


class DenseNetModel(nn.Module):
    """DenseNet-121 wrapper providing unified architecture metadata and feature extraction."""

    def __init__(
        self,
        arch: str = "densenet121",
        num_classes: int = 100,
        pretrained: bool = False,
        in_channels: int = 3,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.arch = arch.lower().replace("-", "_")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = drop_rate

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_densenet121(weights=weights, drop_rate=drop_rate)
        in_features = self.backbone.classifier.in_features

        # Adapt first conv if input channels != 3
        if in_channels != 3:
            old_conv = self.backbone.features.conv0
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # Replace classification head
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract dense spatial feature representation."""
        features = self.backbone.features(x)
        out = nn.functional.relu(features, inplace=True)
        return out

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.arch,
            "family": "cnn",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def build_densenet121(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> DenseNetModel:
    return DenseNetModel(
        arch="densenet121",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
    )
