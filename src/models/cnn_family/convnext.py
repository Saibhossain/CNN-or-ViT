"""ConvNeXt Architecture Implementation (ConvNeXt-Tiny) for CNN vs. ViT Benchmark."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny as tv_convnext_tiny,
)


class ConvNeXtModel(nn.Module):
    """ConvNeXt-Tiny wrapper providing unified architecture metadata and feature extraction."""

    def __init__(
        self,
        arch: str = "convnext_tiny",
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

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features

        # Adapt first conv layer if input channels != 3
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # Replace classification head
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-stage 7x7 depthwise spatial features."""
        return self.backbone.features(x)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.arch,
            "family": "cnn",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def build_convnext_tiny(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> ConvNeXtModel:
    return ConvNeXtModel(
        arch="convnext_tiny",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
    )
