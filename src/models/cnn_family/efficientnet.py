"""EfficientNet Architecture Implementations (EfficientNetV2-S, EfficientNet-B0) for CNN vs. ViT Benchmark."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_b0 as tv_efficientnet_b0,
    efficientnet_v2_s as tv_efficientnet_v2_s,
)


class EfficientNetModel(nn.Module):
    """EfficientNet wrapper providing unified architecture metadata and feature extraction."""

    def __init__(
        self,
        arch: str = "efficientnetv2_s",
        num_classes: int = 100,
        pretrained: bool = False,
        in_channels: int = 3,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.arch = arch.lower().replace("-", "_")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = drop_rate

        if self.arch in ("efficientnetv2_s", "efficientnetv2", "efficientnet_v2_s"):
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tv_efficientnet_v2_s(weights=weights, dropout=drop_rate)
            in_features = self.backbone.classifier[1].in_features
        elif self.arch in ("efficientnet_b0", "efficientnetb0"):
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tv_efficientnet_b0(weights=weights, dropout=drop_rate)
            in_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported EfficientNet architecture: {arch}")

        # Adapt first conv if input channels != 3
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
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-stage Fused-MBConv spatial features."""
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


def build_efficientnetv2_s(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.2,
    **kwargs: Any,
) -> EfficientNetModel:
    return EfficientNetModel(
        arch="efficientnetv2_s",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
    )


def build_efficientnet_b0(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.2,
    **kwargs: Any,
) -> EfficientNetModel:
    return EfficientNetModel(
        arch="efficientnet_b0",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
    )
