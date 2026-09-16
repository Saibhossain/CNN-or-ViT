"""ResNet Architecture Implementations (ResNet-18, ResNet-50) for CNN vs. ViT Benchmark."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18 as tv_resnet18,
    resnet50 as tv_resnet50,
)


class ResNetModel(nn.Module):
    """ResNet wrapper providing unified architecture metadata and feature extraction."""

    def __init__(
        self,
        arch: str = "resnet50",
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

        if self.arch in ("resnet18", "rasnet18"):
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tv_resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
        elif self.arch in ("resnet50", "rasnet50"):
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = tv_resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

        # Adapt first conv if input channels != 3
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # Custom classification head
        if drop_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map prior to global pooling."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.arch,
            "family": "cnn",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def build_resnet18(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> ResNetModel:
    return ResNetModel(
        arch="resnet18",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
    )


def build_resnet50(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> ResNetModel:
    return ResNetModel(
        arch="resnet50",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
    )
