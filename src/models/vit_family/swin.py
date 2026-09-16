"""Swin Transformer Architecture Implementations (Swin-T, Swin-S) for CNN vs. ViT Benchmark."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import timm


class SwinTransformerModel(nn.Module):
    """Swin Transformer wrapper supporting Swin-T and Swin-S with shifted-window attention."""

    def __init__(
        self,
        variant: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 100,
        pretrained: bool = False,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.variant = variant.lower().replace("-", "_")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.drop_rate = drop_rate

        timm_name = "swin_tiny_patch4_window7_224"
        if any(k in self.variant for k in ["swin_s", "swin_small"]):
            timm_name = "swin_small_patch4_window7_224"
        elif any(k in self.variant for k in ["swin_t", "swin_tiny"]):
            timm_name = "swin_tiny_patch4_window7_224"

        try:
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                drop_rate=drop_rate,
                **kwargs,
            )
        except Exception:
            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                drop_rate=drop_rate,
                **kwargs,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-stage hierarchical shifted-window feature representations."""
        return self.backbone.forward_features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def get_classifier(self) -> nn.Module:
        return self.backbone.get_classifier()

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.backbone.reset_classifier(num_classes=num_classes)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.variant,
            "family": "vit",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def build_swin_t(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> SwinTransformerModel:
    return SwinTransformerModel(
        variant="swin_tiny_patch4_window7_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )


def build_swin_s(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> SwinTransformerModel:
    return SwinTransformerModel(
        variant="swin_small_patch4_window7_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )
