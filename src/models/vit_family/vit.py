"""Vision Transformer Architecture Implementations (ViT-B/16, ViT-S/16, ViT-Ti/16, DeiT) for CNN vs. ViT Benchmark."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import timm


class VisionTransformerModel(nn.Module):
    """Vision Transformer wrapper supporting ViT-B/16, ViT-S/16, ViT-Ti/16, and DeiT."""

    def __init__(
        self,
        variant: str = "vit_base_patch16_224",
        num_classes: int = 100,
        pretrained: bool = False,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.variant = variant.lower().replace("-", "_").replace("/", "_")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.drop_rate = drop_rate

        # Map canonical aliases to timm model identifiers
        timm_name = "vit_base_patch16_224"
        if any(k in self.variant for k in ["vit_b16", "vit_base", "vit_b_16"]):
            timm_name = "vit_base_patch16_224"
        elif any(k in self.variant for k in ["vit_s16", "vit_small", "vit_s_16"]):
            timm_name = "vit_small_patch16_224"
        elif any(k in self.variant for k in ["vit_ti16", "vit_tiny", "vit_ti_16", "vit_t16"]):
            timm_name = "vit_tiny_patch16_224"
        elif any(k in self.variant for k in ["deit_s16", "deit_small"]):
            timm_name = "deit_small_patch16_224"
        elif any(k in self.variant for k in ["deit_b16", "deit_base"]):
            timm_name = "deit_base_patch16_224"

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
            # Fallback to standard base ViT architecture
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                drop_rate=drop_rate,
                **kwargs,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens and class token representations."""
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


def build_vit_b16(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> VisionTransformerModel:
    return VisionTransformerModel(
        variant="vit_base_patch16_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )


def build_vit_s16(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> VisionTransformerModel:
    return VisionTransformerModel(
        variant="vit_small_patch16_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )


def build_vit_ti16(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> VisionTransformerModel:
    return VisionTransformerModel(
        variant="vit_tiny_patch16_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )


def build_deit_s16(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> VisionTransformerModel:
    return VisionTransformerModel(
        variant="deit_small_patch16_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )


def build_deit_b16(
    num_classes: int = 100,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> VisionTransformerModel:
    return VisionTransformerModel(
        variant="deit_base_patch16_224",
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        drop_rate=drop_rate,
        **kwargs,
    )
