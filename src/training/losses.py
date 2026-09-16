"""Loss functions and criteria for CNN vs. ViT training."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.0,
    weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build standardized loss criterion.

    Args:
        loss_type: Loss identifier ('cross_entropy', 'label_smoothing', 'focal').
        label_smoothing: Label smoothing regularization epsilon [0.0, 0.2].
        weight: Optional per-class loss weight tensor.

    Returns:
        PyTorch nn.Module criterion.
    """
    clean = loss_type.lower().strip()
    if clean in ("cross_entropy", "ce", "label_smoothing"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    elif clean == "focal":
        return FocalLoss(gamma=2.0, weight=weight)
    else:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for addressing class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
