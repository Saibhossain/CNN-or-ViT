"""Learning rate scheduling with linear warmup and cosine decay."""

import math
from typing import Any, Optional
import torch
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class CosineWarmupScheduler(LambdaLR):
    """Linear warmup followed by cosine annealing learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = max(1, total_epochs)
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < self.warmup_epochs:
                # Linear warmup from min_lr_ratio to 1.0
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * float(current_epoch + 1) / float(max(1, self.warmup_epochs))
            else:
                # Cosine decay from 1.0 down to min_lr_ratio
                progress = float(current_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    base_lr: float = 5e-4,
) -> _LRScheduler:
    """Build standardized learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_type: 'cosine', 'step', or 'none'.
        epochs: Total training epochs.
        warmup_epochs: Number of linear warmup epochs.
        min_lr: Minimum target learning rate.
        base_lr: Initial maximum learning rate.

    Returns:
        Configured PyTorch LR scheduler.
    """
    clean = scheduler_type.lower().strip()
    min_ratio = min_lr / base_lr if base_lr > 0 else 1e-4

    if clean in ("cosine", "cosine_warmup"):
        return CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr_ratio=min_ratio,
        )
    elif clean == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
