"""Optimizer construction with weight decay parameter grouping."""

from typing import Any, Dict, List
import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Construct optimizer with parameter grouping (excluding biases & LayerNorm from decay).

    Args:
        model: PyTorch model module.
        optimizer_type: Optimizer name ('adamw', 'adam', 'sgd').
        lr: Base learning rate.
        weight_decay: L2 penalty factor.
        betas: Adam beta tuple.
        eps: Epsilon stability parameter.

    Returns:
        Instantiated PyTorch optimizer.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D normalization params do not receive weight decay
        if param.ndim <= 1 or "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    clean_type = optimizer_type.lower().strip()
    if clean_type == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif clean_type == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    elif clean_type == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    else:
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
