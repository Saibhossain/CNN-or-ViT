"""Model checkpoint saving, metadata serialization, and restoration."""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


def get_git_commit_hash() -> Optional[str]:
    """Retrieve current git commit hash if running in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_metric: float,
    model_name: str,
    dataset_name: str,
    seed: int,
    regime: str,
    output_path: Union[str, Path],
    hyperparameters: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint with complete training state and provenance metadata.

    Args:
        model: PyTorch model module.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        epoch: Current training epoch index.
        best_val_metric: Highest validation score achieved.
        model_name: Architecture identifier.
        dataset_name: Dataset identifier.
        seed: Random seed.
        regime: 'from_scratch' or 'imagenet1k_pretrained'.
        output_path: Destination file path for checkpoint.
        hyperparameters: Dictionary of training hyperparameters.
        is_best: Whether this checkpoint achieved highest validation score.

    Returns:
        Resolved Path to saved checkpoint.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_metric": best_val_metric,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "regime": regime,
        "hyperparameters": hyperparameters or {},
        "git_commit": get_git_commit_hash(),
        "is_best": is_best,
    }

    torch.save(state, out_p)
    return out_p.resolve()


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load model weights and optional optimizer/scheduler state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        model: PyTorch model to populate.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Target device to map tensors to.

    Returns:
        Dictionary of checkpoint metadata (epoch, best_val_metric, etc.).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")

    map_loc = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_loc)

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "epoch": state.get("epoch", 0),
        "best_val_metric": state.get("best_val_metric", 0.0),
        "model_name": state.get("model_name"),
        "dataset_name": state.get("dataset_name"),
        "seed": state.get("seed"),
        "regime": state.get("regime"),
        "hyperparameters": state.get("hyperparameters", {}),
        "git_commit": state.get("git_commit"),
    }
