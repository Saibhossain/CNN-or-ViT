"""Compatibility wrapper for CNNTrainer delegating to unified ModelTrainer."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import torch.nn as nn

from src.training.trainer import ModelTrainer


class CNNTrainer(ModelTrainer):
    """Compatibility wrapper preserving CNNTrainer interface for legacy imports and tests."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        data_loaders: Dict[str, DataLoader],
        dataset_id: str = "cifar100",
        num_classes: int = 100,
        epochs: int = 100,
        warmup_epochs: int = 5,
        base_lr: float = 5e-4,
        weight_decay: float = 0.05,
        data_fraction: int = 100,
        seed: int = 42,
        pretrained: bool = False,
        device: Optional[Union[str, Any]] = None,
        use_amp: bool = True,
        output_dir: Optional[Any] = None,
        quick_test: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_name=model_name,
            data_loaders=data_loaders,
            dataset_id=dataset_id,
            num_classes=num_classes,
            epochs=1 if quick_test else epochs,
            warmup_epochs=0 if quick_test else warmup_epochs,
            base_lr=base_lr,
            weight_decay=weight_decay,
            data_fraction=data_fraction,
            seed=seed,
            pretrained=pretrained,
            device=device,
            use_amp=use_amp,
            output_dir=output_dir,
            **kwargs,
        )


__all__ = ["CNNTrainer", "ModelTrainer"]
