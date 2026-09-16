"""Training callbacks for monitoring, early stopping, and checkpoint triggers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Callback(ABC):
    """Abstract base class for training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
    ) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


class EarlyStopping(Callback):
    """Early stopping callback monitoring validation metric."""

    def __init__(
        self,
        monitor: str = "val_macro_f1",
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
    ) -> None:
        current = val_metrics.get("macro_f1", 0.0) if "f1" in self.monitor else val_metrics.get("top1_accuracy", 0.0)

        if self.best_score is None:
            self.best_score = current
            self.counter = 0
        elif (self.mode == "max" and current < self.best_score + self.min_delta) or (
            self.mode == "min" and current > self.best_score - self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                trainer.stop_training = True
        else:
            self.best_score = current
            self.counter = 0
