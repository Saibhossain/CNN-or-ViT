"""Training history tracking and serialization."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class TrainingHistory:
    """Tracks per-epoch metrics and exports structured history files."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record_epoch(
        self,
        epoch: int,
        lr: float,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        duration_sec: float,
    ) -> None:
        """Record a single completed training epoch."""
        rec = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics.get("loss"),
            "train_top1_acc": train_metrics.get("top1_accuracy"),
            "train_macro_f1": train_metrics.get("macro_f1"),
            "train_balanced_acc": train_metrics.get("balanced_accuracy"),
            "val_loss": val_metrics.get("loss"),
            "val_top1_acc": val_metrics.get("top1_accuracy"),
            "val_top5_acc": val_metrics.get("top5_accuracy"),
            "val_macro_f1": val_metrics.get("macro_f1"),
            "val_balanced_acc": val_metrics.get("balanced_accuracy"),
            "val_precision": val_metrics.get("macro_precision"),
            "val_recall": val_metrics.get("macro_recall"),
            "val_kappa": val_metrics.get("cohen_kappa"),
            "val_ece": val_metrics.get("ece"),
            "val_brier_score": val_metrics.get("brier_score"),
            "duration_sec": round(duration_sec, 2),
        }
        self.records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, output_dir: Union[str, Path]) -> None:
        """Export history to training_history.csv and training_history.json."""
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(out_d / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

        # CSV
        df = self.to_dataframe()
        df.to_csv(out_d / "training_history.csv", index=False)
