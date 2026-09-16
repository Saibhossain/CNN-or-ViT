"""Confusion matrix computation, normalization, per-class decomposition, and export."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class ConfusionMatrixAnalyzer:
    """Computes, decomposes, and exports raw and normalized confusion matrices."""

    def __init__(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.num_classes = num_classes
        self.class_names = (
            class_names
            if class_names and len(class_names) == num_classes
            else [f"class_{i}" for i in range(num_classes)]
        )

        self.cm_raw = confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=list(range(num_classes)),
        )

        # Row-normalized (recall-normalized percentages)
        row_sums = self.cm_raw.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cm_normalized = np.where(row_sums > 0, (self.cm_raw / row_sums) * 100.0, 0.0)
            self.cm_normalized = np.round(self.cm_normalized, 2)

    def get_per_class_breakdown(self) -> List[Dict[str, Any]]:
        """Decompose confusion matrix into per-class True/False Positive/Negative counts."""
        total_samples = int(np.sum(self.cm_raw))
        breakdown: List[Dict[str, Any]] = []

        for k in range(self.num_classes):
            tp = int(self.cm_raw[k, k])
            fn = int(np.sum(self.cm_raw[k, :]) - tp)
            fp = int(np.sum(self.cm_raw[:, k]) - tp)
            tn = int(total_samples - (tp + fn + fp))

            sens = round((tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0, 2)
            spec = round((tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 100.0, 2)
            prec = round((tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0, 2)
            f1 = round((2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0, 2)

            breakdown.append({
                "class_index": k,
                "class_name": self.class_names[k],
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1_score": f1,
                "support": tp + fn,
            })

        return breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert confusion matrix and per-class metrics to serializable dictionary."""
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "raw_matrix": self.cm_raw.tolist(),
            "normalized_matrix": self.cm_normalized.tolist(),
            "per_class_breakdown": self.get_per_class_breakdown(),
        }

    def save_csv(self, output_path: Union[str, Path], normalized: bool = False) -> None:
        """Export confusion matrix to CSV."""
        matrix = self.cm_normalized if normalized else self.cm_raw
        df = pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Export confusion matrix data to JSON."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_plot(self, output_path: Union[str, Path], title: str = "Confusion Matrix") -> None:
        """Generate and save confusion matrix heatmap image."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(self.cm_normalized, interpolation="nearest", cmap="Blues")
            fig.colorbar(im)

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)

            if self.num_classes <= 20:
                tick_marks = np.arange(self.num_classes)
                ax.set_xticks(tick_marks)
                ax.set_xticklabels(self.class_names, rotation=45, ha="right")
                ax.set_yticks(tick_marks)
                ax.set_yticklabels(self.class_names)

            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_p, dpi=300)
            plt.close()
        except Exception:
            pass
