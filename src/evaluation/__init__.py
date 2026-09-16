"""Evaluation package for CNN vs. ViT Benchmarking Suite."""

from src.evaluation.aggregation import (
    aggregate_experiment_results,
    save_comparison_summary,
)
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import (
    compute_auroc_and_ap,
    compute_classification_metrics,
    compute_multiclass_brier_score,
    compute_per_class_sensitivity_specificity,
    compute_topk_accuracy,
)
from src.evaluation.confusion import ConfusionMatrixAnalyzer
from src.evaluation.profiling import profile_model_hardware
from src.evaluation.robustness import CORRUPTION_FUNCTIONS, evaluate_robustness
from src.evaluation.schemas import (
    CalibrationMetrics,
    ClassificationMetrics,
    HardwareProfile,
    RobustnessResult,
)

__all__ = [
    "ClassificationMetrics",
    "CalibrationMetrics",
    "HardwareProfile",
    "RobustnessResult",
    "compute_classification_metrics",
    "compute_topk_accuracy",
    "compute_multiclass_brier_score",
    "compute_per_class_sensitivity_specificity",
    "compute_auroc_and_ap",
    "compute_calibration_metrics",
    "ConfusionMatrixAnalyzer",
    "evaluate_robustness",
    "CORRUPTION_FUNCTIONS",
    "profile_model_hardware",
    "aggregate_experiment_results",
    "save_comparison_summary",
]
