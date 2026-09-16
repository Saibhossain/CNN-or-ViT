"""Typed schemas and data models for model evaluation results."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ClassificationMetrics:
    """Standardized classification performance metrics."""

    top1_accuracy: float
    top5_accuracy: Optional[float]
    macro_precision: float
    weighted_precision: float
    micro_precision: float
    macro_recall: float
    weighted_recall: float
    micro_recall: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    balanced_accuracy: float
    macro_sensitivity: float
    weighted_sensitivity: float
    macro_specificity: float
    weighted_specificity: float
    cohen_kappa: float
    auroc_macro_ovr: Optional[float]
    auroc_weighted_ovr: Optional[float]
    ap_macro_ovr: Optional[float]
    ap_weighted_ovr: Optional[float]
    brier_score: float
    negative_log_likelihood: float
    mean_iou: Dict[str, Any] = field(
        default_factory=lambda: {"value": None, "status": "not_applicable_for_image_classification"}
    )
    per_class_sensitivity: Optional[Dict[str, float]] = None
    per_class_specificity: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationMetrics:
    """Probability calibration diagnostics."""

    ece: float  # Expected Calibration Error (%)
    mce: float  # Maximum Calibration Error (%)
    brier_score: float
    num_bins: int = 15
    reliability_diagram: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareProfile:
    """Computational complexity and inference latency profiling."""

    total_params: int
    trainable_params: int
    non_trainable_params: int
    param_size_mb: float
    flops_g: Optional[float]
    macs_g: Optional[float]
    batch_size_1_latency_ms: Optional[float]
    batch_size_32_latency_ms: Optional[float]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_img_per_sec: float
    peak_vram_mb: Optional[float]
    device: str
    warmup_iterations: int
    measurement_iterations: int
    input_resolution: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessResult:
    """Robustness evaluation across corruptions and severities."""

    clean_accuracy: float
    clean_macro_f1: float
    mean_corrupted_accuracy: float
    mean_corrupted_macro_f1: float
    mean_absolute_degradation: float
    mean_relative_degradation: float
    corruptions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
