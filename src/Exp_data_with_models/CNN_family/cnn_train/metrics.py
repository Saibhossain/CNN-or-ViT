"""Compatibility metrics module delegating to src.evaluation."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import (
    compute_classification_metrics,
    compute_multiclass_brier_score,
)
from src.evaluation.profiling import profile_model_hardware


def calculate_classification_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Calculate multi-dimensional classification performance metrics."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if num_classes is None:
        num_classes = max(int(np.max(y_true)) + 1, int(np.max(y_pred)) + 1) if len(y_true) > 0 else 1

    if y_prob is not None:
        probs = np.asarray(y_prob)
    else:
        # Build one-hot probabilities from predictions
        probs = np.zeros((len(y_pred), num_classes), dtype=np.float32)
        for i, p in enumerate(y_pred):
            if 0 <= p < num_classes:
                probs[i, p] = 1.0

    metrics = compute_classification_metrics(probs, y_true, num_classes, is_logits=False)
    cal = compute_calibration_metrics(probs, y_true)

    return {
        "top1_accuracy": metrics.top1_accuracy,
        "top5_accuracy": metrics.top5_accuracy or 0.0,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "cohen_kappa": metrics.cohen_kappa,
        "ece": cal.ece,
        "brier_score": cal.brier_score,
    }


def calculate_ece(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
) -> float:
    """Calculate Expected Calibration Error (normalized [0, 1])."""
    cal = compute_calibration_metrics(y_prob, y_true, num_bins=n_bins)
    return round(cal.ece / 100.0, 4)


def calculate_brier_score(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, torch.Tensor],
) -> float:
    """Calculate multiclass Brier score."""
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()
    num_classes = y_prob.shape[1] if hasattr(y_prob, "shape") and len(y_prob.shape) > 1 else 2
    return compute_multiclass_brier_score(np.asarray(y_prob), np.asarray(y_true), num_classes)


def benchmark_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark inference latency percentiles and throughput."""
    prof = profile_model_hardware(
        model,
        input_size=input_size,
        device=device,
        warmup_iters=warmup_runs,
        measure_iters=benchmark_runs,
    )
    return {
        "latency_mean_ms": prof["batch_size_1_latency_ms"] or 0.0,
        "latency_p50_ms": prof["latency_p50_ms"],
        "latency_p95_ms": prof["latency_p95_ms"],
        "latency_p99_ms": prof["latency_p99_ms"],
        "throughput_img_per_sec": prof["throughput_img_per_sec"],
        "peak_vram_mb": prof["peak_vram_mb"] or 0.0,
    }
