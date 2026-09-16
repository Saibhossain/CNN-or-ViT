"""Probability calibration diagnostics, ECE, MCE, and reliability diagrams."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.evaluation.schemas import CalibrationMetrics


def compute_calibration_metrics(
    probabilities: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor, List[int]],
    num_bins: int = 15,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|

    Args:
        probabilities: Predicted class probabilities matrix of shape [N, num_classes].
        targets: True integer class indices [N].
        num_bins: Number of confidence bins (default 15).

    Returns:
        CalibrationMetrics dataclass with ECE, MCE, and reliability diagram data.
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = np.asarray(probabilities)

    if isinstance(targets, torch.Tensor):
        y_true = targets.detach().cpu().numpy()
    else:
        y_true = np.asarray(targets)

    if len(y_true) == 0:
        return CalibrationMetrics(ece=0.0, mce=0.0, brier_score=0.0, num_bins=num_bins)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    reliability_diagram: List[Dict[str, Any]] = []

    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)

            ece += (bin_count / total_samples) * bin_error
            mce = max(mce, bin_error)

            reliability_diagram.append({
                "bin_index": i,
                "bin_lower": round(float(bin_lower), 4),
                "bin_upper": round(float(bin_upper), 4),
                "sample_count": bin_count,
                "accuracy": round(bin_acc * 100.0, 2),
                "confidence": round(bin_conf * 100.0, 2),
                "calibration_gap": round(bin_error * 100.0, 2),
            })

    # Multiclass Brier score
    num_classes = probs.shape[1]
    one_hot = np.zeros((total_samples, num_classes), dtype=np.float32)
    for i, t in enumerate(y_true):
        if 0 <= t < num_classes:
            one_hot[i, t] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return CalibrationMetrics(
        ece=round(ece * 100.0, 2),
        mce=round(mce * 100.0, 2),
        brier_score=round(brier, 4),
        num_bins=num_bins,
        reliability_diagram=reliability_diagram,
    )
