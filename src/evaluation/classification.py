"""Comprehensive classification metrics computation for CNN vs. ViT model comparison."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.schemas import ClassificationMetrics


def compute_topk_accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
) -> float:
    """Compute Top-k classification accuracy in percentage [0, 100]."""
    if len(targets) == 0:
        return 0.0
    num_classes = logits.shape[1]
    if k >= num_classes:
        return 100.0
    topk_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = sum(targets[i] in topk_preds[i] for i in range(len(targets)))
    return round((correct / len(targets)) * 100.0, 2)


def compute_multiclass_brier_score(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None,
) -> float:
    """Compute multiclass Brier score: (1/N) * sum_i sum_k (p_{ik} - y_{ik})^2."""
    if len(targets) == 0:
        return 0.0
    if probabilities.ndim > 1:
        num_classes = probabilities.shape[1]
    elif num_classes is None:
        num_classes = int(max(targets)) + 1 if len(targets) > 0 else 1
    n = len(targets)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    for i, t in enumerate(targets):
        if 0 <= t < num_classes:
            one_hot[i, t] = 1.0
    diff = probabilities - one_hot
    brier = np.mean(np.sum(diff ** 2, axis=1))
    return round(float(brier), 4)


def compute_per_class_sensitivity_specificity(
    cm: np.ndarray,
    num_classes: int,
) -> Tuple[Dict[int, float], Dict[int, float], float, float, float, float]:
    """Compute per-class sensitivity (TPR) and specificity (TNR) from confusion matrix.

    TPR_k = TP_k / (TP_k + FN_k)
    TNR_k = TN_k / (TN_k + FP_k)

    Returns:
        Tuple of (sensitivities_dict, specificities_dict, macro_sens, weighted_sens, macro_spec, weighted_spec).
    """
    total_samples = np.sum(cm)
    if total_samples == 0:
        return {}, {}, 0.0, 0.0, 0.0, 0.0

    sensitivities: Dict[int, float] = {}
    specificities: Dict[int, float] = {}
    class_weights: List[float] = []
    sens_list: List[float] = []
    spec_list: List[float] = []

    for k in range(num_classes):
        tp = float(cm[k, k])
        fn = float(np.sum(cm[k, :]) - tp)
        fp = float(np.sum(cm[:, k]) - tp)
        tn = float(total_samples - (tp + fn + fp))

        # Sensitivity (Recall / TPR)
        sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        # Specificity (TNR)
        spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 100.0

        sensitivities[k] = round(sens, 2)
        specificities[k] = round(spec, 2)

        weight = float(np.sum(cm[k, :]))
        class_weights.append(weight)
        sens_list.append(sens)
        spec_list.append(spec)

    macro_sens = round(float(np.mean(sens_list)), 2) if sens_list else 0.0
    macro_spec = round(float(np.mean(spec_list)), 2) if spec_list else 0.0

    total_weight = sum(class_weights)
    if total_weight > 0:
        weighted_sens = round(float(sum(s * w for s, w in zip(sens_list, class_weights)) / total_weight), 2)
        weighted_spec = round(float(sum(s * w for s, w in zip(spec_list, class_weights)) / total_weight), 2)
    else:
        weighted_sens = macro_sens
        weighted_spec = macro_spec

    return sensitivities, specificities, macro_sens, weighted_sens, macro_spec, weighted_spec


def compute_auroc_and_ap(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute multiclass One-vs-Rest AUROC and Average Precision (mAP) safely.

    Returns:
        Tuple of (auroc_macro, auroc_weighted, ap_macro, ap_weighted).
    """
    unique_targets = set(targets)
    if len(unique_targets) < 2 or len(targets) < num_classes:
        return None, None, None, None

    n = len(targets)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    for i, t in enumerate(targets):
        if 0 <= t < num_classes:
            one_hot[i, t] = 1.0

    auroc_macro, auroc_weighted = None, None
    ap_macro, ap_weighted = None, None

    try:
        auroc_macro = round(
            float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro")), 4
        )
        auroc_weighted = round(
            float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="weighted")), 4
        )
    except Exception:
        pass

    try:
        ap_macro = round(
            float(average_precision_score(one_hot, probabilities, average="macro")), 4
        )
        ap_weighted = round(
            float(average_precision_score(one_hot, probabilities, average="weighted")), 4
        )
    except Exception:
        pass

    return auroc_macro, auroc_weighted, ap_macro, ap_weighted


def compute_classification_metrics(
    logits_or_probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor, List[int]],
    num_classes: int,
    is_logits: bool = True,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Compute complete suite of classification metrics for CNN vs ViT model evaluation.

    Args:
        logits_or_probs: Raw output logits or probability matrix of shape [N, num_classes].
        targets: Ground truth class integer indices [N].
        num_classes: Total number of classes.
        is_logits: If True, apply softmax to compute probability distributions.
        class_names: Optional list of class string names for per-class breakdowns.

    Returns:
        ClassificationMetrics dataclass instance.
    """
    if isinstance(logits_or_probs, torch.Tensor):
        arr = logits_or_probs.detach().cpu().numpy()
    else:
        arr = np.asarray(logits_or_probs)

    if isinstance(targets, torch.Tensor):
        y_true = targets.detach().cpu().numpy()
    else:
        y_true = np.asarray(targets)

    if is_logits:
        # Softmax with numerical stability
        exp_logits = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    else:
        probs = arr

    if probs.ndim > 1:
        num_classes = probs.shape[1]

    preds = np.argmax(probs, axis=-1)

    # 1. Accuracies
    top1_acc = round(float(np.mean(preds == y_true) * 100.0), 2)
    top5_acc = compute_topk_accuracy(probs, y_true, k=5) if num_classes >= 5 else None

    # 2. Precision, Recall, F1 (Macro, Weighted, Micro)
    macro_prec = round(float(precision_score(y_true, preds, average="macro", zero_division=0) * 100.0), 2)
    weighted_prec = round(float(precision_score(y_true, preds, average="weighted", zero_division=0) * 100.0), 2)
    micro_prec = round(float(precision_score(y_true, preds, average="micro", zero_division=0) * 100.0), 2)

    macro_rec = round(float(recall_score(y_true, preds, average="macro", zero_division=0) * 100.0), 2)
    weighted_rec = round(float(recall_score(y_true, preds, average="weighted", zero_division=0) * 100.0), 2)
    micro_rec = round(float(recall_score(y_true, preds, average="micro", zero_division=0) * 100.0), 2)

    macro_f1 = round(float(f1_score(y_true, preds, average="macro", zero_division=0) * 100.0), 2)
    weighted_f1 = round(float(f1_score(y_true, preds, average="weighted", zero_division=0) * 100.0), 2)
    micro_f1 = round(float(f1_score(y_true, preds, average="micro", zero_division=0) * 100.0), 2)

    # 3. Balanced Accuracy
    bal_acc = round(float(balanced_accuracy_score(y_true, preds) * 100.0), 2)

    # 4. Cohen's Kappa
    kappa = round(float(cohen_kappa_score(y_true, preds)), 4)

    # 5. Sensitivity & Specificity from Confusion Matrix
    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
    (
        sens_dict,
        spec_dict,
        macro_sens,
        weighted_sens,
        macro_spec,
        weighted_spec,
    ) = compute_per_class_sensitivity_specificity(cm, num_classes)

    # Format per-class sensitivity & specificity
    per_class_sens_formatted: Dict[str, float] = {}
    per_class_spec_formatted: Dict[str, float] = {}
    for k in range(num_classes):
        name = class_names[k] if class_names and k < len(class_names) else f"class_{k}"
        per_class_sens_formatted[name] = sens_dict.get(k, 0.0)
        per_class_spec_formatted[name] = spec_dict.get(k, 0.0)

    # 6. AUROC and Average Precision (mAP)
    auroc_macro, auroc_weighted, ap_macro, ap_weighted = compute_auroc_and_ap(
        probs, y_true, num_classes
    )

    # 7. Brier Score and Negative Log-Likelihood
    brier = compute_multiclass_brier_score(probs, y_true, num_classes)

    # Cross entropy / NLL calculation
    eps = 1e-12
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    nll = float(-np.mean([np.log(clipped_probs[i, y_true[i]]) for i in range(len(y_true))]))
    nll = round(nll, 4)

    return ClassificationMetrics(
        top1_accuracy=top1_acc,
        top5_accuracy=top5_acc,
        macro_precision=macro_prec,
        weighted_precision=weighted_prec,
        micro_precision=micro_prec,
        macro_recall=macro_rec,
        weighted_recall=weighted_rec,
        micro_recall=micro_rec,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        micro_f1=micro_f1,
        balanced_accuracy=bal_acc,
        macro_sensitivity=macro_sens,
        weighted_sensitivity=weighted_sens,
        macro_specificity=macro_spec,
        weighted_specificity=weighted_spec,
        cohen_kappa=kappa,
        auroc_macro_ovr=auroc_macro,
        auroc_weighted_ovr=auroc_weighted,
        ap_macro_ovr=ap_macro,
        ap_weighted_ovr=ap_weighted,
        brier_score=brier,
        negative_log_likelihood=nll,
        mean_iou={"value": None, "status": "not_applicable_for_image_classification"},
        per_class_sensitivity=per_class_sens_formatted,
        per_class_specificity=per_class_spec_formatted,
    )
