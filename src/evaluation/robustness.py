"""Synthetic Robustness Benchmarking Suite (10 Corruptions x 5 Severities) for Test Evaluation."""

import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.base import StandardDatasetSample
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.schemas import RobustnessResult


# 10 Standard Synthetic Corruption Functions
def apply_gaussian_noise(image: Image.Image, severity: int) -> Image.Image:
    sigmas = [0.04, 0.08, 0.12, 0.18, 0.26]
    sigma = sigmas[min(severity - 1, 4)]
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    corrupted = np.clip(arr + noise, 0.0, 1.0) * 255.0
    return Image.fromarray(corrupted.astype(np.uint8))


def apply_gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    radii = [0.5, 1.0, 1.5, 2.0, 3.0]
    return image.filter(ImageFilter.GaussianBlur(radius=radii[min(severity - 1, 4)]))


def apply_brightness(image: Image.Image, severity: int) -> Image.Image:
    factors = [1.2, 1.4, 1.6, 1.8, 2.0]
    return ImageEnhance.Brightness(image).enhance(factors[min(severity - 1, 4)])


def apply_contrast(image: Image.Image, severity: int) -> Image.Image:
    factors = [0.8, 0.6, 0.4, 0.2, 0.1]
    return ImageEnhance.Contrast(image).enhance(factors[min(severity - 1, 4)])


def apply_saturation(image: Image.Image, severity: int) -> Image.Image:
    factors = [0.7, 0.5, 0.3, 0.15, 0.0]
    return ImageEnhance.Color(image).enhance(factors[min(severity - 1, 4)])


def apply_rotation(image: Image.Image, severity: int) -> Image.Image:
    angles = [5, 10, 15, 25, 40]
    return image.rotate(angles[min(severity - 1, 4)], resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def apply_jpeg_compression(image: Image.Image, severity: int) -> Image.Image:
    qualities = [80, 60, 40, 20, 10]
    quality = qualities[min(severity - 1, 4)]
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_pixelate(image: Image.Image, severity: int) -> Image.Image:
    factors = [0.8, 0.6, 0.4, 0.25, 0.15]
    factor = factors[min(severity - 1, 4)]
    w, h = image.size
    small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), resample=Image.NEAREST)
    return small.resize((w, h), resample=Image.NEAREST)


def apply_defocus_blur(image: Image.Image, severity: int) -> Image.Image:
    radii = [1, 2, 3, 4, 6]
    return image.filter(ImageFilter.BoxBlur(radius=radii[min(severity - 1, 4)]))


def apply_random_occlusion(image: Image.Image, severity: int) -> Image.Image:
    mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    ratio = mask_ratios[min(severity - 1, 4)]
    arr = np.array(image).copy()
    h, w, _ = arr.shape
    occ_h, occ_w = int(h * ratio), int(w * ratio)
    top = np.random.randint(0, max(1, h - occ_h))
    left = np.random.randint(0, max(1, w - occ_w))
    arr[top:top + occ_h, left:left + occ_w, :] = 0
    return Image.fromarray(arr)


CORRUPTION_FUNCTIONS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "saturation": apply_saturation,
    "rotation": apply_rotation,
    "gaussian_noise": apply_gaussian_noise,
    "gaussian_blur": apply_gaussian_blur,
    "jpeg_compression": apply_jpeg_compression,
    "pixelate": apply_pixelate,
    "defocus_blur": apply_defocus_blur,
    "random_occlusion": apply_random_occlusion,
}


def evaluate_robustness(
    model: nn.Module,
    samples: List[StandardDatasetSample],
    eval_transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    batch_size: int = 64,
    clean_macro_f1: Optional[float] = None,
    clean_accuracy: Optional[float] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Benchmark model robustness across synthetic corruptions on the test set.

    Args:
        model: Evaluated PyTorch model.
        samples: List of clean test StandardDatasetSample objects.
        eval_transform: Evaluation transform (without ToTensor/Normalize applied to PIL, or function).
        device: PyTorch device.
        batch_size: Inference batch size.
        clean_macro_f1: Baseline clean test Macro-F1 score.
        clean_accuracy: Baseline clean test Top-1 Accuracy.
        corruptions: List of corruption names to evaluate (default all 10).
        severities: List of severity levels (default 1..5).

    Returns:
        Structured robustness results dictionary with degradation metrics.
    """
    model.eval()
    if corruptions is None:
        corruptions = list(CORRUPTION_FUNCTIONS.keys())
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    y_true = [s.label for s in samples]
    results_by_corruption: Dict[str, Any] = {}
    all_corrupted_f1s: List[float] = []
    all_corrupted_accs: List[float] = []

    # If clean baseline not provided, evaluate clean first
    if clean_macro_f1 is None or clean_accuracy is None:
        clean_preds = []
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                tensors = torch.stack([eval_transform(s.load_image()) for s in batch_samples]).to(device)
                logits = model(tensors)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                clean_preds.extend(preds.tolist())
        clean_macro_f1 = round(float(f1_score(y_true, clean_preds, average="macro", zero_division=0) * 100.0), 2)
        clean_accuracy = round(float(np.mean(np.array(clean_preds) == np.array(y_true)) * 100.0), 2)

    for corr_name in corruptions:
        corr_fn = CORRUPTION_FUNCTIONS.get(corr_name)
        if corr_fn is None:
            continue

        corr_severity_results: Dict[str, Any] = {}
        corr_f1s: List[float] = []
        corr_accs: List[float] = []

        for sev in severities:
            preds_list: List[int] = []
            probs_list: List[np.ndarray] = []

            with torch.no_grad():
                for i in range(0, len(samples), batch_size):
                    batch_samples = samples[i:i + batch_size]
                    corrupted_images = [corr_fn(s.load_image(), sev) for s in batch_samples]
                    tensors = torch.stack([eval_transform(img) for img in corrupted_images]).to(device)
                    logits = model(tensors)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                    preds_list.extend(preds.tolist())
                    probs_list.append(probs)

            probs_arr = np.concatenate(probs_list, axis=0) if probs_list else np.array([])
            f1 = round(float(f1_score(y_true, preds_list, average="macro", zero_division=0) * 100.0), 2)
            acc = round(float(np.mean(np.array(preds_list) == np.array(y_true)) * 100.0), 2)
            cal = compute_calibration_metrics(probs_arr, y_true)

            corr_f1s.append(f1)
            corr_accs.append(acc)
            all_corrupted_f1s.append(f1)
            all_corrupted_accs.append(acc)

            corr_severity_results[f"severity_{sev}"] = {
                "macro_f1": f1,
                "top1_accuracy": acc,
                "ece": cal.ece,
            }

        results_by_corruption[corr_name] = {
            "severities": corr_severity_results,
            "mean_f1": round(float(np.mean(corr_f1s)), 2),
            "mean_acc": round(float(np.mean(corr_accs)), 2),
        }

    mean_corr_f1 = round(float(np.mean(all_corrupted_f1s)), 2) if all_corrupted_f1s else 0.0
    mean_corr_acc = round(float(np.mean(all_corrupted_accs)), 2) if all_corrupted_accs else 0.0

    deg_f1 = round(clean_macro_f1 - mean_corr_f1, 2)
    rel_deg_f1 = round((deg_f1 / clean_macro_f1 * 100.0) if clean_macro_f1 > 0 else 0.0, 2)

    deg_acc = round(clean_accuracy - mean_corr_acc, 2)
    rel_deg_acc = round((deg_acc / clean_accuracy * 100.0) if clean_accuracy > 0 else 0.0, 2)

    return {
        "clean_macro_f1": clean_macro_f1,
        "clean_top1_acc": clean_accuracy,
        "mean_corrupted_macro_f1": mean_corr_f1,
        "mean_corrupted_top1_acc": mean_corr_acc,
        "mean_degradation_f1": deg_f1,
        "relative_degradation_f1": rel_deg_f1,
        "mean_degradation_acc": deg_acc,
        "relative_degradation_acc": rel_deg_acc,
        "corruptions": results_by_corruption,
    }
