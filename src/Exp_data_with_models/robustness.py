"""Synthetic Image Corruptions and Robustness Evaluation Engine.

Evaluates trained neural network architectures across standard image corruptions
and distribution shifts (Noise, Blur, Photometric, Compression, Occlusion) across 5
severity levels, computing Mean Performance Degradation (Dm) and Relative Degradation (RDm).
"""

import io
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.Exp_data_with_models.CNN_family.cnn_train.metrics import calculate_classification_metrics


def corrupt_gaussian_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    """Add Gaussian noise scaled by severity."""
    scales = [0.04, 0.08, 0.12, 0.18, 0.26]
    scale = scales[min(max(1, severity) - 1, len(scales) - 1)]
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, scale, arr.shape)
    corrupted = np.clip(arr + noise, 0.0, 1.0) * 255.0
    return Image.fromarray(corrupted.astype(np.uint8))


def corrupt_shot_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    """Add Poisson shot noise."""
    rates = [60, 25, 12, 5, 3]
    rate = rates[min(max(1, severity) - 1, len(rates) - 1)]
    arr = np.array(image, dtype=np.float32) / 255.0
    corrupted = np.random.poisson(arr * rate) / float(rate)
    corrupted = np.clip(corrupted, 0.0, 1.0) * 255.0
    return Image.fromarray(corrupted.astype(np.uint8))


def corrupt_impulse_noise(image: Image.Image, severity: int = 1) -> Image.Image:
    """Add Salt-and-Pepper impulse noise."""
    amounts = [0.02, 0.05, 0.09, 0.15, 0.22]
    amount = amounts[min(max(1, severity) - 1, len(amounts) - 1)]
    arr = np.array(image).copy()
    num_salt = int(np.ceil(amount * arr.size * 0.5))
    num_pepper = int(np.ceil(amount * arr.size * 0.5))

    # Salt (white)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in arr.shape[:2]]
    arr[tuple(coords_salt)] = 255

    # Pepper (black)
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in arr.shape[:2]]
    arr[tuple(coords_pepper)] = 0

    return Image.fromarray(arr)


def corrupt_gaussian_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    """Apply Gaussian blur."""
    radii = [0.75, 1.25, 1.75, 2.5, 3.5]
    radius = radii[min(max(1, severity) - 1, len(radii) - 1)]
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_defocus_blur(image: Image.Image, severity: int = 1) -> Image.Image:
    """Apply Defocus blur (BoxBlur approximation)."""
    radii = [1, 2, 3, 4, 6]
    radius = radii[min(max(1, severity) - 1, len(radii) - 1)]
    return image.filter(ImageFilter.BoxBlur(radius=radius))


def corrupt_brightness(image: Image.Image, severity: int = 1) -> Image.Image:
    """Adjust brightness."""
    factors = [1.15, 1.3, 1.5, 0.7, 0.45]
    factor = factors[min(max(1, severity) - 1, len(factors) - 1)]
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def corrupt_contrast(image: Image.Image, severity: int = 1) -> Image.Image:
    """Adjust contrast."""
    factors = [0.8, 0.6, 0.4, 0.25, 0.12]
    factor = factors[min(max(1, severity) - 1, len(factors) - 1)]
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def corrupt_jpeg_compression(image: Image.Image, severity: int = 1) -> Image.Image:
    """Apply JPEG compression artifacts."""
    qualities = [60, 40, 25, 15, 7]
    quality = qualities[min(max(1, severity) - 1, len(qualities) - 1)]
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def corrupt_pixelate(image: Image.Image, severity: int = 1) -> Image.Image:
    """Apply pixelation downsampling."""
    factors = [0.7, 0.5, 0.35, 0.2, 0.1]
    factor = factors[min(max(1, severity) - 1, len(factors) - 1)]
    orig_w, orig_h = image.size
    small_w = max(1, int(orig_w * factor))
    small_h = max(1, int(orig_h * factor))
    small = image.resize((small_w, small_h), resample=Image.Resampling.NEAREST)
    return small.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)


def corrupt_random_occlusion(image: Image.Image, severity: int = 1) -> Image.Image:
    """Apply random rectangular cutout / occlusion."""
    scales = [0.15, 0.25, 0.35, 0.45, 0.55]
    scale = scales[min(max(1, severity) - 1, len(scales) - 1)]
    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    cut_h = int(h * scale)
    cut_w = int(w * scale)

    top = np.random.randint(0, max(1, h - cut_h))
    left = np.random.randint(0, max(1, w - cut_w))
    arr[top : top + cut_h, left : left + cut_w] = 0
    return Image.fromarray(arr)


CORRUPTIONS_REGISTRY: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_noise": corrupt_gaussian_noise,
    "shot_noise": corrupt_shot_noise,
    "impulse_noise": corrupt_impulse_noise,
    "gaussian_blur": corrupt_gaussian_blur,
    "defocus_blur": corrupt_defocus_blur,
    "brightness": corrupt_brightness,
    "contrast": corrupt_contrast,
    "jpeg_compression": corrupt_jpeg_compression,
    "pixelate": corrupt_pixelate,
    "random_occlusion": corrupt_random_occlusion,
}


class CorruptedDataset(Dataset):
    """Dataset wrapper applying a specific corruption and severity to test samples on the fly."""

    def __init__(
        self,
        base_dataset: Any,
        corruption_fn: Callable[[Image.Image, int], Image.Image],
        severity: int = 1,
        transform: Optional[transforms.Compose] = None,
    ):
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        # Extract sample
        sample = self.base_dataset.samples[idx]
        raw_img = sample.load_image()

        # Apply synthetic corruption
        corrupted_img = self.corruption_fn(raw_img, self.severity)

        # Apply evaluation transform
        if self.transform is not None:
            tensor_img = self.transform(corrupted_img)
        else:
            tensor_img = transforms.ToTensor()(corrupted_img)

        return tensor_img, sample.label, sample.sample_id


class RobustnessEvaluator:
    """Robustness evaluation engine computing degradation profiles across distribution shifts."""

    def __init__(
        self,
        model: nn.Module,
        test_dataset: Any,
        device: str = "cpu",
        batch_size: int = 64,
        num_classes: int = 100,
        eval_transform: Optional[transforms.Compose] = None,
    ):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        self.torch_device = torch.device(device)
        self.batch_size = batch_size
        self.num_classes = num_classes

        if eval_transform is None:
            self.eval_transform = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.eval_transform = eval_transform

    @torch.no_grad()
    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a DataLoader."""
        self.model.eval()
        all_targets: List[int] = []
        all_preds: List[int] = []
        all_probs: List[np.ndarray] = []

        for images, targets, _ in dataloader:
            images = images.to(self.torch_device, non_blocking=True)
            targets = targets.to(self.torch_device, non_blocking=True)

            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()

            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds)
            all_probs.append(probs)

        y_prob = np.concatenate(all_probs, axis=0) if all_probs else None
        return calculate_classification_metrics(all_targets, all_preds, y_prob, num_classes=self.num_classes)

    def evaluate_all_corruptions(
        self,
        severities: Optional[List[int]] = None,
        corruptions: Optional[List[str]] = None,
        clean_metrics: Optional[Dict[str, float]] = None,
        max_eval_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive robustness benchmark across all corruptions and severities.

        Args:
            severities: List of severity levels [1..5] (default [1, 2, 3, 4, 5]).
            corruptions: List of corruption names or None for all.
            clean_metrics: Clean performance metrics if already evaluated.
            max_eval_samples: Optional sample limit for fast testing.

        Returns:
            Dictionary containing:
                - 'clean_macro_f1', 'clean_top1_acc'
                - 'mean_corrupted_macro_f1', 'mean_corrupted_top1_acc'
                - 'mean_degradation_f1' (Dm = P_clean - P_corrupt)
                - 'relative_degradation_f1' (RDm = (P_clean - P_corrupt) / P_clean)
                - 'corruption_breakdown': dict per corruption and severity
        """
        if severities is None:
            severities = [1, 2, 3, 4, 5]
        if corruptions is None:
            target_corruptions = list(CORRUPTIONS_REGISTRY.keys())
        else:
            target_corruptions = [c for c in corruptions if c in CORRUPTIONS_REGISTRY]

        # 1. Clean Evaluation if not provided
        if clean_metrics is None:
            clean_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
            clean_metrics = self.evaluate_dataloader(clean_loader)

        clean_f1 = clean_metrics.get("macro_f1", 0.0)
        clean_acc = clean_metrics.get("top1_accuracy", 0.0)

        # Prepare evaluation dataset subset if limited
        eval_ds = self.test_dataset
        if max_eval_samples is not None and len(eval_ds) > max_eval_samples:
            # Wrap sample subset
            class SubsetWrapper:
                def __init__(self, samples):
                    self.samples = samples
                def __len__(self):
                    return len(self.samples)
            eval_ds = SubsetWrapper(self.test_dataset.samples[:max_eval_samples])

        breakdown: Dict[str, Dict[str, Any]] = {}
        all_corrupt_f1s: List[float] = []
        all_corrupt_accs: List[float] = []

        pbar = tqdm(
            total=len(target_corruptions) * len(severities),
            desc="[Robustness Evaluation]",
            leave=False,
        )

        for c_name in target_corruptions:
            fn = CORRUPTIONS_REGISTRY[c_name]
            breakdown[c_name] = {"severities": {}, "mean_f1": 0.0, "mean_acc": 0.0}
            c_f1s = []
            c_accs = []

            for sev in severities:
                c_dataset = CorruptedDataset(
                    base_dataset=eval_ds,
                    corruption_fn=fn,
                    severity=sev,
                    transform=self.eval_transform,
                )
                loader = DataLoader(
                    c_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                res = self.evaluate_dataloader(loader)
                f1_val = res.get("macro_f1", 0.0)
                acc_val = res.get("top1_accuracy", 0.0)

                breakdown[c_name]["severities"][f"severity_{sev}"] = {
                    "macro_f1": f1_val,
                    "top1_accuracy": acc_val,
                    "ece": res.get("ece", 0.0),
                }
                c_f1s.append(f1_val)
                c_accs.append(acc_val)
                all_corrupt_f1s.append(f1_val)
                all_corrupt_accs.append(acc_val)

                pbar.update(1)

            breakdown[c_name]["mean_f1"] = round(float(np.mean(c_f1s)), 2)
            breakdown[c_name]["mean_acc"] = round(float(np.mean(c_accs)), 2)

        pbar.close()

        mean_corrupt_f1 = float(np.mean(all_corrupt_f1s)) if all_corrupt_f1s else 0.0
        mean_corrupt_acc = float(np.mean(all_corrupt_accs)) if all_corrupt_accs else 0.0

        # Mean Degradation Dm = P_clean - P_corrupt
        mean_deg_f1 = clean_f1 - mean_corrupt_f1
        mean_deg_acc = clean_acc - mean_corrupt_acc

        # Relative Degradation RDm = (P_clean - P_corrupt) / P_clean
        rel_deg_f1 = (mean_deg_f1 / clean_f1) if clean_f1 > 0 else 0.0
        rel_deg_acc = (mean_deg_acc / clean_acc) if clean_acc > 0 else 0.0

        return {
            "clean_macro_f1": round(clean_f1, 2),
            "clean_top1_acc": round(clean_acc, 2),
            "mean_corrupted_macro_f1": round(mean_corrupt_f1, 2),
            "mean_corrupted_top1_acc": round(mean_corrupt_acc, 2),
            "mean_degradation_f1": round(mean_deg_f1, 2),
            "mean_degradation_acc": round(mean_deg_acc, 2),
            "relative_degradation_f1": round(rel_deg_f1 * 100.0, 2),
            "relative_degradation_acc": round(rel_deg_acc * 100.0, 2),
            "corruptions_evaluated": len(target_corruptions),
            "severities_evaluated": severities,
            "corruption_breakdown": breakdown,
        }
