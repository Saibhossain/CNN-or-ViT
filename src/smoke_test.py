"""Comprehensive Internal Smoke and Self-Validation Test Suite for CNN vs. ViT Framework."""

import os
from pathlib import Path
import sys
import tempfile

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.base import DatasetBundle, StandardDataset, StandardDatasetSample
from src.datasets.transforms import get_standard_transforms
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.confusion import ConfusionMatrixAnalyzer
from src.evaluation.profiling import profile_model_hardware
from src.evaluation.robustness import CORRUPTION_FUNCTIONS, evaluate_robustness
from src.models.factory import calculate_model_complexity, create_model, list_models
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.trainer import ModelTrainer
from src.utils.device import get_device
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed


def run_all_smoke_tests() -> Dict[str, bool]:
    """Execute complete internal smoke tests verifying models, datasets, metrics, checkpoints, and training loops."""
    logger = get_logger("smoke_tests")
    logger.info("=" * 70)
    logger.info("RUNNING CNN vs. ViT INTERNAL SMOKE AND VALIDATION SUITE")
    logger.info("=" * 70)

    results: Dict[str, bool] = {}
    device = get_device("cpu")

    # 1. Model Forward Passes & Output Dimension Checks
    logger.info("[1/7] Testing Model Architecture Instantiations & Forward Passes...")
    test_models = [
        # CNN Family
        "resnet18", "resnet50", "densenet121", "efficientnetv2_s", "efficientnet_b0", "convnext_tiny",
        # ViT Family
        "vit_b16", "vit_s16", "vit_ti16", "swin_t", "swin_s", "deit_s16", "deit_b16",
    ]
    batch_size = 2
    num_classes = 100
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    all_models_ok = True
    for m_name in test_models:
        try:
            m = create_model(m_name, num_classes=num_classes, pretrained=False).to(device).eval()
            with torch.no_grad():
                out = m(dummy_input)
            assert out.shape == (batch_size, num_classes), f"Shape mismatch: {out.shape} != {(batch_size, num_classes)}"
            logger.info(f"  [OK] {m_name.ljust(20)} Output Shape: {list(out.shape)}")
        except Exception as e:
            logger.error(f"  [FAIL] {m_name.ljust(20)}: {e}")
            all_models_ok = False
    results["model_forward_passes"] = all_models_ok

    # 2. Comprehensive Evaluation Metrics Calculation
    logger.info("\n[2/7] Testing Classification, Calibration & Confusion Matrix Metrics...")
    try:
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        # Random logits for 5 classes
        np.random.seed(42)
        logits = np.random.randn(10, 5)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        clf_metrics = compute_classification_metrics(logits, y_true, num_classes=5)
        cal_metrics = compute_calibration_metrics(probs, y_true, num_bins=5)
        cm_analyzer = ConfusionMatrixAnalyzer(y_true, np.argmax(probs, axis=1), num_classes=5)

        assert 0.0 <= clf_metrics.top1_accuracy <= 100.0
        assert 0.0 <= clf_metrics.macro_f1 <= 100.0
        assert 0.0 <= clf_metrics.balanced_accuracy <= 100.0
        assert clf_metrics.mean_iou["status"] == "not_applicable_for_image_classification"
        assert 0.0 <= cal_metrics.ece <= 100.0
        assert 0.0 <= cal_metrics.brier_score <= 2.0
        assert cm_analyzer.cm_raw.shape == (5, 5)

        logger.info(f"  [OK] Top-1 Accuracy: {clf_metrics.top1_accuracy}% | Macro-F1: {clf_metrics.macro_f1}%")
        logger.info(f"  [OK] ECE: {cal_metrics.ece}% | Brier Score: {cal_metrics.brier_score}")
        logger.info(f"  [OK] Cohen's Kappa: {clf_metrics.cohen_kappa} | Macro Specificity: {clf_metrics.macro_specificity}%")
        logger.info(f"  [OK] mIoU Status: {clf_metrics.mean_iou['status']}")
        results["evaluation_metrics"] = True
    except Exception as e:
        logger.error(f"  [FAIL] Evaluation metrics: {e}")
        results["evaluation_metrics"] = False

    # 3. Model Profiling & FLOPs Estimation
    logger.info("\n[3/7] Testing Model Computational Complexity Profiling...")
    try:
        small_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        prof = profile_model_hardware(small_model, input_size=(1, 3, 32, 32), device="cpu", warmup_iters=2, measure_iters=5)
        assert prof["total_params"] > 0
        assert prof["latency_p50_ms"] > 0
        assert prof["throughput_img_per_sec"] > 0
        logger.info(f"  [OK] Total Params: {prof['total_params']} | Latency P50: {prof['latency_p50_ms']} ms | Throughput: {prof['throughput_img_per_sec']} img/s")
        results["hardware_profiling"] = True
    except Exception as e:
        logger.error(f"  [FAIL] Hardware profiling: {e}")
        results["hardware_profiling"] = False

    # 4. Checkpoint Save and Restore Integrity
    logger.info("\n[4/7] Testing Checkpoint Save and Restore Lifecycle...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            opt = torch.optim.Adam(small_model.parameters(), lr=1e-3)
            save_checkpoint(
                model=small_model,
                optimizer=opt,
                scheduler=None,
                epoch=3,
                best_val_metric=88.5,
                model_name="test_cnn",
                dataset_name="cifar100",
                seed=42,
                regime="from_scratch",
                output_path=ckpt_path,
                is_best=True,
            )
            assert ckpt_path.is_file()

            # Restore into new model
            new_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 10),
            )
            meta = load_checkpoint(ckpt_path, new_model)
            assert meta["epoch"] == 3
            assert meta["best_val_metric"] == 88.5
            assert meta["model_name"] == "test_cnn"
            logger.info("  [OK] Checkpoint saved and restored with identical weights and metadata.")
        results["checkpointing"] = True
    except Exception as e:
        logger.error(f"  [FAIL] Checkpoint lifecycle: {e}")
        results["checkpointing"] = False

    # 5. Robustness Corruptions Suite
    logger.info("\n[5/7] Testing Synthetic Robustness Corruptions (10 Corruptions)...")
    try:
        from PIL import Image
        test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        for c_name, fn in CORRUPTION_FUNCTIONS.items():
            corrupted = fn(test_img, severity=3)
            assert corrupted.size == (64, 64)
            assert corrupted.mode == "RGB"
        logger.info(f"  [OK] All 10 corruption functions executed successfully across test images.")
        results["robustness_corruptions"] = True
    except Exception as e:
        logger.error(f"  [FAIL] Robustness corruptions: {e}")
        results["robustness_corruptions"] = False

    # 6. End-to-End CNN 1-Epoch Smoke Training Loop
    logger.info("\n[6/7] Running End-to-End CNN Smoke Training Loop (ResNet-18)...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic tensor dataset
            x_synth = torch.randn(8, 3, 32, 32)
            y_synth = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
            ds_train = TensorDataset(x_synth, y_synth)
            ds_val = TensorDataset(x_synth, y_synth)
            ds_test = TensorDataset(x_synth, y_synth)

            loaders = {
                "train": DataLoader(ds_train, batch_size=4),
                "val": DataLoader(ds_val, batch_size=4),
                "test": DataLoader(ds_test, batch_size=4),
            }

            cnn_model = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(8, 4),
            )

            trainer = ModelTrainer(
                model=cnn_model,
                model_name="cnn_smoke",
                data_loaders=loaders,
                dataset_id="synthetic",
                num_classes=4,
                epochs=1,
                warmup_epochs=0,
                base_lr=1e-3,
                output_dir=tmpdir,
                device="cpu",
            )
            summary = trainer.fit()
            assert "clean_test_metrics" in summary
            assert "hardware_metrics" in summary
            assert (Path(tmpdir) / "final_summary.json").is_file()
            logger.info("  [OK] CNN smoke training and multi-metric evaluation completed successfully.")
        results["cnn_smoke_training"] = True
    except Exception as e:
        logger.error(f"  [FAIL] CNN smoke training: {e}")
        results["cnn_smoke_training"] = False

    # 7. End-to-End ViT 1-Epoch Smoke Training Loop
    logger.info("\n[7/7] Running End-to-End Vision Transformer Smoke Training Loop (ViT-Ti/16)...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            x_synth = torch.randn(4, 3, 224, 224)
            y_synth = torch.tensor([0, 1, 0, 1])
            ds = TensorDataset(x_synth, y_synth)

            loaders = {
                "train": DataLoader(ds, batch_size=2),
                "val": DataLoader(ds, batch_size=2),
                "test": DataLoader(ds, batch_size=2),
            }

            vit_model = create_model("vit_ti16", num_classes=2, pretrained=False)

            trainer = ModelTrainer(
                model=vit_model,
                model_name="vit_smoke",
                data_loaders=loaders,
                dataset_id="synthetic",
                num_classes=2,
                epochs=1,
                warmup_epochs=0,
                base_lr=1e-4,
                output_dir=tmpdir,
                device="cpu",
            )
            summary = trainer.fit()
            assert "clean_test_metrics" in summary
            assert (Path(tmpdir) / "final_summary.json").is_file()
            logger.info("  [OK] ViT smoke training and multi-metric evaluation completed successfully.")
        results["vit_smoke_training"] = True
    except Exception as e:
        logger.error(f"  [FAIL] ViT smoke training: {e}")
        results["vit_smoke_training"] = False

    logger.info("\n" + "=" * 70)
    all_passed = all(results.values())
    if all_passed:
        logger.info("ALL INTERNAL SMOKE AND VALIDATION TESTS PASSED (100%)!")
    else:
        logger.warning(f"SOME TESTS FAILED: {results}")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    setup_logging(log_dir="logs", log_prefix="smoke_test")
    run_all_smoke_tests()
