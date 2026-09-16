"""Unified Model Trainer for CNN and Vision Transformer architectures."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.base import StandardDatasetSample
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.confusion import ConfusionMatrixAnalyzer
from src.evaluation.profiling import profile_model_hardware
from src.evaluation.robustness import evaluate_robustness
from src.training.callbacks import Callback
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.history import TrainingHistory
from src.training.losses import build_criterion
from src.training.optimizers import build_optimizer
from src.training.schedulers import build_scheduler
from src.utils.device import get_device, synchronize_device
from src.utils.logging import get_logger


class ModelTrainer:
    """Standardized training engine for CNN and Vision Transformer model comparison."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        data_loaders: Dict[str, DataLoader],
        dataset_id: str = "cifar100",
        num_classes: int = 100,
        epochs: int = 100,
        warmup_epochs: int = 5,
        base_lr: float = 5e-4,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.0,
        grad_clip_norm: float = 1.0,
        data_fraction: int = 100,
        seed: int = 42,
        pretrained: bool = False,
        regime: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.data_loaders = data_loaders
        self.dataset_id = dataset_id

        # Infer num_classes from data_loaders or model if available
        if isinstance(data_loaders, dict) and "num_classes" in data_loaders:
            self.num_classes = int(data_loaders["num_classes"])
        elif hasattr(model, "num_classes"):
            self.num_classes = getattr(model, "num_classes")
        else:
            self.num_classes = num_classes

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.grad_clip_norm = grad_clip_norm
        self.data_fraction = data_fraction
        self.seed = seed
        self.pretrained = pretrained
        self.regime = regime or ("imagenet1k_pretrained" if pretrained else "from_scratch")
        self.device = get_device(device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.callbacks = callbacks or []
        self.logger = logger or get_logger("model_trainer")

        self.model = self.model.to(self.device)

        # Output directory resolution
        if output_dir is not None:
            out_p = Path(output_dir)
            if out_p.name.startswith("seed_") or model_name in str(out_p):
                self.output_dir = out_p
            else:
                self.output_dir = out_p / dataset_id / model_name / f"seed_{seed}_frac_{data_fraction}"
        else:
            self.output_dir = Path("result") / dataset_id / model_name / f"seed_{seed}_frac_{data_fraction}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimization suite
        self.criterion = build_criterion("cross_entropy", label_smoothing=label_smoothing)
        self.optimizer = build_optimizer(
            self.model,
            optimizer_type="adamw",
            lr=base_lr,
            weight_decay=weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            base_lr=base_lr,
        )

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = TrainingHistory()
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.stop_training = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute a single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_logits: List[np.ndarray] = []
        all_targets: List[int] = []

        loader = self.data_loaders["train"]
        pbar = tqdm(loader, desc=f"[{self.model_name}|{self.dataset_id}] Ep {epoch+1}/{self.epochs}", leave=False)

        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                images, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, targets = batch["image"], batch["label"]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_logits.append(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy().tolist())

        total_samples = len(all_targets)
        avg_loss = total_loss / max(1, total_samples)
        logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
        metrics = compute_classification_metrics(logits_arr, all_targets, self.num_classes)

        return {
            "loss": round(avg_loss, 4),
            "top1_accuracy": metrics.top1_accuracy,
            "macro_f1": metrics.macro_f1,
            "balanced_accuracy": metrics.balanced_accuracy,
        }

    def evaluate(self, split: str = "val") -> Tuple[Dict[str, Any], np.ndarray, List[int]]:
        """Evaluate model on validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_logits: List[np.ndarray] = []
        all_targets: List[int] = []

        loader = self.data_loaders[split]

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    images, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    images, targets = batch["image"], batch["label"]
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
                else:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

                with autocast_ctx:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item() * images.size(0)
                all_logits.append(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy().tolist())

        total_samples = len(all_targets)
        avg_loss = total_loss / max(1, total_samples)
        logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.array([])

        clf_metrics = compute_classification_metrics(logits_arr, all_targets, self.num_classes)
        cal_metrics = compute_calibration_metrics(logits_arr, all_targets)

        metrics_dict = clf_metrics.to_dict()
        metrics_dict["loss"] = round(avg_loss, 4)
        metrics_dict["ece"] = cal_metrics.ece
        metrics_dict["mce"] = cal_metrics.mce

        return metrics_dict, logits_arr, all_targets

    def fit(self) -> Dict[str, Any]:
        """Execute full model training loop with validation, LR scheduling, and checkpointing."""
        start_time = time.perf_counter()
        self.logger.info(
            f"Starting training: model={self.model_name}, dataset={self.dataset_id}, "
            f"epochs={self.epochs}, seed={self.seed}, fraction={self.data_fraction}%, "
            f"regime={self.regime}, device={self.device.type}"
        )

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(self.epochs):
            if self.stop_training:
                self.logger.info("Early stopping triggered. Halting training.")
                break

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_start = time.perf_counter()

            train_metrics = self.train_epoch(epoch)
            val_metrics, _, _ = self.evaluate(split="val")
            self.scheduler.step()

            duration = time.perf_counter() - epoch_start
            self.history.record_epoch(epoch + 1, current_lr, train_metrics, val_metrics, duration)

            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["top1_accuracy"]

            # Best validation checkpoint selection
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    best_val_metric=self.best_val_f1,
                    model_name=self.model_name,
                    dataset_name=self.dataset_id,
                    seed=self.seed,
                    regime=self.regime,
                    output_path=self.output_dir / "best_model.pt",
                    is_best=True,
                )

            # Save latest checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                best_val_metric=self.best_val_f1,
                model_name=self.model_name,
                dataset_name=self.dataset_id,
                seed=self.seed,
                regime=self.regime,
                output_path=self.output_dir / "last_model.pt",
                is_best=False,
            )

            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs:03d} | Train Loss: {train_metrics['loss']:.4f}, "
                f"F1: {train_metrics['macro_f1']:.2f}% | Val Loss: {val_metrics['loss']:.4f}, "
                f"F1: {val_metrics['macro_f1']:.2f}%, Top1: {val_metrics['top1_accuracy']:.2f}% | "
                f"Best F1: {self.best_val_f1:.2f}% (Epoch {self.best_epoch})"
            )

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch + 1, train_metrics, val_metrics)

        total_duration = time.perf_counter() - start_time
        self.history.save(self.output_dir)

        # Load best weights for test evaluation
        best_pt = self.output_dir / "best_model.pt"
        if best_pt.is_file():
            load_checkpoint(best_pt, self.model, device=self.device)
            self.logger.info(f"Loaded best checkpoint from epoch {self.best_epoch} for test evaluation.")

        # Save config.json
        config_data = {
            "model_name": self.model_name,
            "dataset_id": self.dataset_id,
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "base_lr": self.base_lr,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "grad_clip_norm": self.grad_clip_norm,
            "data_fraction": self.data_fraction,
            "seed": self.seed,
            "pretrained": self.pretrained,
            "regime": self.regime,
            "device": str(self.device),
            "use_amp": self.use_amp,
        }
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Clean Test Evaluation
        test_metrics, test_logits, test_targets = self.evaluate(split="test")

        # Save clean test metrics JSON
        with open(self.output_dir / "clean_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        with open(self.output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        # Confusion Matrix
        test_preds = np.argmax(test_logits, axis=1)
        cm_analyzer = ConfusionMatrixAnalyzer(test_targets, test_preds, self.num_classes)
        cm_analyzer.save_json(self.output_dir / "confusion_matrix.json")
        cm_analyzer.save_csv(self.output_dir / "confusion_matrix.csv")

        # Hardware profiling
        self.logger.info("Profiling model computational complexity and latency...")
        hw_metrics = profile_model_hardware(self.model, device=self.device)
        with open(self.output_dir / "hardware_metrics.json", "w", encoding="utf-8") as f:
            json.dump(hw_metrics, f, indent=2)

        # Robustness evaluation
        robustness_res: Dict[str, Any] = {}
        if "test" in self.data_loaders and hasattr(self.data_loaders["test"].dataset, "samples"):
            self.logger.info(f"Evaluating robustness across synthetic corruptions on '{self.dataset_id}' test set...")
            test_samples = self.data_loaders["test"].dataset.samples
            from src.datasets.transforms import get_eval_transforms
            eval_tf = get_eval_transforms()
            robustness_res = evaluate_robustness(
                model=self.model,
                samples=test_samples,
                eval_transform=eval_tf,
                device=self.device,
                clean_macro_f1=test_metrics["macro_f1"],
                clean_accuracy=test_metrics["top1_accuracy"],
            )
            with open(self.output_dir / "robustness_metrics.json", "w", encoding="utf-8") as f:
                json.dump(robustness_res, f, indent=2)

        # Final Summary
        summary = {
            "model_name": self.model_name,
            "dataset_id": self.dataset_id,
            "num_classes": self.num_classes,
            "data_fraction": self.data_fraction,
            "seed": self.seed,
            "pretrained": self.pretrained,
            "regime": self.regime,
            "best_epoch": self.best_epoch,
            "total_epochs": self.epochs,
            "total_train_time_sec": round(total_duration, 2),
            "best_val_macro_f1": self.best_val_f1,
            "best_val_top1_acc": self.best_val_acc,
            "test_top1_acc": test_metrics.get("top1_accuracy", 0.0),
            "test_macro_f1": test_metrics.get("macro_f1", 0.0),
            "total_params": hw_metrics.get("total_params", 0),
            "clean_test_metrics": test_metrics,
            "hardware_metrics": hw_metrics,
            "robustness_metrics": robustness_res,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.output_dir / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(
            f"Experiment complete! Clean F1: {test_metrics['macro_f1']}%, "
            f"Clean Top-1: {test_metrics['top1_accuracy']}%, Latency: {hw_metrics['latency_p95_ms']} ms | "
            f"Saved: {self.output_dir}"
        )

        return summary
