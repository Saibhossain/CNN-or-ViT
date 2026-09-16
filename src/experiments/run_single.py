"""Single Experiment Runner CLI and Python API."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.train_one_run import run_training_experiment
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single CNN or ViT training experiment.")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name (e.g. resnet50, vit_b16, swin_t)")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset identifier (e.g. cifar100, dtd, eurosat)")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Path to datasets root")
    parser.add_argument("--fraction", type=int, default=100, choices=[5, 10, 25, 50, 75, 100], help="Training data percentage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--pretrained", action="store_true", help="Load ImageNet-1K pretrained weights")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--quick-test", action="store_true", help="Fast smoke run (1 epoch)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir="logs", log_prefix=f"{args.dataset}_{args.model}")
    run_training_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        data_fraction=args.fraction,
        seed=args.seed,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        device=args.device,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
    )


if __name__ == "__main__":
    main()
