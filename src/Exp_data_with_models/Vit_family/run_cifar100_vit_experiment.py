"""Compatibility runner for CIFAR-100 Vision Transformer experiments."""

import argparse
from src.experiments.run_cifar100 import run_cifar100_suite
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 Vision Transformer Experiments.")
    parser.add_argument("--model", type=str, default="vit_b16", help="ViT model name (vit_b16, swin_t, deit_s16)")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Path to datasets root")
    parser.add_argument("--fraction", type=int, default=100, help="Data fraction percentage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-1K pretrained weights")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--results-dir", type=str, default="result/cifar100", help="Output directory")
    parser.add_argument("--quick-test", action="store_true", help="Fast smoke run")
    args = parser.parse_args()

    setup_logging(log_dir="logs", log_prefix=f"cifar100_{args.model}")
    run_cifar100_suite(
        models=[args.model],
        fractions=[args.fraction],
        seeds=[args.seed],
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        device=args.device,
        quick_test=args.quick_test,
    )


if __name__ == "__main__":
    main()
