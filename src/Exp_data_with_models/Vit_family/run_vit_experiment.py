"""Compatibility runner for multi-dataset Vision Transformer experiments."""

import argparse
from src.experiments.run_comparison import run_comparison_suite
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Multi-Dataset Vision Transformer Experiments.")
    parser.add_argument("--models", type=str, default="vit_b16,swin_t,deit_s16")
    parser.add_argument("--datasets", type=str, default="cifar100")
    parser.add_argument("--fractions", type=str, default="100")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--data-dir", type=str, default="DATA")
    parser.add_argument("--results-dir", type=str, default="result")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    setup_logging(log_dir="logs", log_prefix="vit_experiment")
    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    fractions = [int(f.strip()) for f in args.fractions.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_comparison_suite(
        models=models,
        datasets=datasets,
        fractions=fractions,
        seeds=seeds,
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        quick_test=args.quick_test,
    )


if __name__ == "__main__":
    main()
