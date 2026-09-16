"""Multi-Model Multi-Dataset Comparative Experiment Runner."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.aggregation import aggregate_experiment_results, save_comparison_summary
from src.training.train_one_run import run_training_experiment
from src.utils.logging import get_logger, setup_logging


def run_comparison_suite(
    models: List[str],
    datasets: List[str],
    fractions: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    pretrained: bool = False,
    data_dir: str = "DATA",
    results_dir: str = "result/comparison",
    epochs: int = 100,
    quick_test: bool = False,
) -> pd.DataFrame:
    """Execute cross-model cross-dataset comparative benchmark."""
    logger = get_logger("comparison_suite")
    target_fractions = fractions or [100]
    target_seeds = seeds or [42]
    out_root = Path(results_dir)

    for d in datasets:
        for m in models:
            for frac in target_fractions:
                for seed in target_seeds:
                    run_dir = out_root / d / m / f"seed_{seed}_frac_{frac}"
                    if (run_dir / "final_summary.json").is_file() and not quick_test:
                        logger.info(f"Skipping completed condition: {run_dir}")
                        continue

                    logger.info(f"Running: model={m}, dataset={d}, fraction={frac}%, seed={seed}")
                    try:
                        run_training_experiment(
                            model_name=m,
                            dataset_name=d,
                            data_dir=data_dir,
                            data_fraction=frac,
                            seed=seed,
                            epochs=epochs,
                            pretrained=pretrained,
                            output_dir=run_dir,
                            quick_test=quick_test,
                        )
                    except Exception as e:
                        logger.error(f"Failed condition {m} on {d}: {e}")

    df = aggregate_experiment_results(out_root)
    save_comparison_summary(df, out_root / "master_comparison_summary")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Multi-Model Cross-Dataset Benchmark.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--datasets", type=str, default="cifar100", help="Comma-separated dataset names")
    parser.add_argument("--fractions", type=str, default="100", help="Comma-separated fractions")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--pretrained", action="store_true", help="Pretrained ImageNet-1K regime")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Path to data directory")
    parser.add_argument("--results-dir", type=str, default="result/comparison", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--quick-test", action="store_true", help="Quick smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir="logs", log_prefix="comparison_suite")
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
