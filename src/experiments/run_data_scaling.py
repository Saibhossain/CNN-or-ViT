"""Data Scaling Experiment Suite (5%, 10%, 25%, 50%, 75%, 100%) for Crossover Estimation."""

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

DATA_FRACTIONS: List[int] = [5, 10, 25, 50, 75, 100]


def run_data_scaling_suite(
    models: List[str],
    dataset_name: str = "cifar100",
    fractions: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    pretrained: bool = False,
    data_dir: str = "DATA",
    results_dir: str = "result/data_scaling",
    epochs: int = 100,
    quick_test: bool = False,
) -> pd.DataFrame:
    """Execute data scaling experiments across 6 fractions."""
    logger = get_logger("data_scaling_suite")
    target_fractions = fractions or DATA_FRACTIONS
    target_seeds = seeds or [42]
    out_root = Path(results_dir) / dataset_name

    for m in models:
        for frac in target_fractions:
            for seed in target_seeds:
                run_dir = out_root / m / f"seed_{seed}_frac_{frac}"
                if (run_dir / "final_summary.json").is_file() and not quick_test:
                    logger.info(f"Skipping completed scaling point: {run_dir}")
                    continue

                logger.info(f"Running Data Scale: model={m}, dataset={dataset_name}, fraction={frac}%, seed={seed}")
                run_training_experiment(
                    model_name=m,
                    dataset_name=dataset_name,
                    data_dir=data_dir,
                    data_fraction=frac,
                    seed=seed,
                    epochs=epochs,
                    pretrained=pretrained,
                    output_dir=run_dir,
                    quick_test=quick_test,
                )

    df = aggregate_experiment_results(out_root, dataset_name=dataset_name)
    save_comparison_summary(df, out_root / f"{dataset_name}_scaling_summary")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Data Scaling Benchmark.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated models list")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset identifier")
    parser.add_argument("--fractions", type=str, default="5,10,25,50,75,100", help="Fractions list")
    parser.add_argument("--seeds", type=str, default="42", help="Random seeds list")
    parser.add_argument("--pretrained", action="store_true", help="Pretrained regime")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Path to data")
    parser.add_argument("--results-dir", type=str, default="result/data_scaling", help="Results root")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--quick-test", action="store_true", help="Fast smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir="logs", log_prefix=f"data_scaling_{args.dataset}")
    models = [m.strip() for m in args.models.split(",")]
    fractions = [int(f.strip()) for f in args.fractions.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_data_scaling_suite(
        models=models,
        dataset_name=args.dataset,
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
