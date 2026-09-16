"""All-in-One Master Benchmark Runner for CNN vs. ViT Empirical Comparison.

This script executes:
1. PHASE 1: Pre-flight smoke validation on a small data fraction to verify the entire pipeline (models, metrics, calibration, robustness, profiling, checkpointing).
2. PHASE 2: Full-scale comparative training across CNN and ViT models on CIFAR-100 (or custom datasets).
3. PHASE 3: Consolidated multi-metric aggregation and tabular summary generation (CSV/JSON).
"""

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
from src.smoke_test import run_all_smoke_tests
from src.training.train_one_run import run_training_experiment
from src.utils.logging import get_logger, setup_logging


CANONICAL_CNN_MODELS: List[str] = ["resnet50", "densenet121", "efficientnetv2_s", "convnext_tiny"]
CANONICAL_VIT_MODELS: List[str] = ["vit_b16", "swin_t"]


def execute_full_benchmark(
    models: Optional[List[str]] = None,
    dataset_name: str = "cifar100",
    fractions: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    pretrained: bool = False,
    data_dir: str = "DATA",
    results_dir: Optional[str] = None,
    epochs: int = 100,
    warmup_epochs: int = 5,
    batch_size: int = 64,
    device: Optional[str] = None,
    verify_first: bool = True,
    quick_test: bool = False,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Execute complete end-to-end benchmark with automated pre-flight verification."""
    logger = get_logger("full_benchmark")
    out_root = Path(results_dir or f"result/{dataset_name}")
    out_root.mkdir(parents=True, exist_ok=True)

    target_models = models or (CANONICAL_CNN_MODELS + CANONICAL_VIT_MODELS)
    target_fractions = fractions or [100]
    target_seeds = seeds or [42]

    logger.info("=" * 80)
    logger.info(f"STARTING CNN vs. ViT BENCHMARK SUITE ON DATASET: {dataset_name.upper()}")
    logger.info(f"Models ({len(target_models)}): {', '.join(target_models)}")
    logger.info(f"Fractions: {target_fractions}% | Seeds: {target_seeds} | Regime: {'pretrained' if pretrained else 'scratch'}")
    logger.info("=" * 80)

    # -------------------------------------------------------------
    # PHASE 1: PRE-FLIGHT VALIDATION
    # -------------------------------------------------------------
    if verify_first and not quick_test:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1/3: PRE-FLIGHT VALIDATION (Smoke testing pipeline on small data)")
        logger.info("=" * 80)
        
        smoke_res = run_all_smoke_tests()
        if smoke_res.get("status") != "ALL_PASSED":
            logger.error("\n[FAIL] Pre-flight pipeline verification failed! Halting benchmark before full training.")
            return pd.DataFrame()

        logger.info("\n[OK] Pre-flight validation passed (100%)! All architectures, metrics, and loaders verified.\n")

    # -------------------------------------------------------------
    # PHASE 2: FULL EXPERIMENT MATRIX EXECUTION
    # -------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 2/3: FULL BENCHMARK TRAINING & MULTI-METRIC EVALUATION")
    logger.info("=" * 80)

    total_runs = len(target_models) * len(target_fractions) * len(target_seeds)
    run_idx = 0

    for m in target_models:
        for frac in target_fractions:
            for seed in target_seeds:
                run_idx += 1
                run_dir = out_root / m / f"seed_{seed}_frac_{frac}"
                summary_file = run_dir / "final_summary.json"

                if not force_rerun and summary_file.is_file() and not quick_test:
                    logger.info(f"[{run_idx}/{total_runs}] Skipping completed condition: {run_dir.name} for {m}")
                    continue

                logger.info(
                    f"\n[{run_idx}/{total_runs}] Running: model={m}, dataset={dataset_name}, "
                    f"fraction={frac}%, seed={seed}, epochs={1 if quick_test else epochs}"
                )

                try:
                    run_training_experiment(
                        model_name=m,
                        dataset_name=dataset_name,
                        data_dir=data_dir,
                        data_fraction=frac,
                        seed=seed,
                        epochs=epochs,
                        warmup_epochs=warmup_epochs,
                        batch_size=batch_size,
                        pretrained=pretrained,
                        device=device,
                        output_dir=run_dir,
                        quick_test=quick_test,
                    )
                except Exception as e:
                    logger.error(f"Execution failed for model {m} (fraction {frac}%, seed {seed}): {e}")

    # -------------------------------------------------------------
    # PHASE 3: CONSOLIDATION & COMPARISON REPORTING
    # -------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3/3: CONSOLIDATING METRICS & GENERATING COMPARISON SUMMARY")
    logger.info("=" * 80)

    df_summary = aggregate_experiment_results(out_root, dataset_name=dataset_name)
    if not df_summary.empty:
        summary_base = out_root / f"{dataset_name}_master_comparison_summary"
        save_comparison_summary(df_summary, summary_base)
        logger.info(f"[OK] Master comparison table saved: {summary_base}.csv and {summary_base}.json")
        logger.info(f"\nConsolidated Comparison Table Preview:\n{df_summary[['model_name', 'data_fraction', 'test_top1_acc', 'test_macro_f1', 'ece', 'latency_p95_ms', 'total_params']].to_string(index=False)}")
    else:
        logger.warning("No completed experiment summaries found to consolidate.")

    return df_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN vs. ViT All-in-One Benchmark Runner.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models list (e.g. resnet50,convnext_tiny,vit_b16,swin_t)")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name (e.g. cifar100, dtd, eurosat)")
    parser.add_argument("--fractions", type=str, default="100", help="Comma-separated training fractions (e.g. 5,10,25,50,75,100)")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds (e.g. 42,123,999)")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-1K pretrained weights")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Path to data root directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Path to results directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per condition")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--quick-test", action="store_true", help="Run fast 1-epoch smoke test across models")
    parser.add_argument("--skip-verify", action="store_false", dest="verify_first", help="Skip Phase 1 pre-flight verification")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun even if results exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir="logs", log_prefix=f"benchmark_{args.dataset}")

    model_list = [m.strip() for m in args.models.split(",")] if args.models else None
    fraction_list = [int(f.strip()) for f in args.fractions.split(",")]
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    execute_full_benchmark(
        models=model_list,
        dataset_name=args.dataset,
        fractions=fraction_list,
        seeds=seed_list,
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        device=args.device,
        verify_first=args.verify_first,
        quick_test=args.quick_test,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
