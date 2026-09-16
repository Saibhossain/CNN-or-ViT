"""CIFAR-100 Master Experiment Runner for CNN vs. Vision Transformer comparison."""

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


DEFAULT_CNN_MODELS: List[str] = ["resnet50", "densenet121", "efficientnetv2_s", "convnext_tiny", "resnet18"]
DEFAULT_VIT_MODELS: List[str] = ["vit_b16", "swin_t", "deit_s16"]


def verify_pipeline_first(data_dir: str = "DATA", device: Optional[str] = None) -> bool:
    """Run lightweight pre-flight smoke check on small subset before full benchmark."""
    logger = get_logger("cifar100_preflight")
    logger.info("=" * 70)
    logger.info("PHASE 1: PRE-FLIGHT VALIDATION (Verifying pipeline on small data subset)")
    logger.info("=" * 70)
    try:
        from src.smoke_test import run_all_smoke_tests
        smoke_res = run_all_smoke_tests()
        if smoke_res.get("status") != "ALL_PASSED":
            logger.error("Pre-flight internal validation failed!")
            return False
        logger.info("=" * 70)
        logger.info("PHASE 1 PASSED: Pipeline verified! Starting Phase 2 (Full Training)...")
        logger.info("=" * 70)
        return True
    except Exception as e:
        logger.error(f"Pre-flight verification encountered error: {e}")
        return False


def run_cifar100_suite(
    models: Optional[List[str]] = None,
    fractions: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    pretrained: bool = False,
    data_dir: str = "DATA",
    results_dir: str = "result/cifar100",
    epochs: int = 100,
    warmup_epochs: int = 5,
    batch_size: int = 64,
    device: Optional[str] = None,
    quick_test: bool = False,
    skip_completed: bool = True,
    verify_first: bool = False,
) -> pd.DataFrame:
    """Execute CIFAR-100 benchmark across architectures, fractions, and seeds.

    Args:
        models: List of model names (defaults to all CNNs and ViTs).
        fractions: Data scaling fractions (defaults to [100]).
        seeds: Random seeds (defaults to [42]).
        pretrained: If True, fine-tune from ImageNet-1K.
        data_dir: Path to dataset root.
        results_dir: Path to output directory.
        epochs: Training epochs.
        warmup_epochs: Warmup epochs.
        batch_size: Batch size.
        device: PyTorch device.
        quick_test: If True, fast 1-epoch smoke run.
        skip_completed: If True, skips previously completed runs.
        verify_first: If True, executes pre-flight validation before full training.

    Returns:
        DataFrame containing consolidated comparison metrics.
    """
    logger = get_logger("cifar100_suite")

    if verify_first and not quick_test:
        verified = verify_pipeline_first(data_dir=data_dir, device=device)
        if not verified:
            logger.error("Aborting full benchmark because pre-flight verification failed.")
            return pd.DataFrame()

    target_models = models or (DEFAULT_CNN_MODELS + DEFAULT_VIT_MODELS)
    target_fractions = fractions or [100]
    target_seeds = seeds or [42]

    out_root = Path(results_dir)

    for m in target_models:
        for frac in target_fractions:
            for seed in target_seeds:
                run_dir = out_root / m / f"seed_{seed}_frac_{frac}"
                summary_file = run_dir / "final_summary.json"

                if skip_completed and summary_file.is_file() and not quick_test:
                    logger.info(f"Skipping already completed run: {run_dir}")
                    continue

                logger.info(f"\n{'='*70}\nExecuting CIFAR-100: model={m}, fraction={frac}%, seed={seed}\n{'='*70}")
                try:
                    run_training_experiment(
                        model_name=m,
                        dataset_name="cifar100",
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
                    logger.error(f"Run failed for {m} (fraction {frac}%, seed {seed}): {e}")

    # Aggregate results into comparison summary
    df_summary = aggregate_experiment_results(out_root, dataset_name="cifar100")
    if not df_summary.empty:
        save_comparison_summary(df_summary, out_root / "cifar100_comparison_summary")
        logger.info(f"Updated CIFAR-100 comparison summary with {len(df_summary)} completed runs.")

    return df_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 CNN vs. ViT Benchmark Suite.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models list")
    parser.add_argument("--fractions", type=str, default="100", help="Comma-separated fractions (e.g. 5,10,25,50,75,100)")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-1K pretrained weights")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Path to datasets root")
    parser.add_argument("--results-dir", type=str, default="result/cifar100", help="Output results directory")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    parser.add_argument("--quick-test", action="store_true", help="Fast 1-epoch smoke test")
    parser.add_argument("--force-rerun", action="store_true", help="Overwrite existing completed runs")
    parser.add_argument("--verify-first", action="store_true", default=False, help="Run pre-flight small-data verification before full training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir="logs", log_prefix="cifar100_suite")

    model_list = [m.strip() for m in args.models.split(",")] if args.models else None
    fraction_list = [int(f.strip()) for f in args.fractions.split(",")]
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    run_cifar100_suite(
        models=model_list,
        fractions=fraction_list,
        seeds=seed_list,
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        device=args.device,
        quick_test=args.quick_test,
        skip_completed=not args.force_rerun,
        verify_first=args.verify_first,
    )


if __name__ == "__main__":
    main()
