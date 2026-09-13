#!/usr/bin/env python3
"""Deterministic Subset Manifest Generator.

Generates and saves deterministic, class-stratified cumulative nested training subset manifests
and SHA-256 hashes for all combinations of datasets, random seeds, and data fractions:
  - Data Fractions: [5%, 10%, 25%, 50%, 75%, 100%]
  - Random Seeds: [42, 123, 2024, 3407, 9999]

Usage:
  python scripts/generate_manifests.py --dataset all --seeds all
  python scripts/generate_manifests.py --dataset cifar100 --seed 42
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.subsets import (
    DEFAULT_FRACTIONS,
    DEFAULT_SEEDS,
    generate_stratified_nested_subsets,
    save_subset_manifests,
    verify_subset_nesting,
)
from src.utils.logging import get_logger, setup_logging

logger = get_logger("generate_manifests")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic nested subset manifests with SHA-256 hashes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration YAML.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="DATA/manifests",
        help="Target directory to store manifest CSVs (default: DATA/manifests).",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        default=["all"],
        help="Specific dataset ID(s) or 'all'.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=int,
        default=DEFAULT_FRACTIONS,
        help="Data fractions in percentage (e.g. 5 10 25 50 75 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single random seed to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["all"],
        help="List of seeds or 'all' (default: 42 123 2024 3407 9999).",
    )
    return parser.parse_args()


def main():
    setup_logging(log_dir="logs", log_file_name="generate_manifests.log")
    args = parse_args()

    registry = DatasetRegistry(config_path=args.config)

    # Determine target datasets
    ds_arg = args.dataset
    if isinstance(ds_arg, list) and len(ds_arg) == 1 and ds_arg[0].lower() == "all":
        target_ids = registry.list_datasets(primary_only=False, enabled_only=True)
    elif isinstance(ds_arg, list):
        target_ids = ds_arg
    else:
        target_ids = [ds_arg]

    # Determine target seeds
    if args.seed is not None:
        target_seeds = [args.seed]
    elif isinstance(args.seeds, list) and len(args.seeds) == 1 and args.seeds[0].lower() == "all":
        target_seeds = DEFAULT_SEEDS
    elif isinstance(args.seeds, list):
        target_seeds = [int(s) for s in args.seeds]
    else:
        target_seeds = DEFAULT_SEEDS

    logger.info(f"Target datasets: {target_ids}")
    logger.info(f"Target fractions: {args.fractions}")
    logger.info(f"Target seeds: {target_seeds}")

    total_manifests_created = 0

    for ds_id in target_ids:
        try:
            adapter = registry.get_adapter(ds_id)
        except KeyError:
            logger.error(f"Dataset '{ds_id}' not found in registry.")
            continue

        if not adapter.is_available():
            logger.warning(
                f"Skipping '{ds_id}': local directory not found or empty at {adapter.resolved_path.resolve()}"
            )
            continue

        logger.info(f"Generating manifests for dataset: {ds_id}")
        train_samples = adapter.get_samples("train")
        logger.info(f"  Training partition size: {len(train_samples)} samples")

        for seed in target_seeds:
            subsets = generate_stratified_nested_subsets(
                train_samples=train_samples,
                fractions=args.fractions,
                seed=seed,
            )

            # Verification of nested hierarchy
            is_valid, errors = verify_subset_nesting(subsets)
            if not is_valid:
                logger.error(f"Nesting check failed for {ds_id} (seed {seed}): {errors}")
                sys.exit(1)

            saved_paths = save_subset_manifests(
                subsets_by_fraction=subsets,
                dataset_id=ds_id,
                seed=seed,
                manifest_dir=args.manifest_dir,
            )
            total_manifests_created += len(saved_paths)
            logger.info(
                f"  [Seed {seed:>5}] Generated {len(saved_paths)} fraction manifests -> {list(saved_paths.values())[0].parent}"
            )

    logger.info(f"Manifest generation complete. Total manifests saved: {total_manifests_created}")


if __name__ == "__main__":
    main()
