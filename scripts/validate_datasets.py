#!/usr/bin/env python3
"""Dataset Validation and Audit CLI.

Validates presence, split integrity, zero data-leakage, label validity, image loadability,
compares expected vs observed metadata, and compiles master dataset catalogs:
  - DATA/dataset_catalog.csv
  - DATA/dataset_catalog.json
  - DATA/audit/<dataset_id>.json

Usage:
  python scripts/validate_datasets.py --dataset all
  python scripts/validate_datasets.py --dataset cifar100
  python scripts/validate_datasets.py --dataset all --strict
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.audit import DatasetAuditEngine
from src.datasets.registry import DatasetRegistry
from src.datasets.validator import DatasetValidator, DatasetValidationError
from src.utils.logging import get_logger, setup_logging

logger = get_logger("validate_datasets")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate benchmark datasets for integrity, zero-leakage, and generate catalog audits."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration YAML.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional root directory override for datasets.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        default=["all"],
        help="Specific dataset ID(s) or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict mode: treat any warnings as validation failures.",
    )
    parser.add_argument(
        "--include-secondary",
        "--include-extensions",
        action="store_true",
        help="Include secondary and extension datasets.",
    )
    return parser.parse_args()


def main():
    setup_logging(log_dir="logs", log_file_name="dataset_validation.log")
    args = parse_args()

    registry = DatasetRegistry(config_path=args.config, data_root=args.data_root)

    dataset_arg = args.dataset
    if isinstance(dataset_arg, list) and len(dataset_arg) == 1 and dataset_arg[0].lower() == "all":
        target_ids = registry.list_datasets(primary_only=False, enabled_only=False)
    elif isinstance(dataset_arg, list):
        target_ids = dataset_arg
    else:
        target_ids = [dataset_arg]

    logger.info(f"Target datasets to validate and audit: {target_ids}")

    audit_engine = DatasetAuditEngine(registry=registry)
    csv_path, json_path = audit_engine.audit_all_and_build_catalog(target_ids=target_ids)

    # Print summary console table
    print("\n" + "=" * 105)
    print(
        f"{'DATASET ID':<18} | {'STATUS':<9} | {'TRAIN':<7} | {'VAL':<6} | {'TEST':<6} | {'CLASSES':<7} | {'FINGERPRINT':<16} | {'LOCAL PATH'}"
    )
    print("=" * 105)

    passed_count = 0
    warning_count = 0
    failed_count = 0
    missing_count = 0

    for ds_id in target_ids:
        audit_file = audit_engine.audit_dir / f"{ds_id}.json"
        if audit_file.is_file():
            import json
            with open(audit_file, "r", encoding="utf-8") as f:
                rec = json.load(f)

            status = rec["status"]
            struct = rec["structure"]
            fp = rec["reproducibility"]["dataset_fingerprint"]
            fp_short = fp[:14] + ".." if fp else "N/A"
            path_str = rec["dataset_identity"]["local_path"]

            if not rec["is_available"]:
                missing_count += 1
                status_display = "MISSING"
            elif status == "PASS":
                passed_count += 1
                status_display = "PASS"
            elif status == "WARNING":
                warning_count += 1
                status_display = "WARNING"
            else:
                failed_count += 1
                status_display = "FAIL"

            print(
                f"{ds_id:<18} | {status_display:<9} | {str(struct['train_size']):<7} | {str(struct['validation_size']):<6} | {str(struct['test_size']):<6} | {str(struct['num_classes']):<7} | {fp_short:<16} | {path_str}"
            )

    print("=" * 105)
    print(
        f"Summary: Total: {len(target_ids)} | Passed: {passed_count} | Warnings: {warning_count} | Failed: {failed_count} | Missing on disk: {missing_count}"
    )
    print(f"Master Catalogs Generated: {csv_path} and {json_path}")
    print("=" * 105 + "\n")

    if args.strict and (failed_count > 0 or warning_count > 0):
        logger.error("Strict validation mode failed due to errors or warnings.")
        sys.exit(1)
    elif failed_count > 0:
        logger.warning(f"Validation completed with {failed_count} failures.")
        sys.exit(1)


if __name__ == "__main__":
    main()
