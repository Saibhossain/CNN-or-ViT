#!/usr/bin/env python3
"""Dataset Auditing CLI.

Performs rigorous audit of registered benchmark datasets, computes descriptive metrics,
validates integrity and zero split-leakage, and exports:
- DATA/audit/<dataset_id>.json
- DATA/fingerprints/<dataset_id>.json
- DATA/dataset_catalog.csv
- DATA/dataset_catalog.json

Usage:
    python scripts/audit_datasets.py
    python scripts/audit_datasets.py --dataset cifar100
    python scripts/audit_datasets.py --dataset all
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.audit import DatasetAuditEngine
from src.datasets.registry import DatasetRegistry
from src.utils.logging import get_logger, setup_logging

logger = get_logger("audit_datasets")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit benchmark datasets and generate dataset catalogs and fingerprints."
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
        default="DATA",
        help="Root directory for dataset storage.",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="DATA/audit",
        help="Directory to save audit JSON records.",
    )
    parser.add_argument(
        "--fingerprints-dir",
        type=str,
        default="DATA/fingerprints",
        help="Directory to save dataset fingerprint records.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset ID(s) to audit or 'all'.",
    )
    return parser.parse_args()


def main():
    setup_logging(log_dir="logs", log_file_name="audit_datasets.log")
    args = parse_args()

    registry = DatasetRegistry(config_path=args.config, data_root=args.data_root)

    ds_arg = args.dataset
    if isinstance(ds_arg, list) and len(ds_arg) == 1 and ds_arg[0].lower() == "all":
        target_ids = registry.list_datasets(primary_only=False, enabled_only=False)
    elif isinstance(ds_arg, list):
        target_ids = ds_arg
    else:
        target_ids = [ds_arg]

    print("\n" + "=" * 105)
    print("CNN vs ViT BENCHMARK DATASET AUDIT")
    print(f"Config: {args.config} | Data root: {args.data_root}")
    print(f"Target datasets: {target_ids}")
    print("=" * 105)

    engine = DatasetAuditEngine(
        registry=registry,
        data_dir=args.data_root,
        audit_dir=args.audit_dir,
        fingerprints_dir=args.fingerprints_dir,
    )

    csv_path, json_path = engine.audit_all_and_build_catalog(target_ids=target_ids)

    # Print summary table
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
        audit_file = engine.audit_dir / f"{ds_id}.json"
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
        f"Summary: Total: {len(target_ids)} | Passed: {passed_count} | Warnings: {warning_count} | Failed: {failed_count} | Missing: {missing_count}"
    )
    print(f"Master Catalogs Generated: {csv_path} and {json_path}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
