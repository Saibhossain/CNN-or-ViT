#!/usr/bin/env python3
"""Dataset Statistics Reporter.

Calculates and exports comprehensive statistical profiles for all registered benchmark datasets:
- Train/Validation/Test sample sizes
- Number of classes and domain categorization
- Class imbalance metrics (Max/Min ratio, Gini coefficient, Shannon entropy)
- Native resolution distributions
- Generates Table 1 (Dataset Characteristics)
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.statistics import DatasetStatisticsCalculator
from src.utils.logging import setup_logging, get_logger

logger = get_logger("dataset_stats")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and export dataset statistical characteristics (Table 1)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables",
        help="Directory to save output tables.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        default=["all"],
        help="Specific dataset IDs or 'all' to process.",
    )
    return parser.parse_args()


def main():
    setup_logging(log_dir="logs", log_file_name="dataset_stats.log")
    args = parse_args()

    registry = DatasetRegistry(config_path=args.config)
    if "all" in args.dataset:
        target_ids = registry.list_datasets(primary_only=False, enabled_only=False)
    else:
        target_ids = args.dataset

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_stats = []
    table_rows = []

    print("\n" + "=" * 105)
    print(
        f"{'DATASET':<16} | {'DOMAIN':<24} | {'CLASSES':<7} | {'TRAIN':<7} | {'VAL':<6} | {'TEST':<6} | {'GINI':<6} | {'RATIO':<6} | {'STATUS'}"
    )
    print("=" * 105)

    for ds_id in target_ids:
        try:
            adapter = registry.get_adapter(ds_id)
            meta = adapter.metadata
        except KeyError:
            continue

        if not adapter.is_available():
            row_str = f"{meta.dataset_name:<16} | {meta.domain:<24} | {meta.num_classes:<7} | {'-':<7} | {'-':<6} | {'-':<6} | {'-':<6} | {'-':<6} | MISSING ON DISK"
            print(row_str)
            table_rows.append({
                "name": meta.dataset_name,
                "id": ds_id,
                "domain": meta.domain,
                "classes": meta.num_classes,
                "train": "N/A (Pending Download)",
                "val": "N/A (Pending Download)",
                "test": "N/A (Pending Download)",
                "gini": "N/A",
                "imbalance_ratio": "N/A",
                "resolution": f"{meta.image_size[0]}x{meta.image_size[1]}",
                "license": meta.license,
                "status": "Missing local files",
            })
            continue

        calc = DatasetStatisticsCalculator(adapter)
        stats = calc.compute_statistics()
        all_stats.append(stats)

        imb = stats["class_imbalance"]
        res = stats["native_resolution"]
        print(
            f"{meta.dataset_name:<16} | {meta.domain:<24} | {meta.num_classes:<7} | {stats['train_samples']:<7} | {stats['val_samples']:<6} | {stats['test_samples']:<6} | {imb['gini_coefficient']:<6.3f} | {str(imb['imbalance_ratio_max_to_min']):<6} | AVAILABLE"
        )
        table_rows.append({
            "name": meta.dataset_name,
            "id": ds_id,
            "domain": meta.domain,
            "classes": meta.num_classes,
            "train": stats["train_samples"],
            "val": stats["val_samples"],
            "test": stats["test_samples"],
            "gini": imb["gini_coefficient"],
            "imbalance_ratio": imb["imbalance_ratio_max_to_min"],
            "resolution": f"{res['mean_width']}x{res['mean_height']}",
            "license": meta.license,
            "status": "Verified on disk",
        })

    print("=" * 105 + "\n")

    # Save JSON summary
    json_path = out_path / "dataset_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Saved dataset statistics JSON to {json_path}")

    # Also save to DATA/dataset_statistics.json
    data_json_path = Path("DATA/dataset_statistics.json")
    if data_json_path.parent.is_dir():
        with open(data_json_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"Saved dataset statistics JSON to {data_json_path}")

    # Generate Markdown Table 1
    md_path = out_path / "table1_dataset_characteristics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 1: Benchmark Dataset Characteristics\n\n")
        f.write("| Dataset | Domain | Classes | Train | Val | Test | Gini Coeff | Max/Min Ratio | Native Res | License | Status |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|\n")
        for r in table_rows:
            f.write(
                f"| **{r['name']}** | {r['domain']} | {r['classes']} | {r['train']} | {r['val']} | {r['test']} | {r['gini']} | {r['imbalance_ratio']} | {r['resolution']} | {r['license']} | {r['status']} |\n"
            )
    logger.info(f"Saved Table 1 markdown to {md_path}")


if __name__ == "__main__":
    main()
