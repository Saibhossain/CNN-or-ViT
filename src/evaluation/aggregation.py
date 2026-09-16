"""Result aggregation and cross-model comparison utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def aggregate_experiment_results(
    results_dir: Union[str, Path],
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Scan and aggregate all finished run final_summary.json files into a master DataFrame.

    Args:
        results_dir: Directory containing experiment runs.
        dataset_name: Optional filter for specific dataset.

    Returns:
        Aggregated pandas DataFrame with unified columns.
    """
    root_p = Path(results_dir)
    records: List[Dict[str, Any]] = []

    for summary_file in root_p.glob("**/final_summary.json"):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            rec = {
                "model_name": data.get("model_name"),
                "dataset_id": data.get("dataset_id"),
                "seed": data.get("seed"),
                "data_fraction": data.get("data_fraction"),
                "pretrained": data.get("pretrained", False),
                "regime": data.get("regime", "from_scratch"),
                "top1_accuracy": data.get("clean_test_metrics", {}).get("top1_accuracy"),
                "top5_accuracy": data.get("clean_test_metrics", {}).get("top5_accuracy"),
                "macro_f1": data.get("clean_test_metrics", {}).get("macro_f1"),
                "balanced_accuracy": data.get("clean_test_metrics", {}).get("balanced_accuracy"),
                "cohen_kappa": data.get("clean_test_metrics", {}).get("cohen_kappa"),
                "ece": data.get("clean_test_metrics", {}).get("ece"),
                "brier_score": data.get("clean_test_metrics", {}).get("brier_score"),
                "flops_g": data.get("hardware_metrics", {}).get("flops_g"),
                "total_params": data.get("hardware_metrics", {}).get("total_params"),
                "throughput_img_per_sec": data.get("hardware_metrics", {}).get("throughput_img_per_sec"),
                "latency_p95_ms": data.get("hardware_metrics", {}).get("latency_p95_ms"),
                "mean_corrupted_f1": data.get("robustness_metrics", {}).get("mean_corrupted_macro_f1"),
                "relative_degradation_f1": data.get("robustness_metrics", {}).get("relative_degradation_f1"),
                "run_dir": str(summary_file.parent),
            }
            if dataset_name is None or rec["dataset_id"] == dataset_name:
                records.append(rec)
        except Exception:
            pass

    df = pd.DataFrame(records)
    return df


def save_comparison_summary(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """Save aggregated comparison table to CSV and JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p.with_suffix(".csv"), index=False)
    with open(out_p.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
