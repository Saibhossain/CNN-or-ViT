#!/usr/bin/env python3
"""Dataset Downloader and Local Image Cache Manager.

Downloads benchmark datasets from Hugging Face Hub, extracts and saves actual physical image
files (.png / .jpg) organized into standard ImageFolder splits under DATA/<dataset_id>/,
and records structured failure reports to DATA/failures/<dataset_id>.json upon error.

Directory Structure Created:
    DATA/<dataset_id>/
    ├── dataset_info.json
    ├── train/
    │   ├── <class_0>/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── <class_1>/...
    ├── val/ (or validation/)
    └── test/

Usage:
    python scripts/download_datasets.py --dataset cifar100
    python scripts/download_datasets.py --dataset all
    python scripts/download_datasets.py --dataset eurosat --force
    python scripts/download_datasets.py --dataset flowers102 --image-format jpg
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.datasets.registry import DatasetRegistry
# pyrefly: ignore [missing-import]
from src.datasets.base import DatasetMetadata
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger, setup_logging

# Hugging Face datasets library
try:
    from datasets import load_dataset, Dataset, DatasetDict
    from datasets.features import ClassLabel, Image as HFImage
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = get_logger("download_datasets")


def sanitize_name(name: str) -> str:
    """Sanitize class/split name for filesystem safety across OS platforms."""
    clean = re.sub(r'[\\/*?:"<>|]', '_', str(name))
    clean = clean.strip()
    return clean if clean else "unknown"


def log_dataset_failure(
    dataset_id: str,
    metadata: Optional[DatasetMetadata],
    exception: Exception,
    attempted_hf_ids: List[str],
    failures_dir: Path,
    suggested_action: Optional[str] = None,
) -> Path:
    """Record detailed failure report to DATA/failures/<dataset_id>.json.

    Args:
        dataset_id: Identifier of the failed dataset.
        metadata: DatasetMetadata instance if available.
        exception: The raw Exception object.
        attempted_hf_ids: List of Hugging Face IDs attempted.
        failures_dir: Directory where failure JSONs are stored.
        suggested_action: Recommended remediation step.

    Returns:
        Path to the saved failure JSON.
    """
    failures_dir.mkdir(parents=True, exist_ok=True)
    fail_path = failures_dir / f"{dataset_id}.json"

    tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

    if suggested_action is None:
        if "DatasetNotFound" in type(exception).__name__ or "404" in str(exception):
            suggested_action = (
                "Verify the dataset identifier on Hugging Face Hub or update 'hf_id' / 'alternate_hf_ids' "
                "in configs/datasets.yaml."
            )
        elif "Dataset scripts are no longer supported" in str(exception) or "RuntimeError" in type(exception).__name__:
            suggested_action = (
                "The upstream dataset uses legacy dataset scripts. Locate an updated Parquet-based mirror or "
                "download the raw image archive and use GenericImageFolderAdapter."
            )
        else:
            suggested_action = "Inspect the exception details and verify network connectivity / repository permissions."

    payload = {
        "dataset": dataset_id,
        "dataset_name": metadata.dataset_name if metadata else dataset_id,
        "hf_id": metadata.hf_id if metadata else None,
        "attempted_hf_ids": attempted_hf_ids,
        "status": "FAILED",
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": tb_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suggested_next_action": suggested_action,
    }

    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.error(f"Saved failure report: {fail_path}")
    return fail_path


def discover_columns(split_dataset: Any) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Discover the image column, label column, and class names list from a Hugging Face Dataset split.

    Args:
        split_dataset: A datasets.Dataset instance.

    Returns:
        Tuple of (image_column_name, label_column_name, class_names_list).
    """
    features = getattr(split_dataset, "features", {})
    col_names = getattr(split_dataset, "column_names", [])

    # 1. Identify image column
    image_col = None
    for candidate in ["image", "img", "pixel_values", "bytes", "raw_image"]:
        if candidate in col_names:
            image_col = candidate
            break

    if image_col is None:
        for col, feat in features.items():
            if type(feat).__name__ == "Image" or isinstance(feat, HFImage):
                image_col = col
                break

    # 2. Identify label column
    label_col = None
    for candidate in ["fine_label", "label", "category", "target", "class", "labels", "label_cat_dog"]:
        if candidate in col_names:
            label_col = candidate
            break

    if label_col is None:
        for col, feat in features.items():
            if type(feat).__name__ == "ClassLabel" or isinstance(feat, ClassLabel):
                label_col = col
                break

    # 3. Discover class names
    class_names: List[str] = []
    if label_col and label_col in features:
        feat = features[label_col]
        if hasattr(feat, "names") and feat.names:
            class_names = [str(n) for n in feat.names]

    return image_col, label_col, class_names


def extract_and_save_images(
    hf_dataset: Union[DatasetDict, Dataset],
    output_dir: Path,
    dataset_id: str,
    image_format: str = "png",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract PIL images from Hugging Face dataset and save to standard ImageFolder layout.

    Args:
        hf_dataset: Loaded Hugging Face Dataset or DatasetDict.
        output_dir: Output base directory (e.g. DATA/<dataset_id>).
        dataset_id: Identifier of the dataset.
        image_format: Image file extension / format ('png' or 'jpg').
        max_samples: Optional sample cap per split for testing.

    Returns:
        Dictionary of dataset metadata and split statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(hf_dataset, Dataset):
        splits_dict = {"train": hf_dataset}
    else:
        splits_dict = dict(hf_dataset)

    split_counts: Dict[str, int] = {}
    discovered_class_names: List[str] = []

    # First pass: find global class names across all splits
    for s_name, s_data in splits_dict.items():
        _, _, c_names = discover_columns(s_data)
        if c_names and len(c_names) > len(discovered_class_names):
            discovered_class_names = c_names

    ext = f".{image_format.lower().lstrip('.')}"
    total_images_saved = 0

    for split_name, split_data in splits_dict.items():
        img_col, lbl_col, split_classes = discover_columns(split_data)
        if not discovered_class_names and split_classes:
            discovered_class_names = split_classes

        if img_col is None:
            logger.warning(f"Could not find an image column in split '{split_name}' for '{dataset_id}'. Available: {split_data.column_names}")
            continue

        num_examples = len(split_data)
        if max_samples is not None:
            num_examples = min(num_examples, max_samples)

        logger.info(f"Extracting {num_examples} images for split '{split_name}' in '{dataset_id}'...")

        # Standardize split directory name
        dir_split = split_name
        if split_name == "validation":
            dir_split = "val"

        split_dir = output_dir / dir_split
        split_dir.mkdir(parents=True, exist_ok=True)

        iterator = range(num_examples)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc=f"[{dataset_id}] {split_name}", unit="img")

        saved_in_split = 0
        for idx in iterator:
            item = split_data[idx]
            img_val = item.get(img_col)

            # Resolve label and class name
            if lbl_col and lbl_col in item:
                lbl_val = item[lbl_col]
                if isinstance(lbl_val, int):
                    if 0 <= lbl_val < len(discovered_class_names):
                        class_name = discovered_class_names[lbl_val]
                    else:
                        class_name = f"class_{lbl_val:03d}"
                elif isinstance(lbl_val, str):
                    class_name = lbl_val
                else:
                    class_name = str(lbl_val)
            else:
                class_name = "unlabeled"

            clean_cls = sanitize_name(class_name)
            cls_dir = split_dir / clean_cls
            cls_dir.mkdir(parents=True, exist_ok=True)

            img_path = cls_dir / f"{idx:06d}{ext}"

            # Convert to PIL Image and save
            if img_val is not None:
                if isinstance(img_val, Image.Image):
                    pil_img = img_val
                elif isinstance(img_val, dict) and "bytes" in img_val and img_val["bytes"]:
                    import io
                    pil_img = Image.open(io.BytesIO(img_val["bytes"]))
                elif isinstance(img_val, (str, Path)) and Path(img_val).is_file():
                    pil_img = Image.open(img_val)
                else:
                    try:
                        import numpy as np
                        arr = np.array(img_val)
                        pil_img = Image.fromarray(arr)
                    except Exception:
                        pil_img = None

                if pil_img is not None:
                    # Ensure RGB mode
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    if ext in [".jpg", ".jpeg"]:
                        pil_img.save(img_path, quality=95)
                    else:
                        pil_img.save(img_path)

                    saved_in_split += 1
                    total_images_saved += 1

        split_counts[dir_split] = saved_in_split
        logger.info(f"Saved {saved_in_split} images to {split_dir}")

    # Build and save dataset_info.json metadata
    info_payload = {
        "dataset_id": dataset_id,
        "format": "image_folder",
        "num_classes": len(discovered_class_names),
        "class_names": discovered_class_names,
        "split_names": list(split_counts.keys()),
        "split_sizes": split_counts,
        "total_images": total_images_saved,
        "image_format": image_format.lower().lstrip('.'),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    info_file = output_dir / "dataset_info.json"
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info_payload, f, indent=2)

    logger.info(f"Saved dataset metadata info to {info_file}")
    return info_payload


def download_single_dataset(
    dataset_id: str,
    registry: DatasetRegistry,
    force: bool = False,
    image_format: str = "png",
    max_samples: Optional[int] = None,
    failures_dir: Path = Path("DATA/failures"),
) -> bool:
    """Download a single dataset, extract physical image files, and save to DATA/<dataset_id>.

    Args:
        dataset_id: Identifier of the dataset.
        registry: DatasetRegistry instance.
        force: If True, re-downloads even if already present.
        image_format: Image file format to save ('png' or 'jpg').
        max_samples: Optional max samples per split.
        failures_dir: Directory for failure logs.

    Returns:
        True if dataset is successfully available on disk as images, False otherwise.
    """
    try:
        adapter = registry.get_adapter(dataset_id)
        meta = adapter.metadata
    except KeyError:
        logger.error(f"Dataset '{dataset_id}' not found in registry.")
        return False

    output_dir = adapter.resolved_path

    # Check if physical images already exist on disk
    if not force and adapter.is_available():
        # Check if actual image files exist in output_dir
        has_images = False
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        if output_dir.exists():
            for root, _, files in os.walk(str(output_dir)):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        has_images = True
                        break
                if has_images:
                    break

        if has_images:
            logger.info(f"Dataset '{dataset_id}' image files already present at {output_dir}. Reusing existing data.")
            prev_fail = failures_dir / f"{dataset_id}.json"
            if prev_fail.is_file():
                prev_fail.unlink(missing_ok=True)
            return True

    if not HAS_HF:
        exc = ImportError("The 'datasets' package is required to download Hugging Face datasets.")
        log_dataset_failure(dataset_id, meta, exc, [meta.hf_id] if meta.hf_id else [], failures_dir)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build candidate HF IDs list: primary first, then alternates
    candidate_hf_ids: List[str] = []
    if meta.hf_id:
        candidate_hf_ids.append(meta.hf_id)
    if meta.alternate_hf_ids:
        for alt in meta.alternate_hf_ids:
            if alt not in candidate_hf_ids:
                candidate_hf_ids.append(alt)

    if not candidate_hf_ids:
        exc = ValueError(f"No Hugging Face ID configured for dataset '{dataset_id}'.")
        log_dataset_failure(dataset_id, meta, exc, [], failures_dir)
        return False

    last_exception = None
    attempted: List[str] = []

    for hf_id in candidate_hf_ids:
        attempted.append(hf_id)
        logger.info(f"Attempting download for '{dataset_id}' from Hugging Face ID: '{hf_id}'...")
        try:
            kwargs: Dict[str, Any] = {"trust_remote_code": False}
            if meta.hf_config:
                kwargs["name"] = meta.hf_config

            ds = load_dataset(hf_id, **kwargs)
            logger.info(f"Successfully loaded '{hf_id}'. Available splits: {list(ds.keys())}")

            logger.info(f"Extracting image files into {output_dir}...")
            extract_and_save_images(
                hf_dataset=ds,
                output_dir=output_dir,
                dataset_id=dataset_id,
                image_format=image_format,
                max_samples=max_samples,
            )
            logger.info(f"Successfully extracted image files for '{dataset_id}' into {output_dir}")

            # Clean up failure record if previously recorded
            fail_file = failures_dir / f"{dataset_id}.json"
            if fail_file.is_file():
                fail_file.unlink(missing_ok=True)

            return True

        except Exception as e:
            logger.warning(f"Failed loading/extracting '{hf_id}' for dataset '{dataset_id}': {type(e).__name__}: {e}")
            last_exception = e

    # All candidate HF IDs failed
    if last_exception is not None:
        log_dataset_failure(
            dataset_id=dataset_id,
            metadata=meta,
            exception=last_exception,
            attempted_hf_ids=attempted,
            failures_dir=failures_dir,
        )

    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets and save physical image files under DATA/<dataset_id>."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration YAML.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset ID to download (or 'all' for all configured datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extract images even if dataset already exists locally.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image file format to save (default: png).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples per split (for testing/development).",
    )
    parser.add_argument(
        "--failures-dir",
        type=str,
        default="DATA/failures",
        help="Directory to save failure logs.",
    )
    return parser.parse_args()


def main():
    setup_logging(log_dir="logs", log_file_name="download_datasets.log")
    args = parse_args()

    registry = DatasetRegistry(config_path=args.config)
    failures_dir = Path(args.failures_dir)
    failures_dir.mkdir(parents=True, exist_ok=True)

    ds_arg = args.dataset
    if isinstance(ds_arg, list) and len(ds_arg) == 1 and ds_arg[0].lower() == "all":
        target_ids = registry.list_datasets(primary_only=False, enabled_only=True)
    elif isinstance(ds_arg, list):
        target_ids = ds_arg
    else:
        target_ids = [ds_arg]

    print("\n" + "=" * 80)
    print("CNN vs ViT BENCHMARK DATASET IMAGE DOWNLOADER & EXTRACTOR")
    print(f"Data root: DATA/ | Failures dir: {failures_dir} | Format: {args.image_format}")
    print("=" * 80)

    results: Dict[str, bool] = {}

    for ds_id in target_ids:
        print(f"\nProcessing: {ds_id}...")
        success = download_single_dataset(
            dataset_id=ds_id,
            registry=registry,
            force=args.force,
            image_format=args.image_format,
            max_samples=args.max_samples,
            failures_dir=failures_dir,
        )
        results[ds_id] = success
        status_text = "AVAILABLE / SUCCESS" if success else "FAILED (Logged to DATA/failures)"
        print(f"  -> {ds_id}: {status_text}")

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    for ds_id, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {ds_id:<20s} : {status}")
    print("=" * 80 + "\n")

    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning(f"Some datasets failed: {failed}. See DATA/failures/ for details.")
        sys.exit(1)
    else:
        logger.info("All requested datasets processed successfully.")


if __name__ == "__main__":
    main()