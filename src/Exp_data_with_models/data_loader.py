"""Standardized Data Loading and Transformation Pipeline for CNNs and Vision Transformers."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.datasets.registry import DatasetRegistry
from src.datasets.base import BaseDatasetAdapter, StandardDataset, StandardDatasetSample
from src.datasets.subsets import generate_stratified_nested_subsets


def get_standard_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    is_training: bool = True,
    use_augmentation: bool = True,
) -> transforms.Compose:
    """Build standardized torchvision transformation pipeline.

    Args:
        image_size: Target (height, width) resolution (default: (224, 224)).
        mean: RGB normalization channel means.
        std: RGB normalization channel standard deviations.
        is_training: If True, apply training augmentations.
        use_augmentation: If True and is_training, use RandomResizedCrop/Flip/ColorJitter.

    Returns:
        torchvision transforms.Compose pipeline.
    """
    if is_training and use_augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Evaluation / Test preprocessing
        resize_dim = int(round(image_size[0] * (256.0 / 224.0)))
        return transforms.Compose([
            transforms.Resize(resize_dim, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def load_manifest_subset(
    all_train_samples: List[StandardDatasetSample],
    manifest_path: Path,
) -> List[StandardDatasetSample]:
    """Filter training samples according to a precomputed nested manifest CSV.

    Args:
        all_train_samples: Complete list of training StandardDatasetSample objects.
        manifest_path: Path to train_<fraction>.csv manifest.

    Returns:
        Filtered list of samples matching the manifest sample IDs.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    sample_id_set = set(df["sample_id"].astype(str))

    sample_map = {str(s.sample_id): s for s in all_train_samples}
    filtered = [sample_map[sid] for sid in df["sample_id"].astype(str) if sid in sample_map]
    return filtered


def get_data_loaders(
    dataset_id: str = "cifar100",
    config_path: str = "configs/datasets.yaml",
    data_root: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    data_fraction: int = 100,
    seed: int = 42,
    image_size: Tuple[int, int] = (224, 224),
    pin_memory: bool = False,
    shuffle_train: bool = True,
    use_manifest: bool = True,
    use_augmentation: bool = True,
) -> Dict[str, Any]:
    """Create PyTorch DataLoaders for train, val, and test splits.

    Args:
        dataset_id: Identifier of the dataset (e.g. 'cifar100', 'eurosat', 'dtd').
        config_path: Path to datasets.yaml.
        data_root: Optional root directory override.
        batch_size: Mini-batch size.
        num_workers: Number of DataLoader subprocesses.
        data_fraction: Percentage of training data to use (5, 10, 25, 50, 75, 100).
        seed: Random seed for subset partitioning and determinism.
        image_size: Input spatial resolution (default: (224, 224)).
        pin_memory: If True, copy Tensors into pinned memory before returning.
        shuffle_train: Whether to shuffle training DataLoader.
        use_manifest: If True, load precomputed manifest from DATA/manifests/ if available.
        use_augmentation: Whether to apply data augmentation during training.

    Returns:
        Dictionary containing:
            - 'train': DataLoader for training partition
            - 'val': DataLoader for validation partition
            - 'test': DataLoader for test partition
            - 'num_classes': Total number of classes
            - 'classes': List of class names
            - 'train_samples_count': Count of training images
            - 'val_samples_count': Count of validation images
            - 'test_samples_count': Count of test images
            - 'dataset_id': Identifier of the dataset
    """
    registry = DatasetRegistry(config_path=config_path, data_root=data_root)
    adapter = registry.get_adapter(dataset_id)

    if not adapter.is_available():
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' is not available at {adapter.resolved_path.resolve()}. "
            f"Run 'python scripts/download_datasets.py --dataset {dataset_id}' first."
        )

    all_splits = adapter.load_samples()
    classes = adapter.get_classes()
    num_classes = len(classes)

    raw_train_samples = all_splits.get("train", [])
    val_samples = all_splits.get("val", [])
    test_samples = all_splits.get("test", [])

    # Apply data fraction scaling (Nested Subsets)
    if data_fraction < 100:
        manifest_file = Path("DATA/manifests") / dataset_id / f"seed_{seed}" / f"train_{data_fraction}.csv"
        if use_manifest and manifest_file.is_file():
            train_samples = load_manifest_subset(raw_train_samples, manifest_file)
        else:
            # Generate stratified nested subset dynamically
            subsets = generate_stratified_nested_subsets(
                train_samples=raw_train_samples,
                fractions=[data_fraction],
                seed=seed,
            )
            train_samples = subsets[data_fraction]
    else:
        train_samples = raw_train_samples

    # Build transforms
    train_transform = get_standard_transforms(
        image_size=image_size,
        is_training=True,
        use_augmentation=use_augmentation,
    )
    eval_transform = get_standard_transforms(
        image_size=image_size,
        is_training=False,
    )

    train_dataset = StandardDataset(samples=train_samples, transform=train_transform)
    val_dataset = StandardDataset(samples=val_samples, transform=eval_transform)
    test_dataset = StandardDataset(samples=test_samples, transform=eval_transform)

    # PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": num_classes,
        "classes": classes,
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "test_samples_count": len(test_samples),
        "dataset_id": dataset_id,
        "data_fraction": data_fraction,
        "seed": seed,
    }
