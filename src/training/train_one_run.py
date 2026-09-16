"""Single experiment execution helper function."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch

from src.datasets import get_data_loaders
from src.models.factory import create_model
from src.training.trainer import ModelTrainer
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def run_training_experiment(
    model_name: str,
    dataset_name: str = "cifar100",
    data_dir: Union[str, Path] = "DATA",
    data_fraction: int = 100,
    seed: int = 42,
    epochs: int = 100,
    warmup_epochs: int = 5,
    batch_size: int = 64,
    base_lr: float = 5e-4,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.0,
    pretrained: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    quick_test: bool = False,
) -> Dict[str, Any]:
    """Execute a single model-dataset training condition and save complete multi-dimensional artifacts.

    Args:
        model_name: Model name (e.g. 'resnet50', 'vit_b16', 'swin_t', 'convnext_tiny').
        dataset_name: Dataset identifier (e.g. 'cifar100', 'dtd', 'eurosat').
        data_dir: Root dataset folder.
        data_fraction: Training percentage (5, 10, 25, 50, 75, 100).
        seed: Random seed.
        epochs: Total training epochs (overridden to 1 if quick_test=True).
        warmup_epochs: Warmup epochs (overridden to 0 if quick_test=True).
        batch_size: Mini-batch size.
        base_lr: Initial learning rate.
        weight_decay: Weight decay penalty.
        label_smoothing: Label smoothing regularization.
        pretrained: If True, load ImageNet-1K pretrained weights.
        device: Device string ('cuda', 'cpu').
        output_dir: Optional custom result directory.
        quick_test: If True, runs a single-epoch fast smoke run.

    Returns:
        Summary metrics dictionary.
    """
    set_seed(seed)
    logger = get_logger("train_one_run")

    actual_epochs = 1 if quick_test else epochs
    actual_warmup = 0 if quick_test else warmup_epochs

    logger.info(
        f"Initializing Run: model={model_name}, dataset={dataset_name}, "
        f"fraction={data_fraction}%, seed={seed}, pretrained={pretrained}, quick_test={quick_test}"
    )

    loaders, bundle = get_data_loaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        data_fraction=data_fraction,
        seed=seed,
        batch_size=batch_size,
    )

    model = create_model(
        model_name=model_name,
        num_classes=bundle.num_classes,
        pretrained=pretrained,
    )

    trainer = ModelTrainer(
        model=model,
        model_name=model_name,
        data_loaders=loaders,
        dataset_id=dataset_name,
        num_classes=bundle.num_classes,
        epochs=actual_epochs,
        warmup_epochs=actual_warmup,
        base_lr=base_lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        data_fraction=data_fraction,
        seed=seed,
        pretrained=pretrained,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )

    summary = trainer.fit()
    return summary
