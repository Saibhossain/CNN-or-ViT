"""Configuration loading, validation, and resolution utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing parsed YAML content.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML parsing fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path}: {e}") from e


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        Merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    main_config_path: Union[str, Path] = "configs/experiment.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load master experiment configuration with all referenced sub-configs.

    Args:
        main_config_path: Path to master configuration YAML.
        overrides: Optional dictionary of parameter overrides.

    Returns:
        Combined configuration dictionary.
    """
    config = load_yaml(main_config_path)

    # Load referenced sub-configs if present in paths block
    paths_block = config.get("paths", {})
    sub_config_keys = [
        ("datasets_config", "datasets"),
        ("models_config", "models"),
        ("training_config", "training"),
        ("benchmark_config", "benchmark"),
        ("robustness_config", "robustness"),
        ("hardware_config", "hardware"),
    ]

    for path_key, target_key in sub_config_keys:
        if path_key in paths_block:
            sub_path = Path(paths_block[path_key])
            if sub_path.is_file():
                sub_content = load_yaml(sub_path)
                config[target_key] = sub_content

    if overrides:
        config = deep_merge(config, overrides)

    return config


def validate_dataset_entry(entry: Dict[str, Any]) -> List[str]:
    """Validate that a dataset configuration entry contains all required fields.

    Args:
        entry: Dataset configuration dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    required_fields = [
        "dataset_id",
        "dataset_name",
        "local_path",
        "domain",
        "license",
    ]
    errors = []
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field '{field}' in dataset entry: {entry.get('dataset_id', 'UNKNOWN')}")

    if "expected" in entry:
        expected = entry["expected"]
        if "num_classes" in expected and (not isinstance(expected["num_classes"], int) or expected["num_classes"] <= 0):
            errors.append(f"expected.num_classes must be a positive integer in dataset: {entry.get('dataset_id')}")
        if "expected_image_channels" in expected and expected["expected_image_channels"] not in [1, 3, 4]:
            errors.append(f"expected.expected_image_channels must be 1, 3, or 4 in dataset: {entry.get('dataset_id')}")
    elif "num_classes" in entry:
        if not isinstance(entry["num_classes"], int) or entry["num_classes"] <= 0:
            errors.append(f"num_classes must be a positive integer in dataset: {entry.get('dataset_id')}")

    return errors


def validate_model_entry(model_id: str, entry: Dict[str, Any]) -> List[str]:
    """Validate that a model configuration entry contains all required fields.

    Args:
        model_id: Key identifier of the model.
        entry: Model configuration dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    required_fields = ["name", "family", "default_input_size"]
    errors = []
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field '{field}' in model '{model_id}'")

    if "family" in entry and entry["family"] not in ["CNN", "ViT", "Transformer", "Hybrid"]:
        errors.append(f"Invalid architecture family '{entry['family']}' for model '{model_id}'")

    return errors


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the loaded configuration structure against project requirements.

    Args:
        config: Full configuration dictionary.

    Returns:
        True if valid.

    Raises:
        ValueError: If validation fails with detailed error messages.
    """
    all_errors: List[str] = []

    # Validate dataset definitions if loaded
    if "datasets" in config:
        datasets_cfg = config["datasets"]
        primary = datasets_cfg.get("primary_datasets", [])
        if not primary:
            all_errors.append("No primary_datasets defined in datasets configuration.")
        for ds in primary:
            all_errors.extend(validate_dataset_entry(ds))

    # Validate model definitions if loaded
    if "models" in config:
        models_cfg = config["models"].get("models", {})
        if not models_cfg:
            all_errors.append("No models defined in models configuration.")
        for model_id, model_data in models_cfg.items():
            all_errors.extend(validate_model_entry(model_id, model_data))

    if all_errors:
        error_msg = "\n  - ".join(all_errors)
        raise ValueError(f"Configuration validation failed:\n  - {error_msg}")

    return True


validate_all_configs = validate_config

