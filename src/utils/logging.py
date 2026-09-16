"""Structured and human-readable logging utilities."""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formatter for machine-readable JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_entry.update(record.extra)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(
    log_dir: Optional[str] = "logs",
    log_level: int = logging.INFO,
    log_file_name: Optional[str] = None,
    log_prefix: Optional[str] = None,
    console_json: bool = False,
) -> None:
    """Initialize root logging configuration for console and optional file output.

    Args:
        log_dir: Directory to save log files (created if not present, None to disable file logging).
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file_name: Base name for the main log file.
        log_prefix: Optional prefix to construct log file name ({prefix}.log).
        console_json: Whether console output should be formatted as JSON.
    """
    if log_file_name is None:
        if log_prefix is not None:
            log_file_name = f"{log_prefix}.log"
        else:
            log_file_name = "framework.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if console_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_fmt = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File Handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file_name, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

        # Separate JSON machine-readable log file
        json_file_handler = logging.FileHandler(log_path / f"{Path(log_file_name).stem}.jsonl", encoding="utf-8")
        json_file_handler.setLevel(log_level)
        json_file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: Logger identifier.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


def log_step(step_name: str, status: str = "started", logger: Optional[logging.Logger] = None, **kwargs: Any) -> None:
    """Log structured execution step information."""
    lg = logger or get_logger("framework")
    extra_info = " | " + ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    lg.info(f"[{step_name}] Status: {status}{extra_info}")

