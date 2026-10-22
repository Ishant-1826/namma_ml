"""Configuration loading utilities for Namma Space pipeline stages."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Please ensure configs/default.yaml exists or pass a valid --config path."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}
