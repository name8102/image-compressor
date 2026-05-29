"""Configuration management for image-compressor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path("config.toml")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load TOML configuration file.

    Args:
        path: Path to config file. If None, uses default location.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy config.example.toml to config.toml and edit it."
        )

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_preset(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Get a compression preset by name.

    Args:
        config: Full configuration dictionary.
        name: Preset name.

    Returns:
        Preset configuration dictionary.

    Raises:
        KeyError: If preset doesn't exist.
    """
    presets = config.get("presets", {})
    if name not in presets:
        available = ", ".join(presets.keys()) if presets else "(none)"
        raise KeyError(f"Preset '{name}' not found. Available: {available}")
    return presets[name]


def get_workers(config: dict[str, Any]) -> int:
    """Get number of worker processes.

    Args:
        config: Full configuration dictionary.

    Returns:
        Number of workers (default: 3).
    """
    return config.get("workers", 3)
