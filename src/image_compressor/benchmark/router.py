"""Metrics router — maps image categories to evaluator sets.

Loads the metrics_routing.yaml configuration and provides lookup
by category type.  Also handles automatic category detection when
no explicit category is given.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default config path ──────────────────────────────────────────────────

_DEFAULT_ROUTING_PATH = Path(__file__).resolve().parents[3] / "config" / "metrics_routing.yaml"

# ── Category detection ───────────────────────────────────────────────────


def _is_animated(path: Path) -> bool:
    """Check if image has multiple frames (GIF, APNG, animated WebP)."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return hasattr(img, "n_frames") and img.n_frames > 1
    except Exception:
        return False


def _is_grayscale(path: Path) -> bool:
    """Check if image is effectively grayscale.

    Uses the project's grayscale_detector if available for accurate
    detection (handles tinted scans vs. true color); falls back to
    a simple chroma check.
    """
    try:
        from image_compressor.grayscale_detector import detect_grayscale
        import pyvips
        img = pyvips.Image.new_from_file(str(path), access="sequential")
        if img.bands < 3:
            return True
        hsv = img.colourspace("hsv")
        h = hsv[0].write_to_memory()
        s = hsv[1].write_to_memory()
        v = hsv[2].write_to_memory()
        import numpy as np
        sat_np = np.frombuffer(s, dtype=np.uint8).copy()
        hue_np = np.frombuffer(h, dtype=np.uint8).copy().astype(np.float64)
        hue_np = hue_np * (360.0 / 255.0)
        val_np = np.frombuffer(v, dtype=np.uint8).copy()
        is_gray, _ = detect_grayscale(sat_np, hue_np, val_np=val_np)
        return is_gray
    except Exception:
        pass
    # Fallback: check via Pillow mode
    try:
        from PIL import Image
        with Image.open(path) as img:
            if img.mode in ("L", "LA", "1", "P"):
                # P mode could be color palette — check actual colors used
                if img.mode == "P":
                    palette = img.getpalette()
                    if palette:
                        # Check if palette has non-gray entries
                        for i in range(0, min(len(palette), 768), 3):
                            r, g, b = palette[i], palette[i+1], palette[i+2]
                            if abs(r - g) > 5 or abs(r - b) > 5 or abs(g - b) > 5:
                                return False
                        return True
                return True
            return False
    except Exception:
        return False


def detect_category(image_path: Path) -> str:
    """Auto-detect image category for routing.

    Returns one of: 'photo', 'manga_bw', 'manga_color', 'animated'
    """
    if _is_animated(image_path):
        return "animated"
    if _is_grayscale(image_path):
        return "manga_bw"
    # Default: treat as manga_color (worst case — runs more evaluators)
    return "manga_color"


# ── Router ───────────────────────────────────────────────────────────────


class MetricsRouter:
    """Maps image categories to their evaluator weights and ranking config."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._config = self._load_config(config_path)
        self._categories: dict[str, dict] = self._config.get("categories", {})
        self._ranking_weights: dict[str, float] = self._config.get("ranking_weights", {
            "size": 0.20, "quality": 0.50, "encode": 0.10, "decode": 0.20,
        })

    @staticmethod
    def _load_config(path: Path | str | None) -> dict[str, Any]:
        import sys
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]

        try:
            import yaml
        except ModuleNotFoundError:
            logger.warning("PyYAML not installed — cannot load metrics_routing.yaml")
            return {}

        filepath = Path(path) if path else _DEFAULT_ROUTING_PATH
        if not filepath.exists():
            logger.warning("Routing config not found: %s", filepath)
            return {}
        with open(filepath) as f:
            return yaml.safe_load(f) or {}

    @property
    def category_names(self) -> list[str]:
        return list(self._categories.keys())

    @property
    def ranking_weights(self) -> dict[str, float]:
        return dict(self._ranking_weights)

    def get_metrics(self, category: str) -> list[tuple[str, float, bool]]:
        """Return evaluator configs for a category.

        Returns:
            List of (metric_name, weight, higher_is_better) tuples.
        """
        cat = self._categories.get(category)
        if cat is None:
            raise KeyError(
                f"Unknown category: '{category}'. "
                f"Available: {self.category_names}"
            )
        result: list[tuple[str, float, bool]] = []
        for name, cfg in cat.get("metrics", {}).items():
            result.append((
                name,
                float(cfg["weight"]),
                bool(cfg["higher_is_better"]),
            ))
        return result

    def get_metric_names(self, category: str) -> list[str]:
        """Return just the metric name list for a category."""
        return [m[0] for m in self.get_metrics(category)]

    def get_all_metric_names(self) -> list[str]:
        """Union of all metric names across all categories."""
        names: set[str] = set()
        for cat_name in self._categories:
            for name, _, _ in self.get_metrics(cat_name):
                names.add(name)
        return sorted(names)
