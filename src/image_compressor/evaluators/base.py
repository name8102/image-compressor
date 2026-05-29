"""Base classes and dataclasses for image quality evaluation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class EncodeResult:
    """Result of a single encode operation.

    Tracks both compression efficiency and speed metrics needed
    for the weighted scoring pipeline.
    """

    format: str
    config_id: str
    output_path: Path
    original_path: Path
    original_size: int
    compressed_size: int
    encode_time: float          # seconds
    decode_time: float = 0.0    # seconds — populated by decode bench
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Compressed / original (0.0–1.0, lower = better compression)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size

    @property
    def savings_pct(self) -> float:
        """Percentage of bytes saved: (1 - ratio) * 100."""
        return (1.0 - self.compression_ratio) * 100.0

    def add_score(self, metric_name: str, score: float) -> None:
        self.scores[metric_name] = score

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "config_id": self.config_id,
            "output_path": str(self.output_path),
            "original_path": str(self.original_path),
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": round(self.compression_ratio, 6),
            "savings_pct": round(self.savings_pct, 2),
            "encode_time": round(self.encode_time, 4),
            "decode_time": round(self.decode_time, 4),
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
        }


# ── Abstract evaluator ───────────────────────────────────────────────────


class QualityEvaluator(ABC):
    """Abstract base class for all quality evaluators.

    Each subclass implements a specific quality metric.
    The evaluate() method takes two file paths and returns a float score.
    """

    # Human-readable name for reports
    name: ClassVar[str] = "base"

    # Interpretation: "higher_is_better" or "lower_is_better"
    direction: ClassVar[str] = "higher_is_better"

    @abstractmethod
    def evaluate(self, original: Path, compressed: Path) -> float:
        """Compute a quality score for the compressed image vs. original.

        Args:
            original: Path to the original (reference) image.
            compressed: Path to the compressed image.

        Returns:
            A float score. Interpretation depends on ``direction``.
        """
        ...

    @classmethod
    def metric_name(cls) -> str:
        """Return the evaluator key (snake_case of name)."""
        return cls.name.lower().replace(" ", "_")


# ── Registry ─────────────────────────────────────────────────────────────


class EvaluatorRegistry:
    """Registry of QualityEvaluator instances, keyed by metric name.

    Supports lazy instantiation: register_class() stores the class
    and only creates the instance on first use.
    """

    def __init__(self) -> None:
        self._registry: dict[str, QualityEvaluator] = {}
        self._class_registry: dict[str, type[QualityEvaluator]] = {}

    def register_instance(self, evaluator: QualityEvaluator) -> None:
        key = evaluator.metric_name()
        self._registry[key] = evaluator
        logger.debug("Registered evaluator instance: %s", key)

    def register_class(self, cls: type[QualityEvaluator], **kwargs: Any) -> None:
        key = cls.metric_name()
        self._class_registry[key] = (cls, kwargs)
        logger.debug("Registered evaluator class: %s", key)

    def get(self, name: str) -> QualityEvaluator:
        """Retrieve an evaluator by metric name, creating it lazily if needed.

        Raises KeyError if the name is not registered.
        """
        if name in self._registry:
            return self._registry[name]
        if name in self._class_registry:
            cls, kwargs = self._class_registry[name]
            instance = cls(**kwargs)
            self._registry[name] = instance
            return instance
        available = list(self._registry.keys()) + list(self._class_registry.keys())
        raise KeyError(f"Unknown evaluator: '{name}'. Available: {available}")

    def list_names(self) -> list[str]:
        return sorted(set(self._registry.keys()) | set(self._class_registry.keys()))

    def list_evaluators(self) -> list[QualityEvaluator]:
        """Return all instantiated evaluators (may trigger lazy init)."""
        for name in self._class_registry:
            self.get(name)
        return list(self._registry.values())
