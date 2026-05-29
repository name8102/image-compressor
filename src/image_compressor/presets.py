"""Compression presets management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Strategy(str, Enum):
    """Compression strategy."""
    LOSSY = "lossy"
    LOSSLESS = "lossless"
    AUTO = "auto"


class OutputFormat(str, Enum):
    """Output image format."""
    WEBP = "webp"
    JPEG_XL = "jpeg-xl"
    AVIF = "avif"
    JPEG = "jpeg"
    PNG = "png"
    KEEP = "keep"  # Keep original format


@dataclass
class Preset:
    """Compression preset configuration."""

    name: str
    description: str = ""
    strategy: Strategy = Strategy.AUTO
    format: OutputFormat = OutputFormat.WEBP
    quality: int = 85
    process_cbz: bool = False

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> Preset:
        """Create preset from configuration dictionary.

        Args:
            name: Preset name.
            data: Preset configuration dictionary.

        Returns:
            Preset instance.
        """
        return cls(
            name=name,
            description=data.get("description", ""),
            strategy=Strategy(data.get("strategy", "auto")),
            format=OutputFormat(data.get("format", "webp")),
            quality=data.get("quality", 85),
            process_cbz=data.get("process_cbz", False),
        )

    def get_compression_params(self, source_format: str) -> dict[str, Any]:
        """Get compression parameters for a given source format.

        Args:
            source_format: Source image format (e.g., 'jpeg', 'png', 'webp').

        Returns:
            Dictionary of compression parameters.
        """
        # Determine effective strategy
        strategy = self.strategy
        if strategy == Strategy.AUTO:
            # Auto: lossy for photos (jpeg), lossless for others (png, webp)
            if source_format.lower() in ("jpeg", "jpg"):
                strategy = Strategy.LOSSY
            else:
                strategy = Strategy.LOSSLESS

        # Determine effective format
        output_format = self.format
        if output_format == OutputFormat.KEEP:
            output_format = OutputFormat(source_format)

        params: dict[str, Any] = {
            "format": output_format.value,
            "strategy": strategy.value,
        }

        if strategy == Strategy.LOSSY:
            params["quality"] = self.quality
        else:
            params["lossless"] = True

        return params


def load_presets(config: dict[str, Any]) -> dict[str, Preset]:
    """Load all presets from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Dictionary mapping preset name to Preset instance.
    """
    presets_data = config.get("presets", {})
    return {
        name: Preset.from_dict(name, data)
        for name, data in presets_data.items()
    }
