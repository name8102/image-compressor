"""Modern perceptual metrics via external binary tools.

SSIMULACRA2 and Butteraugli are state-of-the-art perceptual
quality metrics developed by Cloudinary and Google respectively.
They are invoked via subprocess to their compiled binaries.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from image_compressor.evaluators.base import QualityEvaluator

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def _find_tool(name: str) -> str | None:
    """Locate a binary on PATH; return None if absent."""
    path = shutil.which(name)
    if path is None:
        logger.debug("Tool '%s' not found on PATH", name)
    return path


def _run_tool(cmd: list[str], timeout: int = 120) -> str:
    """Run a subprocess and return stripped stdout. Raises on failure."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def _parse_float_output(raw: str) -> float:
    """Extract the first float from a tool's stdout."""
    import re
    match = re.search(r"[\d.]+", raw)
    if match:
        return float(match.group())
    raise ValueError(f"Cannot parse float from output: {raw!r}")


# ── Evaluators ───────────────────────────────────────────────────────────


class SSIMULACRA2Evaluator(QualityEvaluator):
    """SSIMULACRA2 — Cloudinary's structural similarity metric.

    Returns a score where *lower is better* (typical range 0-100).
    See: https://github.com/cloudinary/ssimulacra2

    direction: lower_is_better
    """

    name = "ssimulacra2"
    direction = "lower_is_better"

    def __init__(self, *, binary: str | None = None) -> None:
        self._binary = binary or _find_tool("ssimulacra2")
        if self._binary is None:
            logger.warning(
                "SSIMULACRA2 binary not found. "
                "Install via: https://github.com/cloudinary/ssimulacra2"
            )

    @property
    def available(self) -> bool:
        return self._binary is not None

    def evaluate(self, original: Path, compressed: Path) -> float:
        if not self.available:
            return float("nan")

        cmd = [self._binary, str(original), str(compressed)]
        try:
            raw = _run_tool(cmd)
            return _parse_float_output(raw)
        except Exception as e:
            logger.warning("SSIMULACRA2 failed: %s", e)
            return float("nan")


class ButteraugliEvaluator(QualityEvaluator):
    """Butteraugli — Google's psychovisual difference metric.

    Returns a heatmap-based score where *lower is better*.
    Typical values: <1.0 = imperceptible, 1-2 = barely visible,
    >3 = clearly visible artifacts.

    direction: lower_is_better
    """

    name = "butteraugli"
    direction = "lower_is_better"

    def __init__(self, *, binary: str | None = None) -> None:
        self._binary = binary or _find_tool("butteraugli")
        # Also try the libjxl-bundled butteraugli_main
        if self._binary is None:
            self._binary = _find_tool("butteraugli_main")
        if self._binary is None:
            logger.warning(
                "Butteraugli binary not found. "
                "Install libjxl-tools or build from: https://github.com/google/butteraugli"
            )

    @property
    def available(self) -> bool:
        return self._binary is not None

    def evaluate(self, original: Path, compressed: Path) -> float:
        if not self.available:
            return float("nan")

        # butteraugli expects PNGs (or PNM). Convert via pyvips/Pillow if needed.
        orig_png = _ensure_png(original)
        comp_png = _ensure_png(compressed)

        try:
            cmd = [self._binary, str(orig_png), str(comp_png)]
            raw = _run_tool(cmd)
            return _parse_butteraugli_output(raw)
        except Exception as e:
            logger.warning("Butteraugli failed: %s", e)
            return float("nan")
        finally:
            # Clean up temps only if we created them
            if orig_png != original:
                _safe_unlink(orig_png)
            if comp_png != compressed:
                _safe_unlink(comp_png)


def _ensure_png(path: Path) -> Path:
    """Return path as-is if PNG, otherwise convert to a temp PNG."""
    from image_compressor.evaluators.io_util import detect_real_format
    if detect_real_format(path) == ".png":
        return path
    import tempfile
    from PIL import Image
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    with Image.open(path) as img:
        img.save(tmp_path, format="PNG")
    return tmp_path


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _parse_butteraugli_output(raw: str) -> float:
    """Butteraugli outputs the pnorm-max value; extract it."""
    # Typical output: "pnorm-max: 1.2345"
    for line in raw.splitlines():
        if "pnorm-max" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                return float(parts[1].strip())
    # Fallback: try any float
    return _parse_float_output(raw)
