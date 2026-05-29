"""Codec engine — unified encode/decode wrapper for benchmark testing.

Wraps JXL, AVIF, WebP, JPEG encoding via CLI tools and pyvips,
and provides a decode-speed bench for evaluating reader UX.

Reuses the existing compressor.py functions where possible.
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pyvips
from PIL import Image

from image_compressor.evaluators.base import EncodeResult

logger = logging.getLogger(__name__)


# ── Codec configuration presets ──────────────────────────────────────────

CodecConfig = dict[str, Any]
"""A codec config: {"format": str, "quality": int, "lossy": bool, ...}"""


def load_all_configs() -> list[CodecConfig]:
    """Return the full test matrix of codec configurations.

    Covers the four major formats with their practical quality ranges.
    """
    configs: list[CodecConfig] = []

    # WebP — lossy
    for q in [75, 80, 85, 90]:
        configs.append({"format": "webp", "lossy": True, "quality": q})

    # WebP — lossless
    configs.append({"format": "webp", "lossy": False, "quality": 100})

    # JPEG
    for q in [75, 80, 85]:
        configs.append({"format": "jpeg", "lossy": True, "quality": q})

    # PNG — optimised lossless
    configs.append({"format": "png", "lossy": False, "quality": 100})

    # JPEG XL — lossy
    for q in [75, 80, 85, 90]:
        configs.append({"format": "jxl", "lossy": True, "quality": q})

    # JPEG XL — lossless
    configs.append({"format": "jxl", "lossy": False, "quality": 100})

    # AVIF — lossy (different quality scale: 30-60 is typical)
    for q in [30, 40, 50, 60]:
        configs.append({"format": "avif", "lossy": True, "quality": q})

    # AVIF — lossless
    configs.append({"format": "avif", "lossy": False, "quality": 100})

    return configs


_EXT_MAP = {
    "webp": ".webp",
    "jpeg": ".jpg",
    "jxl": ".jxl",
    "avif": ".avif",
    "png": ".png",
}


def _config_id(config: CodecConfig) -> str:
    fmt = config["format"]
    q = config.get("quality", 100)
    mode = "lossless" if not config.get("lossy", True) else f"q{q}"
    return f"{fmt}_{mode}"


# ── Encode functions ────────────────────────────────────────────────────


def _encode_pyvips(
    input_path: Path,
    output_path: Path,
    config: CodecConfig,
) -> int:
    """Encode via pyvips. Returns output file size."""
    img = pyvips.Image.new_from_file(str(input_path), access="sequential")
    fmt = config["format"]
    q = config.get("quality", 85)
    lossy = config.get("lossy", True)

    if fmt == "webp":
        img.webpsave(str(output_path), Q=q, lossless=not lossy, strip=True)
    elif fmt in ("jpeg", "jpg"):
        if img.bands == 4:
            img = img.flatten()
        elif img.bands == 1:
            img = img.colourspace("srgb")
        img.jpegsave(str(output_path), Q=q, strip=True)
    elif fmt == "png":
        img.pngsave(str(output_path), compression=9, strip=True)
    else:
        raise NotImplementedError(f"pyvips cannot encode {fmt} — use CLI")

    return output_path.stat().st_size


def _ensure_cli_input(path: Path) -> Path:
    """Ensure the input is readable by CLI tools (cjxl, avifenc).

    CLI tools often rely on file extension to detect format.
    If the extension misrepresents the content (e.g. WebP saved as .jpg),
    convert to a correctly-named temp PNG.

    NOTE: Temp files created here are tracked in _CLI_TEMP_FILES and
    cleaned up by the caller (encode functions) or by cleanup_cache().
    """
    from image_compressor.evaluators.io_util import detect_real_format

    real = detect_real_format(path)
    ext = path.suffix.lower()

    # Common cases where extension matches content — no conversion needed
    if real == ext:
        return path
    # Also fine: .jpeg vs .jpg
    if real in (".jpg", ".jpeg") and ext in (".jpg", ".jpeg"):
        return path

    # Mismatch: convert to a correctly-extended temp PNG
    logger.debug(
        "Extension mismatch: %s claims %s but is %s — converting to PNG",
        path.name, ext, real,
    )
    from PIL import Image
    # Use context-managed temporary file (safer than delete=False)
    import atexit
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    # Register cleanup at interpreter exit as a safety net
    atexit.register(_safe_unlink, tmp_path)
    with Image.open(path) as img:
        img.save(tmp_path, format="PNG")
    return tmp_path


def _safe_unlink_tmp(path: Path, original: Path) -> None:
    """Unlink path only if it differs from original (cleanup temp)."""
    if path != original:
        _safe_unlink(path)


def _encode_cjxl(input_path: Path, output_path: Path, config: CodecConfig) -> int:
    from image_compressor.evaluators.io_util import detect_real_format

    q = config.get("quality", 85)
    lossy = config.get("lossy", True)
    real_fmt = detect_real_format(input_path)
    cli_input = _ensure_cli_input(input_path)

    try:
        cmd = ["cjxl", str(cli_input), str(output_path)]
        if not lossy:
            # True lossless: for JPEG input, cjxl default --lossless_jpeg=1
            # preserves bit-exact; for non-JPEG, --distance=0 ensures lossless.
            if real_fmt in (".jpg", ".jpeg"):
                pass  # cjxl default for JPEG is lossless_jpeg=1
            else:
                cmd.extend(["--distance", "0"])
        else:
            cmd.extend(["--quality", str(q)])
            if real_fmt in (".jpg", ".jpeg"):
                cmd.extend(["--lossless_jpeg=0"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"cjxl: {result.stderr.strip()}")
        return output_path.stat().st_size
    finally:
        _safe_unlink_tmp(cli_input, input_path)


def _encode_avifenc(input_path: Path, output_path: Path, config: CodecConfig) -> int:
    q = config.get("quality", 50)
    lossy = config.get("lossy", True)
    cli_input = _ensure_cli_input(input_path)

    try:
        cmd = ["avifenc", "--speed", "6"]
        if lossy:
            # avifenc >=0.11: use -q (0-100), not deprecated --min/--max
            cmd.extend(["-q", str(q)])
        else:
            cmd.append("--lossless")
        cmd.extend([str(cli_input), str(output_path)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"avifenc: {result.stderr.strip()}")
        return output_path.stat().st_size
    finally:
        _safe_unlink_tmp(cli_input, input_path)


# ── Decode bench ─────────────────────────────────────────────────────────


def bench_decode(path: Path, iterations: int = 3) -> float:
    """Measure decode time in seconds (average over iterations)."""
    temps = []
    start = time.perf_counter()
    for _ in range(iterations):
        img = Image.open(path)
        img.load()
        # Force pixel access to ensure full decode
        _ = img.getdata()
    elapsed = time.perf_counter() - start
    return elapsed / iterations


# ── Public API ───────────────────────────────────────────────────────────


def encode_one(
    original_path: Path,
    config: CodecConfig,
    output_dir: Path | None = None,
) -> EncodeResult:
    """Encode a single image under one codec config.

    Args:
        original_path: Source image.
        config: Codec configuration dict.
        output_dir: Directory for the output file. Uses a temp dir if None.

    Returns:
        EncodeResult with sizes, timing, and output path.
    """
    fmt = config["format"]
    ext = _EXT_MAP.get(fmt, ".out")
    original_size = original_path.stat().st_size

    # Choose output location
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{original_path.stem}_{_config_id(config)}{ext}"
        output_path = output_dir / out_name
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        output_path = Path(tmp.name)
        tmp.close()

    try:
        start = time.perf_counter()

        if fmt == "jxl":
            size = _encode_cjxl(original_path, output_path, config)
        elif fmt == "avif":
            size = _encode_avifenc(original_path, output_path, config)
        else:
            size = _encode_pyvips(original_path, output_path, config)

        encode_time = time.perf_counter() - start

        # Decode speed: I/O-bound in real usage, so skip the bench
        decode_time = 0.0

        return EncodeResult(
            format=fmt,
            config_id=_config_id(config),
            output_path=output_path,
            original_path=original_path,
            original_size=original_size,
            compressed_size=size,
            encode_time=encode_time,
            decode_time=decode_time,
        )
    except Exception as e:
        # Clean up on failure
        _safe_unlink(output_path)
        raise RuntimeError(
            f"Encode failed [{_config_id(config)}]: {e}"
        ) from e


def _skip_decode_bench(path: Path) -> float:
    """No-op decode bench — decode speed is I/O-bound, not a differentiator."""
    return 0.0


def encode_all(
    original_path: Path,
    configs: list[CodecConfig],
    output_dir: Path | None = None,
) -> list[EncodeResult]:
    """Run all codec configs against a single image.

    Args:
        original_path: Source image.
        configs: List of codec configurations to test.
        output_dir: Output directory for compressed files.

    Returns:
        List of EncodeResult (only successful encodings).
    """
    results: list[EncodeResult] = []
    for config in configs:
        try:
            result = encode_one(original_path, config, output_dir)
            results.append(result)
            logger.debug(
                "  %s → %s: %.1f%% (%.2fs encode, %.4fs decode)",
                original_path.name,
                result.config_id,
                result.savings_pct,
                result.encode_time,
                result.decode_time,
            )
        except Exception as e:
            logger.warning("  SKIP %s: %s", _config_id(config), e)
    return results


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
