"""Image I/O utilities for evaluators.

Provides format-agnostic image reading with automatic fallback
to CLI decoders (djxl, avifdec) for formats Pillow doesn't support.
Also detects real image format from magic bytes to handle files
with misleading extensions (e.g. PNG saved as .jpg).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PNG_CACHE: dict[Path, Path] = {}
"""Cache decoded temp PNGs so repeated evaluations hit the same file."""


# ── Real format detection ────────────────────────────────────────────────

# Magic-byte signatures: (offset, bytes) → canonical extension
_MAGIC_SIGNATURES: list[tuple[int, bytes, str | None]] = [
    (0, b'\xff\xd8\xff',                     '.jpg'),   # JPEG
    (0, b'\x89PNG\r\n\x1a\n',                '.png'),   # PNG
    (0, b'GIF8',                               '.gif'),   # GIF
    (0, b'RIFF',                               None),    # RIFF container → check WebP below
    (0, b'\xff\x0a',                           '.jxl'),   # JXL (partial codestream)
    (0, b'\x00\x00\x00\x0cJXL \r\n\x87\n',   '.jxl'),   # JXL container
]
_WEBP_MAGIC = b'WEBP'


def detect_real_format(path: Path) -> str:
    """Detect actual image format from file header bytes.

    Ignores file extension entirely. Returns a canonical extension
    like '.jpg', '.png', '.webp', '.jxl', '.avif'.

    Falls back to Pillow for exotic formats, then file extension.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(24)
    except OSError:
        return path.suffix.lower()

    for offset, magic, ext in _MAGIC_SIGNATURES:
        if header[offset:offset + len(magic)] == magic:
            if magic == b'RIFF':
                if len(header) >= 12 and header[8:12] == _WEBP_MAGIC:
                    return '.webp'
                continue
            if ext is not None:
                return ext

    try:
        from PIL import Image
        with Image.open(path) as img:
            if img.format:
                return '.' + img.format.lower()
    except Exception:
        pass

    return path.suffix.lower()


def read_gray(path: Path) -> np.ndarray:
    """Read an image as grayscale float64, auto-decoding exotic formats.

    Supports all common formats via Pillow. Falls back to
    djxl/avifdec for JXL/AVIF.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            return np.array(gray, dtype=np.float64)
    except Exception:
        pass

    # Try CLI decoders
    decoded = _decode_to_png(path)
    if decoded is not None:
        with Image.open(decoded) as img:
            gray = img.convert("L")
            return np.array(gray, dtype=np.float64)

    raise ValueError(f"Cannot read image: {path}")


def _decode_to_png(path: Path) -> Path | None:
    """Decode JXL/AVIF to a temp PNG using CLI tools. Results are cached."""
    if path in _PNG_CACHE:
        return _PNG_CACHE[path]

    real_fmt = detect_real_format(path)
    decoder = None

    if real_fmt == ".jxl":
        decoder = _decode_jxl
    elif real_fmt == ".avif":
        decoder = _decode_avif

    if decoder is None:
        return None

    try:
        png_path = decoder(path)
        _PNG_CACHE[path] = png_path
        return png_path
    except Exception as e:
        logger.debug("CLI decode failed for %s: %s", path.name, e)
        return None


def _decode_jxl(path: Path) -> Path:
    """Decode JXL to PNG via djxl."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    result = subprocess.run(
        ["djxl", str(path), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"djxl: {result.stderr.strip()}")
    return tmp_path


def _decode_avif(path: Path) -> Path:
    """Decode AVIF to PNG via avifdec."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    result = subprocess.run(
        ["avifdec", str(path), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"avifdec: {result.stderr.strip()}")
    return tmp_path


def read_pair_for_comparison(original: Path, compressed: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read original and compressed images, using the same decode
    pipeline when necessary for fair comparison.

    Problem: cjxl lossless transcoding preserves JPEG bytes, but
    Pillow's JPEG decoder and djxl's JPEG decoder produce slightly
    different pixel values.  When the compressed image is JXL, this
    decodes BOTH images through the cjxl→djxl pipeline so they use
    the same decoder and lossless formats score correctly.
    """
    comp_fmt = detect_real_format(compressed)
    orig_fmt = detect_real_format(original)

    # Only need the workaround when compressed is JXL and original is JPEG
    if comp_fmt == '.jxl' and orig_fmt in ('.jpg', '.jpeg'):
        orig_png = _jxl_fair_decode(original)
        comp_png = _jxl_fair_decode(compressed)
        try:
            return read_gray(orig_png), read_gray(comp_png)
        finally:
            _safe_unlink(orig_png)
            _safe_unlink(comp_png)

    return read_gray(original), read_gray(compressed)


def _jxl_fair_decode(path: Path) -> Path:
    """Decode an image to a temp PNG via the cjxl→djxl pipeline.

    For JPEG originals, this encodes to lossless JXL then decodes back,
    so the pixel values match what djxl produces from the compressed JXL.
    For existing JXL files, this just runs djxl.
    """
    fmt = detect_real_format(path)
    if fmt == '.jxl':
        return _decode_to_png(path) or path  # type: ignore[return-value]

    # JPEG → cjxl lossless → djxl → PNG
    jxl_tmp = tempfile.NamedTemporaryFile(suffix='.jxl', delete=False)
    jxl_path = Path(jxl_tmp.name)
    jxl_tmp.close()
    try:
        subprocess.run(
            ['cjxl', str(path), str(jxl_path)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        return _decode_to_png(jxl_path) or jxl_path
    finally:
        _safe_unlink(jxl_path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def cleanup_cache() -> None:
    """Remove all cached temp PNGs."""
    for p in _PNG_CACHE.values():
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    _PNG_CACHE.clear()
