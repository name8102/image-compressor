"""Image compression core using pyvips, Pillow, and CLI tools."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_animated(path: str) -> bool:
    """Check if an image is animated (GIF, APNG, animated WebP).
    
    Args:
        path: Path to image file.
        
    Returns:
        True if the image has multiple frames.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            return hasattr(img, 'n_frames') and img.n_frames > 1
    except Exception:
        return False


def get_image_info(path: str) -> dict[str, Any]:
    """Get image information including animation status.
    
    Args:
        path: Path to image file.
        
    Returns:
        Dictionary with image info.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "animated": False,
                "frames": 1,
            }
            if hasattr(img, 'n_frames'):
                info["animated"] = img.n_frames > 1
                info["frames"] = img.n_frames
            return info
    except Exception as e:
        return {"error": str(e)}


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    success: bool
    original_path: str
    output_path: str | None = None
    original_size: int = 0
    compressed_size: int = 0
    processing_time: float = 0.0
    error: str | None = None

    @property
    def savings_bytes(self) -> int:
        """Bytes saved by compression."""
        return self.original_size - self.compressed_size

    @property
    def ratio(self) -> float:
        """Compression ratio (compressed / original * 100)."""
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size * 100


def compress_image_jxl_cli(
    input_path: str,
    output_path: str,
    params: dict[str, Any],
) -> int:
    """Compress an image to JPEG XL using cjxl.

    Args:
        input_path: Path to input image.
        output_path: Path to write compressed image.
        params: Compression parameters.

    Returns:
        Output file size in bytes.
    """
    quality = params.get("quality", 85)
    lossless = params.get("lossless", False)
    
    cmd = ["cjxl", input_path, output_path]
    
    if lossless:
        cmd.extend(["--lossless"])
    else:
        cmd.extend(["--quality", str(quality)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"cjxl failed: {result.stderr}")
    
    return Path(output_path).stat().st_size


def compress_image_avif_cli(
    input_path: str,
    output_path: str,
    params: dict[str, Any],
) -> int:
    """Compress an image to AVIF using avifenc.

    Args:
        input_path: Path to input image.
        output_path: Path to write compressed image.
        params: Compression parameters.

    Returns:
        Output file size in bytes.
    """
    quality = params.get("quality", 50)
    lossless = params.get("lossless", False)
    
    cmd = ["avifenc"]
    
    if lossless:
        cmd.extend(["--lossless"])
    else:
        cmd.extend(["--min", str(quality), "--max", str(quality)])
    
    cmd.extend(["--speed", "6", input_path, output_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"avifenc failed: {result.stderr}")
    
    return Path(output_path).stat().st_size


def compress_animated_image_pillow(
    input_path: str,
    output_path: str,
    params: dict[str, Any],
) -> int:
    """Compress an animated image (GIF, APNG) preserving all frames.
    
    Args:
        input_path: Path to input animated image.
        output_path: Path to write compressed image.
        params: Compression parameters.
    
    Returns:
        Output file size in bytes.
    """
    from PIL import Image
    
    output_format = params.get("format", "webp")
    quality = params.get("quality", 85)
    lossless = params.get("lossless", False)
    
    with Image.open(input_path) as img:
        if not hasattr(img, 'n_frames') or img.n_frames <= 1:
            raise ValueError("Image is not animated")
        
        # Collect all frames
        frames = []
        durations = []
        
        for frame_idx in range(img.n_frames):
            img.seek(frame_idx)
            frame = img.copy()
            
            # Convert to RGBA for WebP
            if output_format == "webp":
                if frame.mode == 'P':
                    frame = frame.convert('RGBA')
                elif frame.mode == 'RGB':
                    frame = frame.convert('RGBA')
            
            frames.append(frame)
            
            # Get frame duration
            duration = img.info.get('duration', 100)
            durations.append(duration)
        
        # Save as animated WebP
        if output_format == "webp":
            frames[0].save(
                output_path,
                format='WEBP',
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=img.info.get('loop', 0),
                quality=quality,
                lossless=lossless,
                method=4,
            )
        elif output_format in ("gif", "keep"):
            # Keep as GIF
            frames[0].save(
                output_path,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=img.info.get('loop', 0),
                optimize=True,
            )
        elif output_format == "apng":
            # Save as APNG
            frames[0].save(
                output_path,
                format='PNG',
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=img.info.get('loop', 0),
            )
        else:
            raise ValueError(f"Unsupported animated format: {output_format}")
    
    return Path(output_path).stat().st_size


def compress_image_pyvips(
    input_path: str,
    output_path: str,
    params: dict[str, Any],
) -> int:
    """Compress an image using pyvips.

    Args:
        input_path: Path to input image.
        output_path: Path to write compressed image.
        params: Compression parameters (format, quality, lossless, etc.).

    Returns:
        Output file size in bytes.
    """
    import pyvips

    image = pyvips.Image.new_from_file(input_path, access="sequential")

    output_format = params.get("format", "webp")
    quality = params.get("quality", 85)
    lossless = params.get("lossless", False)

    if output_format == "webp":
        image.webpsave(
            output_path,
            Q=quality,
            lossless=lossless,
            strip=True,  # Remove metadata
        )
    elif output_format in ("jpeg", "jpg"):
        image.jpegsave(
            output_path,
            Q=quality,
            strip=True,
        )
    elif output_format == "png":
        image.pngsave(
            output_path,
            compression=9 if lossless else 6,
            strip=True,
        )
    elif output_format == "jpeg-xl":
        # pyvips may not support JXL, fall back to CLI
        raise NotImplementedError("JPEG XL requires CLI tool (cjxl)")
    elif output_format == "avif":
        # pyvips may not support AVIF, fall back to CLI
        raise NotImplementedError("AVIF requires CLI tool (avifenc)")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return Path(output_path).stat().st_size


def compress_image_pillow(
    input_path: str,
    output_path: str,
    params: dict[str, Any],
) -> int:
    """Compress an image using Pillow (fallback).

    Args:
        input_path: Path to input image.
        output_path: Path to write compressed image.
        params: Compression parameters (format, quality, lossless, etc.).

    Returns:
        Output file size in bytes.
    """
    from PIL import Image

    output_format = params.get("format", "webp")
    quality = params.get("quality", 85)
    lossless = params.get("lossless", False)

    with Image.open(input_path) as img:
        # Convert RGBA to RGB for JPEG
        if output_format in ("jpeg", "jpg") and img.mode == "RGBA":
            img = img.convert("RGB")

        save_kwargs: dict[str, Any] = {}

        if output_format == "webp":
            save_kwargs["quality"] = quality
            save_kwargs["lossless"] = lossless
            save_kwargs["method"] = 4  # Balance speed/compression
        elif output_format in ("jpeg", "jpg"):
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif output_format == "png":
            if lossless:
                save_kwargs["optimize"] = True
            else:
                # Use quantization for lossy PNG
                img = img.quantize(colors=256)
                save_kwargs["optimize"] = True

        # Map format names to Pillow format strings
        pillow_format = {
            "webp": "WEBP",
            "jpeg": "JPEG",
            "jpg": "JPEG",
            "png": "PNG",
        }.get(output_format)

        if pillow_format is None:
            raise ValueError(f"Unsupported output format for Pillow: {output_format}")

        img.save(output_path, format=pillow_format, **save_kwargs)

    return Path(output_path).stat().st_size


def compress_image(
    input_path: str,
    output_path: str,
    params: dict[str, Any],
) -> int:
    """Compress an image, trying pyvips first, then Pillow.
    
    Handles both static and animated images.
    
    Args:
        input_path: Path to input image.
        output_path: Path to write compressed image.
        params: Compression parameters.
    
    Returns:
        Output file size in bytes.
    
    Raises:
        RuntimeError: If both pyvips and Pillow fail.
    """
    # Check if image is animated
    if is_animated(input_path):
        logger.info(f"Detected animated image: {input_path}")
        try:
            return compress_animated_image_pillow(input_path, output_path, params)
        except Exception as e:
            raise RuntimeError(f"Failed to compress animated image {input_path}: {e}") from e
    
    # Static image - use normal compression
    try:
        return compress_image_pyvips(input_path, output_path, params)
    except Exception as e:
        logger.debug(f"pyvips failed for {input_path}: {e}, trying alternatives")
        
        output_format = params.get("format", "webp")
        
        # Try CLI tools for JXL and AVIF
        if output_format == "jpeg-xl":
            try:
                return compress_image_jxl_cli(input_path, output_path, params)
            except Exception as e2:
                raise RuntimeError(f"Failed to compress {input_path}: pyvips={e}, cjxl={e2}") from e2
        elif output_format == "avif":
            try:
                return compress_image_avif_cli(input_path, output_path, params)
            except Exception as e2:
                raise RuntimeError(f"Failed to compress {input_path}: pyvips={e}, avifenc={e2}") from e2
        else:
            # Try Pillow for other formats
            try:
                return compress_image_pillow(input_path, output_path, params)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to compress {input_path}: pyvips={e}, Pillow={e2}"
                ) from e2


def process_image(
    input_path: str,
    params: dict[str, Any],
    backup_dir: str | None = None,
) -> CompressionResult:
    """Process a single image file: compress and optionally back up original.

    Args:
        input_path: Path to input image.
        params: Compression parameters.
        backup_dir: Directory to move original file to. If None, original is overwritten.

    Returns:
        CompressionResult with details.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return CompressionResult(
            success=False,
            original_path=input_path,
            error=f"File not found: {input_path}",
        )

    original_size = input_file.stat().st_size
    start_time = time.perf_counter()

    # Determine output format and extension
    output_format = params.get("format", "webp")
    ext_map = {
        "webp": ".webp",
        "jpeg": ".jpg",
        "jpg": ".jpg",
        "png": ".png",
        "jpeg-xl": ".jxl",
        "avif": ".avif",
    }
    output_ext = ext_map.get(output_format, ".webp")

    # Skip if already in target format with same extension
    if input_file.suffix.lower() == output_ext and output_format != "keep":
        # Still compress if format matches but we want to re-encode
        pass

    # Create temporary file for output
    try:
        with tempfile.NamedTemporaryFile(
            suffix=output_ext,
            dir=input_file.parent,
            delete=False,
        ) as tmp:
            tmp_path = tmp.name

        # Compress
        compressed_size = compress_image(input_path, tmp_path, params)
        processing_time = time.perf_counter() - start_time

        # Check if compression actually reduced size
        if compressed_size >= original_size:
            # Compression didn't help, keep original
            Path(tmp_path).unlink(missing_ok=True)
            return CompressionResult(
                success=True,
                original_path=input_path,
                original_size=original_size,
                compressed_size=original_size,
                processing_time=processing_time,
            )

        # Backup original if requested
        if backup_dir:
            backup_path = Path(backup_dir) / input_file.name
            # Handle name conflicts
            counter = 1
            while backup_path.exists():
                backup_path = Path(backup_dir) / f"{input_file.stem}_{counter}{input_file.suffix}"
                counter += 1
            input_file.rename(backup_path)
        else:
            # Remove original
            input_file.unlink()

        # Move compressed file to original location (or new location with new extension)
        if output_format == "keep":
            final_path = input_file
        else:
            final_path = input_file.with_suffix(output_ext)

        Path(tmp_path).rename(final_path)

        return CompressionResult(
            success=True,
            original_path=input_path,
            output_path=str(final_path),
            original_size=original_size,
            compressed_size=compressed_size,
            processing_time=processing_time,
        )

    except Exception as e:
        # Clean up temp file on error
        Path(tmp_path).unlink(missing_ok=True) if 'tmp_path' in dir() else None
        processing_time = time.perf_counter() - start_time
        return CompressionResult(
            success=False,
            original_path=input_path,
            original_size=original_size,
            processing_time=processing_time,
            error=str(e),
        )
