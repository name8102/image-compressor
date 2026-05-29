"""CBZ archive handling - unpack, compress, repack."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compressor import compress_image
from .scanner import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class CBZResult:
    """Result of processing a CBZ archive."""

    success: bool
    cbz_path: str
    entry_count: int = 0
    compressed_count: int = 0
    original_size: int = 0
    compressed_size: int = 0
    processing_time: float = 0.0
    error: str | None = None


def list_cbz_entries(cbz_path: str) -> list[dict[str, Any]]:
    """List image entries in a CBZ archive.

    Args:
        cbz_path: Path to CBZ file.

    Returns:
        List of entry dictionaries with keys: name, size, extension.
    """
    entries = []
    with zipfile.ZipFile(cbz_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                entries.append({
                    "name": info.filename,
                    "size": info.file_size,
                    "extension": ext,
                })
    return entries


def extract_cbz(cbz_path: str, output_dir: str) -> list[str]:
    """Extract CBZ archive to a directory.

    Args:
        cbz_path: Path to CBZ file.
        output_dir: Directory to extract to.

    Returns:
        List of extracted file paths.
    """
    extracted = []
    with zipfile.ZipFile(cbz_path, "r") as zf:
        zf.extractall(output_dir)
        for info in zf.infolist():
            if not info.is_dir():
                extracted.append(str(Path(output_dir) / info.filename))
    return extracted


def repack_cbz(
    source_dir: str,
    output_path: str,
    compression: int = zipfile.ZIP_STORED,
) -> int:
    """Repack a directory into a CBZ archive.

    Args:
        source_dir: Directory containing images.
        output_path: Path to write CBZ file.
        compression: ZIP compression method (default: ZIP_STORED).

    Returns:
        Output file size in bytes.
    """
    source = Path(source_dir)
    with zipfile.ZipFile(output_path, "w", compression=compression) as zf:
        for file_path in sorted(source.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source)
                zf.write(file_path, str(arcname))

    return Path(output_path).stat().st_size


def process_cbz(
    cbz_path: str,
    params: dict[str, Any],
    backup_dir: str | None = None,
) -> CBZResult:
    """Process a CBZ archive: extract, compress images, repack.

    Args:
        cbz_path: Path to CBZ file.
        params: Compression parameters for images.
        backup_dir: Directory to backup original CBZ.

    Returns:
        CBZResult with details.
    """
    cbz_file = Path(cbz_path)
    if not cbz_file.exists():
        return CBZResult(
            success=False,
            cbz_path=cbz_path,
            error=f"File not found: {cbz_path}",
        )

    original_size = cbz_file.stat().st_size
    start_time = time.perf_counter()

    try:
        # List entries first
        entries = list_cbz_entries(cbz_path)
        if not entries:
            return CBZResult(
                success=True,
                cbz_path=cbz_path,
                entry_count=0,
                original_size=original_size,
                processing_time=time.perf_counter() - start_time,
            )

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory(prefix="image_compressor_") as tmp_dir:
            # Extract
            extract_dir = Path(tmp_dir) / "extracted"
            extract_dir.mkdir()
            extracted_files = extract_cbz(cbz_path, str(extract_dir))

            # Compress each image
            compressed_count = 0
            for file_path in extracted_files:
                file_ext = Path(file_path).suffix.lower()
                if file_ext not in IMAGE_EXTENSIONS:
                    continue

                try:
                    compress_image(file_path, file_path, params)
                    compressed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to compress {file_path} in {cbz_path}: {e}")

            # Repack
            repack_path = Path(tmp_dir) / cbz_file.name
            repack_cbz(str(extract_dir), str(repack_path), compression=zipfile.ZIP_STORED)
            compressed_size = repack_path.stat().st_size

            # If repacked is larger, keep original
            if compressed_size >= original_size:
                processing_time = time.perf_counter() - start_time
                return CBZResult(
                    success=True,
                    cbz_path=cbz_path,
                    entry_count=len(entries),
                    compressed_count=compressed_count,
                    original_size=original_size,
                    compressed_size=original_size,
                    processing_time=processing_time,
                )

            # Backup original if requested
            if backup_dir:
                backup_path = Path(backup_dir) / cbz_file.name
                counter = 1
                while backup_path.exists():
                    backup_path = Path(backup_dir) / f"{cbz_file.stem}_{counter}{cbz_file.suffix}"
                    counter += 1
                shutil.copy2(cbz_path, str(backup_path))

            # Replace original with compressed version
            shutil.move(str(repack_path), cbz_path)

            processing_time = time.perf_counter() - start_time
            return CBZResult(
                success=True,
                cbz_path=cbz_path,
                entry_count=len(entries),
                compressed_count=compressed_count,
                original_size=original_size,
                compressed_size=compressed_size,
                processing_time=processing_time,
            )

    except Exception as e:
        processing_time = time.perf_counter() - start_time
        return CBZResult(
            success=False,
            cbz_path=cbz_path,
            original_size=original_size,
            processing_time=processing_time,
            error=str(e),
        )
