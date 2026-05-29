"""File discovery and scanning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Any

# Supported image extensions (lowercase)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff", ".tif",
    ".gif",
}

# Supported archive extensions
ARCHIVE_EXTENSIONS = {
    ".cbz",
    ".cb7",
}


def get_image_info(path: Path | str) -> dict[str, Any]:
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


def scan_directory(
    path: Path | str,
    recursive: bool = True,
) -> Generator[dict[str, str | int | bool], None, None]:
    """Scan a directory for images and CBZ archives.
    
    Args:
        path: Directory path to scan.
        recursive: Whether to scan subdirectories.
    
    Yields:
        Dictionary with keys: path, type (image/archive), size, extension, animated.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    walker = os.walk(root) if recursive else [(str(root), [], os.listdir(root))]

    for dirpath, _dirnames, filenames in walker:
        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            if ext in IMAGE_EXTENSIONS:
                try:
                    size = filepath.stat().st_size
                    
                    # Check if animated
                    image_info = get_image_info(filepath)
                    animated = image_info.get("animated", False)
                    frames = image_info.get("frames", 1)
                    
                    yield {
                        "path": str(filepath),
                        "type": "animated_image" if animated else "image",
                        "size": size,
                        "extension": ext,
                        "animated": animated,
                        "frames": frames,
                        "format": image_info.get("format", "unknown"),
                    }
                except OSError:
                    continue
            elif ext in ARCHIVE_EXTENSIONS:
                try:
                    size = filepath.stat().st_size
                except OSError:
                    continue
                yield {
                    "path": str(filepath),
                    "type": "archive",
                    "size": size,
                    "extension": ext,
                    "animated": False,
                    "frames": 1,
                }


def scan_paths(
    paths: list[str],
    recursive: bool = True,
) -> list[dict[str, str | int]]:
    """Scan multiple paths for images and archives.

    Args:
        paths: List of directory/file paths to scan.
        recursive: Whether to scan subdirectories.

    Returns:
        List of file dictionaries.
    """
    results: list[dict[str, str | int]] = []

    for p in paths:
        path = Path(p)
        if path.is_file():
            ext = path.suffix.lower()
            if ext in IMAGE_EXTENSIONS or ext in ARCHIVE_EXTENSIONS:
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                results.append({
                    "path": str(path),
                    "type": "archive" if ext in ARCHIVE_EXTENSIONS else "image",
                    "size": size,
                    "extension": ext,
                })
        elif path.is_dir():
            results.extend(scan_directory(path, recursive=recursive))

    return results


def get_image_extension(path: Path | str) -> str | None:
    """Get the image extension if it's a supported image format.

    Args:
        path: File path.

    Returns:
        Extension string (e.g., '.jpg') or None if not a supported image.
    """
    ext = Path(path).suffix.lower()
    return ext if ext in IMAGE_EXTENSIONS else None


def is_archive(path: Path | str) -> bool:
    """Check if a file is a supported archive format.

    Args:
        path: File path.

    Returns:
        True if the file is a supported archive.
    """
    return Path(path).suffix.lower() in ARCHIVE_EXTENSIONS
