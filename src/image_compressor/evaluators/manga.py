"""Manga/comic-specific quality evaluators.

Focuses on structural line-art preservation rather than
photorealistic fidelity. Key concerns:

  - Edge preservation (Canny + SSIM on binary edges)
  - Screentone/halftone retention (FFT high-frequency energy)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from image_compressor.evaluators.base import QualityEvaluator
from image_compressor.evaluators.io_util import read_gray, read_pair_for_comparison

logger = logging.getLogger(__name__)


def _read_pair_gray_uint8(original: Path, compressed: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read both images as uint8 using fair-comparison pipeline."""
    o, c = read_pair_for_comparison(original, compressed)
    return o.astype(np.uint8), c.astype(np.uint8)


def _read_gray_uint8(path: Path) -> np.ndarray:
    """Read image as grayscale uint8."""
    return read_gray(path).astype(np.uint8)

# ── Optional imports ─────────────────────────────────────────────────────

try:
    import cv2  # type: ignore[import-untyped]
    _HAS_CV2 = True
except ModuleNotFoundError:
    _HAS_CV2 = False
    logger.warning("opencv-python not available — manga evaluators disabled")

try:
    import scipy.fftpack  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except ModuleNotFoundError:
    _HAS_SCIPY = False
    logger.warning("scipy not available — FFT evaluator disabled")


def _crop_same(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


# ── Evaluators ───────────────────────────────────────────────────────────


class EdgePreservationEvaluator(QualityEvaluator):
    """Edge preservation score for line-art / manga.

    Extracts binary edges via Canny, then computes structural similarity
    on the edge maps. This penalises blurring far more than a naive SSIM.

    direction: higher_is_better
    Range:     -1.0 to 1.0 (1.0 = perfect edge preservation)
    """

    name = "edge_preservation"
    direction = "higher_is_better"

    def __init__(
        self,
        *,
        threshold1: int = 100,
        threshold2: int = 200,
    ) -> None:
        self.t1 = threshold1
        self.t2 = threshold2

    def evaluate(self, original: Path, compressed: Path) -> float:
        if not _HAS_CV2:
            logger.warning("edge_preservation unavailable: opencv-python not installed")
            return float("nan")

        try:
            from skimage.metrics import structural_similarity as ssim
        except ModuleNotFoundError:
            logger.warning("edge_preservation needs scikit-image for SSIM")
            return float("nan")

        orig, comp = _read_pair_gray_uint8(original, compressed)
        orig, comp = _crop_same(orig, comp)

        # Extract binary edges
        edge_orig = cv2.Canny(orig, self.t1, self.t2)
        edge_comp = cv2.Canny(comp, self.t1, self.t2)

        # SSIM on edge maps
        score, _ = ssim(edge_orig, edge_comp, full=True)
        return float(score)


class FFTHighFrequencyEvaluator(QualityEvaluator):
    """High-frequency energy retention via 2D FFT.

    Screentones / halftones are periodic high-frequency patterns.
    Lossy codecs often smooth them out ("抹布感"). This evaluator
    measures how much HF energy survives after compression.

    direction: higher_is_better
    Range:     0.0 to 1.0 (1.0 = perfect HF retention), can exceed 1.0
               if compression introduces spurious high frequencies.
    """

    name = "fft_retention"
    direction = "higher_is_better"

    def __init__(self, *, cutoff_ratio: float = 0.2) -> None:
        """Args:
            cutoff_ratio: radius of the low-frequency exclusion circle
                          as a fraction of min(width, height).
        """
        self.cutoff_ratio = cutoff_ratio

    def evaluate(self, original: Path, compressed: Path) -> float:
        if not _HAS_SCIPY or not _HAS_CV2:
            logger.warning("fft_retention unavailable: scipy + opencv-python needed")
            return float("nan")

        import scipy.fftpack

        orig, comp = _read_pair_gray_uint8(original, compressed)
        orig, comp = _crop_same(orig, comp)

        # 2D FFT
        fft_orig = scipy.fftpack.fft2(orig)
        fft_comp = scipy.fftpack.fft2(comp)

        # Shift zero-frequency to centre
        fft_shift_orig = scipy.fftpack.fftshift(fft_orig)
        fft_shift_comp = scipy.fftpack.fftshift(fft_comp)

        # Magnitude spectra
        mag_orig = np.abs(fft_shift_orig)
        mag_comp = np.abs(fft_shift_comp)

        # High-pass mask: exclude centre circle (low frequencies)
        rows, cols = orig.shape
        crow, ccol = rows // 2, cols // 2
        r = int(min(rows, cols) * self.cutoff_ratio)
        mask = np.ones((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (ccol, crow), r, 0, -1)

        hf_energy_orig = np.sum(mag_orig * mask)
        hf_energy_comp = np.sum(mag_comp * mask)

        if hf_energy_orig == 0:
            return 1.0
        ratio = float(hf_energy_comp / hf_energy_orig)
        # 1.0 = perfect preservation.
        # < 1.0 = HF lost (blurring / screentone smoothing).
        # > 1.0 = spurious HF introduced (ringing artifacts).
        # Penalise both directions equally.
        return 1.0 - abs(ratio - 1.0)
