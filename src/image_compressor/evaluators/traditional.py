"""Traditional quality metrics: PSNR, SSIM, MS-SSIM.

Uses scikit-image for SSIM/MS-SSIM and a manual PSNR implementation
to avoid scikit-image's multi-channel handling quirks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from image_compressor.evaluators.base import QualityEvaluator
from image_compressor.evaluators.io_util import read_gray, read_pair_for_comparison

logger = logging.getLogger(__name__)

# ── Optional imports ─────────────────────────────────────────────────────

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SKIMAGE = True
except ModuleNotFoundError:
    _HAS_SKIMAGE = False
    logger.warning("scikit-image not available — SSIM/MS-SSIM evaluators disabled")


# ── Evaluators ───────────────────────────────────────────────────────────


class PSNREvaluator(QualityEvaluator):
    """Peak Signal-to-Noise Ratio.

    direction: higher_is_better
    Range:      ~20 (poor) to ~50+ (near-lossless)
    """

    name = "psnr"
    direction = "higher_is_better"

    def evaluate(self, original: Path, compressed: Path) -> float:
        orig, comp = read_pair_for_comparison(original, compressed)

        # Ensure same dimensions (compressed may be padded)
        h = min(orig.shape[0], comp.shape[0])
        w = min(orig.shape[1], comp.shape[1])
        orig = orig[:h, :w]
        comp = comp[:h, :w]

        mse = np.mean((orig - comp) ** 2)
        if mse == 0:
            return 100.0  # identical
        return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


class SSIMEvaluator(QualityEvaluator):
    """Structural Similarity Index (single-scale).

    direction: higher_is_better
    Range:     -1.0 to 1.0 (1.0 = identical)
    """

    name = "ssim"
    direction = "higher_is_better"

    def __init__(self, *, crop_to_match: bool = True) -> None:
        self._crop = crop_to_match

    def evaluate(self, original: Path, compressed: Path) -> float:
        if not _HAS_SKIMAGE:
            logger.warning("SSIM unavailable: scikit-image not installed")
            return float("nan")

        orig, comp = read_pair_for_comparison(original, compressed)

        if self._crop:
            h = min(orig.shape[0], comp.shape[0])
            w = min(orig.shape[1], comp.shape[1])
            orig = orig[:h, :w]
            comp = comp[:h, :w]

        score, _ = _ssim(orig, comp, full=True, data_range=255)
        return float(score)


class MSSSIMEvaluator(QualityEvaluator):
    """Multi-Scale Structural Similarity Index.

    More robust than single-scale SSIM for images with
    varying viewing distances (photographs, complex scenes).

    direction: higher_is_better
    Range:     0.0 to 1.0
    """

    name = "ms_ssim"
    direction = "higher_is_better"

    def evaluate(self, original: Path, compressed: Path) -> float:
        if not _HAS_SKIMAGE:
            logger.warning("MS-SSIM unavailable: scikit-image not installed")
            return float("nan")

        orig, comp = read_pair_for_comparison(original, compressed)

        h = min(orig.shape[0], comp.shape[0])
        w = min(orig.shape[1], comp.shape[1])
        orig = orig[:h, :w]
        comp = comp[:h, :w]

        from skimage.metrics import structural_similarity as ssim

        # MS-SSIM: compute SSIM across multiple downsampled scales
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levels = len(weights)
        mssim = []
        mcs = []

        for _ in range(levels):
            s, c = ssim(orig, comp, full=True, data_range=255)
            mssim.append(s)
            mcs.append(c)
            # Downsample for next level
            orig = orig[::2, ::2]
            comp = comp[::2, ::2]

        # Overall MS-SSIM = product of weighted similarity components
        overall = np.prod([mcs[i] ** weights[i] for i in range(levels - 1)])
        overall *= mssim[-1] ** weights[-1]
        return float(overall)
