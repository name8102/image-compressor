"""Image quality evaluators — shared between compressor and benchmark.

Evaluators follow a common QualityEvaluator interface and are
grouped by domain:

    traditional   — PSNR, SSIM, MS-SSIM
    perceptual    — SSIMULACRA2, Butteraugli
    manga         — EdgePreservation, FFTHighFrequency
"""

from image_compressor.evaluators.base import (
    EncodeResult,
    EvaluatorRegistry,
    QualityEvaluator,
)
from image_compressor.evaluators.traditional import (
    MSSSIMEvaluator,
    PSNREvaluator,
    SSIMEvaluator,
)
from image_compressor.evaluators.perceptual import (
    ButteraugliEvaluator,
    SSIMULACRA2Evaluator,
)
from image_compressor.evaluators.manga import (
    EdgePreservationEvaluator,
    FFTHighFrequencyEvaluator,
)
from image_compressor.evaluators.io_util import cleanup_cache

__all__ = [
    # base
    "QualityEvaluator",
    "EncodeResult",
    "EvaluatorRegistry",
    # traditional
    "PSNREvaluator",
    "SSIMEvaluator",
    "MSSSIMEvaluator",
    # perceptual
    "SSIMULACRA2Evaluator",
    "ButteraugliEvaluator",
    # manga
    "EdgePreservationEvaluator",
    "FFTHighFrequencyEvaluator",
]
