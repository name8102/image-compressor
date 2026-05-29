"""Multi-dimensional image compression benchmark system.

Provides a modular pipeline for evaluating codec performance
across different image categories (photo, manga_bw, manga_color,
animated) using category-specific quality metrics.

Usage:
    from image_compressor.benchmark import BenchmarkPipeline

    pipeline = BenchmarkPipeline()
    pipeline.add_directory(Path("samples/"))
    pipeline.run_all()
    pipeline.export_reports(Path("results/"))
"""

from image_compressor.benchmark.pipeline import BenchmarkPipeline
from image_compressor.benchmark.codec_engine import encode_all, encode_one, load_all_configs
from image_compressor.benchmark.router import MetricsRouter, detect_category
from image_compressor.benchmark.scorer import (
    compute_pareto_frontier,
    compute_ranked_scores,
)

__all__ = [
    "BenchmarkPipeline",
    "MetricsRouter",
    "detect_category",
    "encode_all",
    "encode_one",
    "load_all_configs",
    "compute_ranked_scores",
    "compute_pareto_frontier",
]
