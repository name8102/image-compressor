"""Benchmark pipeline — main orchestrator for the multi-dimensional
image compression benchmark system.

Ties together:
  1. Codec engine   — encode/decode across all formats
  2. Router         — category → evaluator mapping
  3. Evaluators     — quality metrics (traditional, perceptual, manga)
  4. Scorer         — normalization + weighted ranking
  5. Reporter       — CSV, JSON, console reports
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from image_compressor.benchmark.codec_engine import (
    CodecConfig,
    encode_all,
    load_all_configs,
)
from image_compressor.benchmark.router import MetricsRouter, detect_category
from image_compressor.benchmark.scorer import (
    compute_pareto_frontier,
    compute_ranked_scores,
)
from image_compressor.benchmark.reporter import (
    export_csv,
    export_json,
    print_per_image_report,
    print_summary_report,
)
from image_compressor.evaluators.base import (
    EncodeResult,
    EvaluatorRegistry,
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

logger = logging.getLogger(__name__)


# ── Default evaluator registration ──────────────────────────────────────


def _default_registry() -> EvaluatorRegistry:
    registry = EvaluatorRegistry()
    # Traditional
    registry.register_class(PSNREvaluator)
    registry.register_class(SSIMEvaluator)
    registry.register_class(MSSSIMEvaluator)
    # Perceptual
    registry.register_class(SSIMULACRA2Evaluator)
    registry.register_class(ButteraugliEvaluator)
    # Manga
    registry.register_class(EdgePreservationEvaluator)
    registry.register_class(FFTHighFrequencyEvaluator)
    return registry


# ── Pipeline ────────────────────────────────────────────────────────────


class BenchmarkPipeline:
    """Main benchmark orchestrator.

    Usage::

        pipeline = BenchmarkPipeline()
        pipeline.add_image(Path("sample.png"))
        pipeline.add_directory(Path("samples/"))
        pipeline.run_all()
        pipeline.export_reports(Path("results/"))
    """

    def __init__(
        self,
        *,
        config_path: Path | str | None = None,
        codec_configs: list[CodecConfig] | None = None,
        output_dir: Path | str | None = None,
        clean_outputs: bool = True,
    ) -> None:
        self.router = MetricsRouter(config_path)
        self.registry = _default_registry()
        self.codec_configs = codec_configs or load_all_configs()
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.clean_outputs = clean_outputs

        # Collected data
        self._image_paths: list[Path] = []
        self._categories: list[str] = []
        self._all_results: list[list[EncodeResult]] = []
        self._all_ranked: list[list[dict[str, Any]]] = []

    # ── Image registration ───────────────────────────────────────────────

    def add_image(self, path: Path, category: str | None = None) -> None:
        """Register a single image for benchmarking.

        Args:
            path: Image file path.
            category: Override auto-detected category (photo, manga_bw, etc.).
        """
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        self._image_paths.append(path.resolve())
        self._categories.append(category or detect_category(path))
        logger.info("Registered: %s  →  %s", path.name, self._categories[-1])

    def add_directory(
        self,
        directory: Path,
        *,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif"),
        recursive: bool = True,
        category: str | None = None,
    ) -> int:
        """Register all images in a directory.

        Returns count of images added.
        """
        pattern = "**/*" if recursive else "*"
        added = 0
        for f in sorted(directory.glob(pattern)):
            if f.suffix.lower() in extensions and f.is_file():
                self.add_image(f, category=category)
                added += 1
        return added

    # ── Core pipeline ────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Execute the full pipeline for all registered images."""
        total = len(self._image_paths)
        if total == 0:
            logger.warning("No images registered. Use add_image() or add_directory().")
            return

        logger.info("Starting benchmark: %d image(s) × %d codec config(s)",
                     total, len(self.codec_configs))

        self._all_results = []
        self._all_ranked = []

        for idx, (img_path, cat) in enumerate(
            zip(self._image_paths, self._categories), start=1
        ):
            print(f"\n{'=' * 60}")
            print(f"[{idx}/{total}] {img_path.name}  ({cat})")
            print(f"{'=' * 60}")

            # ── Phase 1: Encode across all codec configs ──
            image_output_dir = self.output_dir / img_path.stem
            results = encode_all(
                img_path,
                self.codec_configs,
                output_dir=image_output_dir,
            )
            if not results:
                logger.warning("  No successful encodings for %s", img_path.name)
                self._all_results.append([])
                self._all_ranked.append([])
                continue

            # ── Phase 2: Evaluate quality metrics ──
            metrics = self.router.get_metrics(cat)
            metric_names = [m[0] for m in metrics]
            print(f"  Metrics: {metric_names}")

            for metric_name, weight, higher in metrics:
                try:
                    evaluator = self.registry.get(metric_name)
                except KeyError:
                    logger.warning("  Evaluator not found: %s — skipping", metric_name)
                    continue
                for r in results:
                    try:
                        score = evaluator.evaluate(img_path, r.output_path)
                        r.add_score(metric_name, score)
                    except Exception as e:
                        logger.warning("  %s failed on %s: %s", metric_name, r.config_id, e)
                        r.add_score(metric_name, float("nan"))

            # ── Phase 3: Score & rank ──
            ranked = compute_ranked_scores(
                results,
                metrics,
                ranking_weights=self.router.ranking_weights,
            )

            self._all_results.append(results)
            self._all_ranked.append(ranked)

            # ── Phase 4: Per-image report ──
            print_per_image_report(img_path, cat, ranked)

            # Clean up compressed outputs if requested
            if self.clean_outputs:
                for r in results:
                    _safe_unlink(r.output_path)
                _safe_rmdir(image_output_dir)

        # ── Summary report ──
        print_summary_report(self._all_ranked, self._image_paths, self._categories)

        # ── Clean up temp decode cache ──
        cleanup_cache()

    # ── Export ────────────────────────────────────────────────────────────

    def export_reports(self, output_dir: Path | str | None = None) -> dict[str, Path]:
        """Export CSV and JSON reports.

        Returns dict of format → output path.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "benchmark_results.csv"
        export_csv(self._all_results, self._image_paths, csv_path)

        json_path = out / "benchmark_results.json"
        export_json(
            self._all_results,
            self._image_paths,
            self._all_ranked,
            self._categories,
            json_path,
        )

        return {"csv": csv_path, "json": json_path}

    # ── Pareto frontier ──────────────────────────────────────────────────

    def pareto_frontier(
        self,
        image_index: int = 0,
        y_metric: str | None = None,
    ) -> list[EncodeResult]:
        """Get Pareto-optimal results for a specific image."""
        if image_index >= len(self._all_results):
            return []
        results = self._all_results[image_index]
        indices = compute_pareto_frontier(results, y_metric=y_metric)
        return [results[i] for i in indices]


# ── Helpers ──────────────────────────────────────────────────────────────


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_rmdir(path: Path) -> None:
    try:
        if path.is_dir():
            path.rmdir()
    except Exception:
        pass
