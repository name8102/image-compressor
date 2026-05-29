"""Benchmark pipeline — main orchestrator for the multi-dimensional
image compression benchmark system.

Ties together:
  1. Codec engine   — encode/decode across all formats
  2. Router         — category → evaluator mapping
  3. Evaluators     — quality metrics (traditional, perceptual, manga)
  4. Scorer         — anchor-based ranking + category aggregation + Pareto
  5. Reporter       — CSV, JSON, console reports

Concurrency & Hardware:
  - ProcessPoolExecutor for parallel encode + evaluate
  - RAM disk (TMPDIR=/dev/shm) guidance to prevent NVMe SSD wear
  - CPU thermal throttling awareness via max_workers control
"""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from image_compressor.benchmark.codec_engine import (
    CodecConfig,
    encode_all,
    load_all_configs,
)
from image_compressor.benchmark.router import MetricsRouter, detect_category
from image_compressor.benchmark.scorer import (
    AnchorRanking,
    CategoryAggregate,
    anchor_rank,
    compute_category_aggregates,
    compute_category_pareto,
    compute_images_per_minute,
    compute_throughput,
    default_gate_for_category,
)
from image_compressor.benchmark.reporter import (
    export_csv,
    export_json,
    print_anchor_report,
    print_category_aggregate_report,
    print_pareto_report,
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


# ═══════════════════════════════════════════════════════════════════════════
# Default evaluator registration
# ═══════════════════════════════════════════════════════════════════════════


def _default_registry() -> EvaluatorRegistry:
    registry = EvaluatorRegistry()
    registry.register_class(PSNREvaluator)
    registry.register_class(SSIMEvaluator)
    registry.register_class(MSSSIMEvaluator)
    registry.register_class(SSIMULACRA2Evaluator)
    registry.register_class(ButteraugliEvaluator)
    registry.register_class(EdgePreservationEvaluator)
    registry.register_class(FFTHighFrequencyEvaluator)
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# Per-image encode + evaluate worker (for ProcessPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════

# Module-level registry cache — evaluators are expensive to re-create per worker
_registry: EvaluatorRegistry | None = None
_router: MetricsRouter | None = None


def _init_worker() -> None:
    """Initialise module-level objects in each worker process."""
    global _registry, _router
    if _registry is None:
        _registry = _default_registry()
    if _router is None:
        _router = MetricsRouter()


def _run_single_image(
    payload: tuple[Path, str, list[CodecConfig], Path | None, bool],
) -> tuple[Path, str, list[EncodeResult]]:
    """Worker function: encode + evaluate one image.

    Args:
        payload: (img_path, category, configs, output_dir, clean_outputs)

    Returns:
        (img_path, category, results)
    """
    _init_worker()
    img_path, cat, configs, output_dir, clean = payload

    # Encode
    image_output_dir = output_dir / img_path.stem if output_dir else None
    results = encode_all(img_path, configs, output_dir=image_output_dir)
    if not results:
        return img_path, cat, []

    # Evaluate
    metrics = _router.get_metrics(cat)  # type: ignore[union-attr]
    for metric_name, weight, higher in metrics:
        try:
            evaluator = _registry.get(metric_name)  # type: ignore[union-attr]
        except KeyError:
            logger.warning("  Evaluator not found: %s — skipping", metric_name)
            continue
        for r in results:
            try:
                score = evaluator.evaluate(img_path, r.output_path)
                r.add_score(metric_name, score)
            except Exception as e:
                logger.warning("  %s on %s: %s", metric_name, r.config_id, e)
                r.add_score(metric_name, float("nan"))

    # Clean up compressed outputs
    if clean:
        for r in results:
            _safe_unlink(r.output_path)
        _safe_rmdir(image_output_dir)

    return img_path, cat, results


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class BenchmarkPipeline:
    """Main benchmark orchestrator.

    Features:
      - ProcessPoolExecutor for parallel encode + evaluate across images
      - Anchor-based ranking (quality gate + size ordering)
      - Category-level aggregation and Pareto-frontier analysis
      - Throughput metrics (GB_saved_per_hour)
      - RAM disk guidance for NVMe SSD protection

    Usage::

        pipeline = BenchmarkPipeline(output_dir="/dev/shm/bench_out")
        pipeline.add_directory(Path("samples/"))
        pipeline.run_all(max_workers=4)
        pipeline.export_reports(Path("results/"))
    """

    def __init__(
        self,
        *,
        config_path: Path | str | None = None,
        codec_configs: list[CodecConfig] | None = None,
        output_dir: Path | str | None = None,
        clean_outputs: bool = True,
        use_ramdisk: bool = True,
    ) -> None:
        """
        Args:
            config_path: Path to metrics_routing.yaml.
            codec_configs: Custom codec config matrix.
            output_dir: Directory for temporary compressed outputs.
                Set to a RAM disk path (/dev/shm/bench_out on Linux)
                to avoid NVMe SSD write wear from thousands of temp files.
            clean_outputs: Delete compressed outputs after evaluation.
            use_ramdisk: When True and output_dir is not explicitly set,
                auto-select /dev/shm on Linux.
        """
        self.router = MetricsRouter(config_path)
        self.codec_configs = codec_configs or load_all_configs()

        # ── RAM disk protection ──
        # NVMe TBW (Total Bytes Written) is finite. Running thousands of
        # encode+decode cycles generates massive temp file I/O. Always
        # prefer a RAM disk when available.
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif use_ramdisk and Path("/dev/shm").is_dir():
            self.output_dir = Path("/dev/shm/bench_out")
            logger.info("Using RAM disk: %s (protects NVMe from write wear)", self.output_dir)
        else:
            self.output_dir = Path("benchmark_results")

        self.clean_outputs = clean_outputs

        # Collected data
        self._image_paths: list[Path] = []
        self._categories: list[str] = []
        self._all_results: list[list[EncodeResult]] = []
        self._all_anchors: list[AnchorRanking] = []
        self._category_aggregates: dict[str, list[CategoryAggregate]] = {}

    # ── Image registration ───────────────────────────────────────────────

    def add_image(self, path: Path, category: str | None = None) -> None:
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
        pattern = "**/*" if recursive else "*"
        added = 0
        for f in sorted(directory.glob(pattern)):
            if f.suffix.lower() in extensions and f.is_file():
                self.add_image(f, category=category)
                added += 1
        return added

    # ── Core pipeline ────────────────────────────────────────────────────

    def run_all(
        self,
        *,
        max_workers: int | None = None,
        use_anchor_ranking: bool = True,
    ) -> None:
        """Execute the full pipeline for all registered images.

        Args:
            max_workers: Max parallel processes for encoding + evaluating.
                         None = auto-detect (os.cpu_count() - 1).
                         Control this to avoid CPU thermal throttling:
                           - For accurate per-image encode_time: set low (1-2)
                           - For system throughput measurement: set high (all cores)
            use_anchor_ranking: Use anchor-based ranking (default).
                                Set False for legacy weighted scoring.
        """
        total = len(self._image_paths)
        if total == 0:
            logger.warning("No images registered. Use add_image() or add_directory().")
            return

        # ── TMPDIR guidance ──
        # Ensure subprocesses (cjxl, avifenc, etc.) also use the RAM disk
        # for their internal temp files when available.
        tmpdir = str(self.output_dir / "tmp")
        os.makedirs(tmpdir, exist_ok=True)

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 4) - 1)

        logger.info(
            "Starting benchmark: %d image(s) × %d codec config(s) | %d workers | output=%s",
            total, len(self.codec_configs), max_workers, self.output_dir,
        )

        self._all_results = []
        self._all_anchors = []
        self._category_aggregates = {}

        # ── Phase 1 & 2: Parallel encode + evaluate ──
        payloads = [
            (img_path, cat, self.codec_configs, self.output_dir, self.clean_outputs)
            for img_path, cat in zip(self._image_paths, self._categories)
        ]

        env = os.environ.copy()
        env.setdefault("TMPDIR", tmpdir)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
        ) as executor:
            # Submit in batches to avoid overloading
            futures = {
                executor.submit(_run_single_image, p): i
                for i, p in enumerate(payloads)
            }

            # Pre-allocate results list (preserving order)
            ordered: list[tuple[Path, str, list[EncodeResult]]] = [
                (Path(), "", []) for _ in range(total)
            ]

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    img_path, cat, results = future.result()
                    ordered[idx] = (img_path, cat, results)
                except Exception as e:
                    logger.error("Worker failed for image %d: %s", idx, e)
                    img_path = self._image_paths[idx]
                    ordered[idx] = (img_path, self._categories[idx], [])

        for img_path, cat, results in ordered:
            self._all_results.append(results)

            # ── Phase 3: Per-image ranking ──
            if use_anchor_ranking and results:
                gate = default_gate_for_category(cat)
                anchor = anchor_rank(
                    results,
                    gate_metric=gate[0],
                    gate_threshold=gate[1],
                    gate_higher_is_better=gate[2],
                    image_name=img_path.name,
                    category=cat,
                )
                self._all_anchors.append(anchor)
            elif results:
                self._all_anchors.append(AnchorRanking(
                    image_name=img_path.name, category=cat,
                    gate_metric="none", gate_threshold=0.0,
                    gate_higher_is_better=True,
                ))

            # ── Per-image report ──
            if use_anchor_ranking and results and self._all_anchors:
                print_anchor_report(self._all_anchors[-1])
            elif not use_anchor_ranking and results:
                metrics = self.router.get_metrics(cat)
                from image_compressor.benchmark.scorer import compute_ranked_scores
                ranked = compute_ranked_scores(results, metrics, self.router.ranking_weights)
                print_per_image_report(img_path, cat, ranked)

        # ── Phase 4: Category-level aggregation ──
        self._category_aggregates = compute_category_aggregates(
            self._all_results, self._categories,
        )

        # ── Phase 5: Category-level Pareto frontier ──
        for cat, aggs in self._category_aggregates.items():
            gate = default_gate_for_category(cat)
            pareto_indices = compute_category_pareto(
                aggs, gate[0], higher_is_better=gate[2],
            )
            pareto_configs = [aggs[i] for i in pareto_indices]
            print_pareto_report(cat, aggs, pareto_configs, gate)
            print_category_aggregate_report(cat, aggs)

        # ── Summary report ──
        if not use_anchor_ranking:
            from image_compressor.benchmark.scorer import compute_ranked_scores
            ranked_all = []
            for results, cat in zip(self._all_results, self._categories):
                if results:
                    metrics = self.router.get_metrics(cat)
                    ranked_all.append(
                        compute_ranked_scores(results, metrics, self.router.ranking_weights)
                    )
                else:
                    ranked_all.append([])
            print_summary_report(ranked_all, self._image_paths, self._categories)
        else:
            _print_anchor_summary(self._all_anchors, self._category_aggregates)

        # ── Cleanup ──
        cleanup_cache()
        if self.clean_outputs:
            _safe_rmdir(self.output_dir / "tmp")

    # ── Binary search for optimal quality parameter ──────────────────────

    def binary_search_quality(
        self,
        image_path: Path,
        *,
        format: str = "webp",
        target_quality_metric: str = "ssimulacra2",
        target_threshold: float = 75.0,
        higher_is_better: bool = False,
        quality_min: int = 1,
        quality_max: int = 100,
        tolerance: float = 3.0,
        max_iterations: int = 10,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Binary-search the optimal quality parameter for a given target.

        Replaces hard-coded quality grids (Q=30,40,50,60...) with an
        adaptive search.  For TB-scale data, this can find the exact
        quality setting that just barely meets the quality gate — saving
        potentially hundreds of GB compared to a coarse grid.

        Args:
            image_path: Single representative image for the category.
            format: Target format (webp, jxl, avif, jpeg).
            target_quality_metric: Metric to gate on.
            target_threshold: Target quality value.
            higher_is_better: Direction of the quality metric.
            quality_min: Lower bound for binary search.
            quality_max: Upper bound for binary search.
            tolerance: Acceptable deviation from target.
            max_iterations: Safety limit.
            category: Override auto-detected category.

        Returns:
            Dict with optimal_quality, achieved_score, iterations, history.
        """
        cat = category or detect_category(image_path)
        history: list[dict[str, Any]] = []

        lo, hi = quality_min, quality_max
        best_q = quality_min
        best_score: float | None = None
        iterations = 0

        while lo <= hi and iterations < max_iterations:
            iterations += 1
            mid = (lo + hi) // 2

            config: CodecConfig = {
                "format": format, "lossy": True, "quality": mid,
            }
            results = encode_all(image_path, [config], output_dir=self.output_dir)
            if not results:
                logger.warning("  Encode failed at Q=%d", mid)
                # Try lower quality
                hi = mid - 1
                continue

            result = results[0]

            # Evaluate quality
            metrics = self.router.get_metrics(cat)
            for metric_name, _, _ in metrics:
                try:
                    evaluator = _default_registry().get(metric_name)
                    score = evaluator.evaluate(image_path, result.output_path)
                    result.add_score(metric_name, score)
                except Exception:
                    result.add_score(metric_name, float("nan"))

            # Cleanup temp
            if self.clean_outputs:
                _safe_unlink(result.output_path)

            score_val = result.scores.get(target_quality_metric, float("nan"))
            if score_val != score_val:
                logger.warning("  Metric %s unavailable at Q=%d", target_quality_metric, mid)
                hi = mid - 1
                continue

            entry = {
                "quality": mid,
                "score": score_val,
                "compression_ratio": result.compression_ratio,
                "encode_time": result.encode_time,
            }
            history.append(entry)

            # Check if we're within tolerance
            if higher_is_better:
                passes = score_val >= target_threshold
                delta = score_val - target_threshold
            else:
                passes = score_val <= target_threshold
                delta = target_threshold - score_val

            logger.debug(
                "  Q=%d  %s=%.3f  delta=%.3f  passes=%s",
                mid, target_quality_metric, score_val, delta, passes,
            )

            best_q = mid
            best_score = score_val

            if abs(delta) <= tolerance:
                break

            if passes:
                # Quality already good enough — try lower quality (smaller file)
                hi = mid - 1
            else:
                # Quality not good enough — need higher quality
                lo = mid + 1

        # Cleanup cache
        cleanup_cache()

        return {
            "optimal_quality": best_q,
            "achieved_score": best_score,
            "format": format,
            "target_metric": target_quality_metric,
            "target_threshold": target_threshold,
            "tolerance": tolerance,
            "iterations": iterations,
            "history": history,
        }

    # ── Export ────────────────────────────────────────────────────────────

    def export_reports(self, output_dir: Path | str | None = None) -> dict[str, Path]:
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "benchmark_results.csv"
        export_csv(self._all_results, self._image_paths, csv_path)

        json_path = out / "benchmark_results.json"
        from image_compressor.benchmark.scorer import compute_ranked_scores
        ranked_all = []
        for results, cat in zip(self._all_results, self._categories):
            if results:
                metrics = self.router.get_metrics(cat)
                ranked_all.append(
                    compute_ranked_scores(results, metrics, self.router.ranking_weights)
                )
            else:
                ranked_all.append([])

        export_json(
            self._all_results, self._image_paths,
            ranked_all, self._categories, json_path,
        )

        return {"csv": csv_path, "json": json_path}

    # ── Category aggregates accessor ─────────────────────────────────────

    @property
    def category_aggregates(self) -> dict[str, list[CategoryAggregate]]:
        return self._category_aggregates

    # ── Pareto frontier ──────────────────────────────────────────────────

    def pareto_frontier(
        self,
        image_index: int = 0,
        y_metric: str | None = None,
    ) -> list[EncodeResult]:
        from image_compressor.benchmark.scorer import compute_pareto_frontier
        if image_index >= len(self._all_results):
            return []
        results = self._all_results[image_index]
        indices = compute_pareto_frontier(results, y_metric=y_metric)
        return [results[i] for i in indices]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_rmdir(path: Path) -> None:
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except Exception:
        pass


def _print_anchor_summary(
    anchors: list[AnchorRanking],
    aggregates: dict[str, list[CategoryAggregate]],
) -> None:
    """Print a summary of anchor-based ranking results."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Anchor Ranking — Summary", show_header=True, header_style="bold yellow")
        table.add_column("Image", width=28)
        table.add_column("Cat", width=10)
        table.add_column("Gate", width=20)
        table.add_column("Best Config", width=18)
        table.add_column("Savings%", justify="right", width=8)
        table.add_column("Size", justify="right", width=10)

        for a in anchors:
            if a.best:
                b = a.best
                gate_str = f"{a.gate_metric} ≤ {a.gate_threshold}"
                if a.gate_higher_is_better:
                    gate_str = f"{a.gate_metric} ≥ {a.gate_threshold}"
                table.add_row(
                    a.image_name[:26], a.category,
                    gate_str, b["config_id"],
                    f"{b['savings_pct']:.1f}%",
                    f"{b['compressed_size']:,}",
                )
            else:
                table.add_row(
                    a.image_name[:26], a.category, "—", "—", "—", "—",
                )
        console.print(table)
    except ModuleNotFoundError:
        print("\nAnchor Ranking — Summary")
        for a in anchors:
            if a.best:
                b = a.best
                print(f"  {a.image_name:<28} {a.category:<10} {b['config_id']:<18} "
                      f"savings={b['savings_pct']:.1f}%  size={b['compressed_size']:,}")
