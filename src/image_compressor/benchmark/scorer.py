"""Score normalization, anchor-based ranking, category aggregation, and
Pareto-frontier analysis for the multi-dimensional benchmark system.

Design principles (post-refactor):
  1. Anchor-based ranking — quality gate, then size-ordered ranking.
     No more fuzzy linear-weighted total_score.
  2. Category-level aggregation — compute mean quality and mean compression
     ratio per config across all images in a category.
  3. Category-level Pareto frontier — draw multi-config Pareto on the
     category-averaged size-vs-quality plane.
  4. Throughput metric — GB_saved_per_hour for production ROI estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from image_compressor.evaluators.base import EncodeResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Anchor-based ranking — quality gate + size ordering
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AnchorRanking:
    """Result of anchor-based ranking for a single image.

    Instead of a fuzzy weighted total_score, we apply a quality gate
    (e.g. SSIMULACRA2 <= 75) and rank passing configs by compression ratio
    (smaller = better).  Configs that fail the gate are listed separately.
    """

    image_name: str
    category: str
    gate_metric: str
    gate_threshold: float
    gate_higher_is_better: bool
    passing: list[dict[str, Any]] = field(default_factory=list)
    failing: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best(self) -> dict[str, Any] | None:
        return self.passing[0] if self.passing else None


def anchor_rank(
    results: list[EncodeResult],
    *,
    gate_metric: str = "ssimulacra2",
    gate_threshold: float = 75.0,
    gate_higher_is_better: bool = False,
    image_name: str = "",
    category: str = "",
) -> AnchorRanking:
    """Rank configs by anchor-based criteria.

    1. Filter by quality gate (gate_metric must pass gate_threshold).
    2. Sort passing by compression_ratio ascending (smallest size first).
    3. Report failing configs separately.

    Args:
        results: All EncodeResult for one image.
        gate_metric: Quality metric name for the gate.
        gate_threshold: Minimum acceptable quality.
        gate_higher_is_better: If True, score >= threshold passes.
                               If False, score <= threshold passes.
        image_name: Image filename for reporting.
        category: Image category label.

    Returns:
        AnchorRanking with passing and failing configs.
    """
    ranking = AnchorRanking(
        image_name=image_name,
        category=category,
        gate_metric=gate_metric,
        gate_threshold=gate_threshold,
        gate_higher_is_better=gate_higher_is_better,
    )

    for r in results:
        score = r.scores.get(gate_metric)
        entry = {
            "config_id": r.config_id,
            "format": r.format,
            "compression_ratio": r.compression_ratio,
            "savings_pct": r.savings_pct,
            "compressed_size": r.compressed_size,
            "original_size": r.original_size,
            "encode_time": r.encode_time,
            "scores": dict(r.scores),
        }

        if score is None or (isinstance(score, float) and score != score):
            # Missing/NaN score → cannot gate, push to failing
            ranking.failing.append(entry)
            continue

        if gate_higher_is_better:
            passes = score >= gate_threshold
        else:
            passes = score <= gate_threshold

        (ranking.passing if passes else ranking.failing).append(entry)

    # Sort passing by compression_ratio ascending (smaller = better)
    ranking.passing.sort(key=lambda e: e["compression_ratio"])
    # Sort failing by the gate metric (best-first) for transparency
    if gate_higher_is_better:
        ranking.failing.sort(
            key=lambda e: e["scores"].get(gate_metric, float("-inf")),
            reverse=True,
        )
    else:
        ranking.failing.sort(
            key=lambda e: e["scores"].get(gate_metric, float("inf")),
        )

    return ranking


def default_gate_for_category(category: str) -> tuple[str, float, bool]:
    """Return sensible default quality gate for a category.

    Returns (metric_name, threshold, higher_is_better).
    """
    gates: dict[str, tuple[str, float, bool]] = {
        "photo":       ("ssimulacra2", 75.0, False),   # SSIMULACRA2 ≤ 75
        "manga_bw":    ("edge_preservation", 0.85, True),  # edge ≥ 0.85
        "manga_color": ("ssimulacra2", 70.0, False),   # SSIMULACRA2 ≤ 70
        "animated":    ("psnr", 35.0, True),            # PSNR ≥ 35
    }
    return gates.get(category, ("compression_ratio", 1.0, False))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Category-level aggregate statistics
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CategoryAggregate:
    """Per-config aggregate stats across all images in a category."""

    category: str
    config_id: str
    format: str

    # Compression stats
    mean_compression_ratio: float = 0.0
    mean_savings_pct: float = 0.0
    mean_compressed_size: float = 0.0
    total_original_size: int = 0
    total_compressed_size: int = 0
    image_count: int = 0

    # Quality stats (per metric: mean, std, min, max)
    quality_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Speed stats
    mean_encode_time: float = 0.0
    total_encode_time: float = 0.0

    @property
    def overall_savings_pct(self) -> float:
        """Total bytes savings across all images."""
        if self.total_original_size == 0:
            return 0.0
        return (1.0 - self.total_compressed_size / self.total_original_size) * 100.0

    @property
    def gb_saved_per_hour(self) -> float:
        """Throughput: GB saved per hour of encoding time."""
        if self.total_encode_time == 0:
            return 0.0
        bytes_saved = self.total_original_size - self.total_compressed_size
        gb_saved = bytes_saved / (1024**3)
        hours = self.total_encode_time / 3600.0
        return gb_saved / hours


def compute_category_aggregates(
    all_results: list[list[EncodeResult]],
    categories: list[str],
) -> dict[str, list[CategoryAggregate]]:
    """Compute per-category aggregate statistics across all images.

    For each category (e.g. 'manga_bw'), groups results by config_id
    across all images and computes mean compression ratio, mean quality
    scores, and throughput metrics.

    Args:
        all_results: Per-image list of EncodeResult lists.
        categories: Per-image category labels.

    Returns:
        Dict mapping category name → list of CategoryAggregate (one per config).
    """
    # Collect per-category per-config data
    # Structure: cat → config_id → list of EncodeResult
    cat_configs: dict[str, dict[str, list[EncodeResult]]] = {}

    for results, cat in zip(all_results, categories):
        if cat not in cat_configs:
            cat_configs[cat] = {}
        for r in results:
            cfg = r.config_id
            if cfg not in cat_configs[cat]:
                cat_configs[cat][cfg] = []
            cat_configs[cat][cfg].append(r)

    # Build aggregates
    output: dict[str, list[CategoryAggregate]] = {}

    for cat, configs in cat_configs.items():
        aggs: list[CategoryAggregate] = []
        for config_id, rlist in configs.items():
            if not rlist:
                continue

            n = len(rlist)
            agg = CategoryAggregate(
                category=cat,
                config_id=config_id,
                format=rlist[0].format,
                image_count=n,
            )

            # Size stats
            ratios = [r.compression_ratio for r in rlist]
            savings = [r.savings_pct for r in rlist]
            sizes = [r.compressed_size for r in rlist]
            encode_times = [r.encode_time for r in rlist]

            agg.mean_compression_ratio = sum(ratios) / n
            agg.mean_savings_pct = sum(savings) / n
            agg.mean_compressed_size = sum(sizes) / n
            agg.total_original_size = sum(r.original_size for r in rlist)
            agg.total_compressed_size = sum(r.compressed_size for r in rlist)
            agg.mean_encode_time = sum(encode_times) / n
            agg.total_encode_time = sum(encode_times)

            # Quality stats per metric
            metric_names: set[str] = set()
            for r in rlist:
                metric_names.update(r.scores.keys())

            for mname in sorted(metric_names):
                values = [
                    r.scores[mname] for r in rlist
                    if mname in r.scores
                    and not (isinstance(r.scores[mname], float) and r.scores[mname] != r.scores[mname])
                ]
                if values:
                    agg.quality_stats[mname] = {
                        "mean": sum(values) / len(values),
                        "std": _std(values),
                        "min": min(values),
                        "max": max(values),
                        "n_valid": len(values),
                    }

            aggs.append(agg)

        # Sort by mean_compression_ratio ascending (smaller = better)
        aggs.sort(key=lambda a: a.mean_compression_ratio)
        output[cat] = aggs

    return output


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 3. Category-level Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════


def compute_category_pareto(
    aggregates: list[CategoryAggregate],
    quality_metric: str,
    *,
    higher_is_better: bool = True,
) -> list[int]:
    """Find Pareto-optimal configs in the size-vs-quality plane.

    X-axis: mean_compression_ratio (lower = smaller file = better).
    Y-axis: mean quality score (direction depends on metric).

    A config A dominates B if:
      - A has smaller or equal compression ratio, AND
      - A has better or equal quality, AND
      - At least one is strictly better.

    Args:
        aggregates: Category-level aggregates (one per config).
        quality_metric: Which quality metric to use for Y-axis.
        higher_is_better: Whether higher quality score is better.

    Returns:
        Indices into aggregates list for Pareto-optimal configs.
    """
    n = len(aggregates)
    if n <= 1:
        return list(range(n))

    xs = [a.mean_compression_ratio for a in aggregates]
    ys: list[float] = []
    for a in aggregates:
        q = a.quality_stats.get(quality_metric, {})
        ys.append(q.get("mean", 0.0))

    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i?
            size_better = xs[j] <= xs[i]
            qual_better = (
                ys[j] >= ys[i] if higher_is_better
                else ys[j] <= ys[i]
            )
            size_strict = xs[j] < xs[i]
            qual_strict = (
                ys[j] > ys[i] if higher_is_better
                else ys[j] < ys[i]
            )
            if size_better and qual_better and (size_strict or qual_strict):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Throughput metric — GB saved per hour
# ═══════════════════════════════════════════════════════════════════════════


def compute_throughput(
    result: EncodeResult,
) -> float:
    """Compute single-image encode throughput: GB saved per hour.

    Args:
        result: A single EncodeResult.

    Returns:
        GB saved per hour of encoding.
    """
    if result.encode_time == 0:
        return 0.0
    bytes_saved = result.original_size - result.compressed_size
    gb_saved = bytes_saved / (1024**3)
    hours = result.encode_time / 3600.0
    return gb_saved / hours


def compute_images_per_minute(
    results: list[EncodeResult],
) -> float:
    """Batch throughput: images processed per minute.

    Args:
        results: All results from a batch run.

    Returns:
        Images per minute.
    """
    if not results:
        return 0.0
    total_time = sum(r.encode_time for r in results)
    if total_time == 0:
        return 0.0
    return len(results) / (total_time / 60.0)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy compatibility — weighted scoring (deprecated)
# ═══════════════════════════════════════════════════════════════════════════


def normalize_scores(
    results: list[EncodeResult],
    metric_name: str,
    *,
    higher_is_better: bool = True,
) -> dict[int, float]:
    """Normalize one metric across all results to [0, 1].

    DEPRECATED: Prefer anchor_rank() for production use.
    Kept for backward compatibility.

    Args:
        results: All EncodeResult for one image (one per codec config).
        metric_name: The metric key in EncodeResult.scores.
        higher_is_better: If True, 1.0 = best. If False, 1.0 = best (inverted).

    Returns:
        Dict mapping result index → normalized score.
    """
    values = []
    valid_indices = []
    for i, r in enumerate(results):
        v = r.scores.get(metric_name)
        if v is not None and not (isinstance(v, float) and (v != v)):
            values.append(v)
            valid_indices.append(i)

    if not values:
        return {i: 0.5 for i in range(len(results))}

    v_min = min(values)
    v_max = max(values)
    span = v_max - v_min

    normalized: dict[int, float] = {}
    for i in range(len(results)):
        normalized[i] = 0.5

    for idx, v in zip(valid_indices, values):
        if span == 0:
            norm = 1.0
        else:
            norm = (v - v_min) / span
        if not higher_is_better:
            norm = 1.0 - norm
        normalized[idx] = norm

    return normalized


def compute_ranked_scores(
    results: list[EncodeResult],
    metrics: list[tuple[str, float, bool]],
    ranking_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """DEPRECATED: Weighted total score ranking.

    Replaced by anchor_rank() which uses quality-gate + size-ordering.
    Kept for backward compatibility only.

    Dimensions and defaults:
        size    0.20 — compression ratio (lower is better)
        quality 0.55 — weighted perceptual score
        encode  0.25 — encode speed (shorter is better)
    """
    if ranking_weights is None:
        ranking_weights = {
            "size": 0.20, "quality": 0.55, "encode": 0.25,
        }

    n = len(results)
    if n == 0:
        return []

    # Quality sub-score
    quality_scores = _compute_quality_subscore(results, metrics)

    sizes = [r.compression_ratio for r in results]
    encode_times = [r.encode_time for r in results]

    def _norm(values: list[float], lower_is_better: bool) -> list[float]:
        vmin, vmax = min(values), max(values)
        span = vmax - vmin
        if span == 0:
            return [1.0] * n
        out = [(v - vmin) / span for v in values]
        return [1.0 - x for x in out] if lower_is_better else out

    size_norm = _norm(sizes, lower_is_better=True)
    encode_norm = _norm(encode_times, lower_is_better=True)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        total = (
            ranking_weights.get("size", 0.20) * size_norm[i]
            + ranking_weights.get("quality", 0.55) * quality_scores[i]
            + ranking_weights.get("encode", 0.25) * encode_norm[i]
        )
        rows.append({
            "config_id": results[i].config_id,
            "format": results[i].format,
            "compression_ratio": round(results[i].compression_ratio, 4),
            "savings_pct": round(results[i].savings_pct, 2),
            "encode_time": round(results[i].encode_time, 4),
            "size_score": round(size_norm[i], 4),
            "quality_score": round(quality_scores[i], 4),
            "encode_score": round(encode_norm[i], 4),
            "total_score": round(total, 4),
        })

    rows.sort(key=lambda r_: r_["total_score"], reverse=True)
    return rows


def _compute_quality_subscore(
    results: list[EncodeResult],
    metrics: list[tuple[str, float, bool]],
) -> list[float]:
    """Weighted perceptual quality per result."""
    n = len(results)
    weighted_sums = [0.0] * n
    total_weight = sum(w for _, w, _ in metrics)

    if total_weight == 0:
        return [0.5] * n

    for name, weight, higher in metrics:
        norm = normalize_scores(results, name, higher_is_better=higher)
        for i in range(n):
            weighted_sums[i] += weight * norm[i]

    return [s / total_weight for s in weighted_sums]


# ═══════════════════════════════════════════════════════════════════════════
# Single-image Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════


def compute_pareto_frontier(
    results: list[EncodeResult],
    x_metric: str | None = None,
    y_metric: str | None = None,
) -> list[int]:
    """Find Pareto-optimal results (non-dominated in size vs quality).

    A result A dominates B if:
      - A.compression_ratio <= B.compression_ratio (smaller or equal)
      - A has a higher total quality score
      - At least one is strictly better

    Args:
        results: EncodeResult list.
        x_metric: Optional metric name for X axis (default: compression_ratio).
        y_metric: Optional metric name for Y axis (default: first score).

    Returns:
        Indices of Pareto-optimal results.
    """
    n = len(results)
    if n <= 1:
        return list(range(n))

    xs = [r.compression_ratio for r in results]
    ys: list[float] = []
    for r in results:
        if y_metric and y_metric in r.scores:
            ys.append(r.scores[y_metric])
        elif r.scores:
            ys.append(next(iter(r.scores.values())))
        else:
            ys.append(0.5)

    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]
