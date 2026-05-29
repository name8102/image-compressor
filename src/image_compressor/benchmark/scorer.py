"""Score normalization and multi-dimensional weighted ranking.

Normalizes heterogeneous metrics (SSIM 0-1, Butteraugli 0-10+, PSNR 20-50)
onto a common 0–1 scale, then computes a weighted total score across
four dimensions: size, quality, encode speed, decode speed.
"""

from __future__ import annotations

import logging
from typing import Any

from image_compressor.evaluators.base import EncodeResult

logger = logging.getLogger(__name__)


# ── Normalization ────────────────────────────────────────────────────────


def normalize_scores(
    results: list[EncodeResult],
    metric_name: str,
    *,
    higher_is_better: bool = True,
) -> dict[int, float]:
    """Normalize one metric across all results to [0, 1].

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
        if v is not None and not (isinstance(v, float) and (v != v)):  # skip NaN
            values.append(v)
            valid_indices.append(i)

    if not values:
        return {i: 0.5 for i in range(len(results))}  # all equal if no data

    v_min = min(values)
    v_max = max(values)
    span = v_max - v_min

    normalized: dict[int, float] = {}
    for i in range(len(results)):
        # Default mid-score for missing/NAN values
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


# ── Four-dimension scoring ───────────────────────────────────────────────


def compute_quality_score(
    results: list[EncodeResult],
    metrics: list[tuple[str, float, bool]],
) -> list[float]:
    """Weighted perceptual quality score per result.

    Args:
        results: EncodeResult list.
        metrics: List of (metric_name, weight, higher_is_better).

    Returns:
        List of quality scores [0,1], one per entry in results.
    """
    # For each metric, normalize across results, then weight
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


def compute_ranked_scores(
    results: list[EncodeResult],
    metrics: list[tuple[str, float, bool]],
    ranking_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute final four-dimension weighted total score for ranking.

    Dimensions and defaults:
        size    0.20 — compression ratio (lower is better)
        quality 0.55 — weighted perceptual score
        encode  0.25 — encode speed (shorter is better)

    Args:
        results: All EncodeResult for one image.
        metrics: Category-specific evaluator configs.
        ranking_weights: Override the default dimension weights.

    Returns:
        List of dicts with per-result breakdowns, sorted best-first.
    """
    if ranking_weights is None:
        ranking_weights = {
            "size": 0.20, "quality": 0.55, "encode": 0.25,
        }

    n = len(results)
    if n == 0:
        return []

    quality_scores = compute_quality_score(results, metrics)

    # Collect raw values for normalization
    sizes = [r.compression_ratio for r in results]     # lower = better
    encode_times = [r.encode_time for r in results]    # lower = better

    def _norm_list(values: list[float], lower_is_better: bool) -> list[float]:
        vmin, vmax = min(values), max(values)
        span = vmax - vmin
        if span == 0:
            return [1.0] * n
        out = [(v - vmin) / span for v in values]
        if lower_is_better:
            out = [1.0 - x for x in out]
        return out

    size_norm = _norm_list(sizes, lower_is_better=True)
    encode_norm = _norm_list(encode_times, lower_is_better=True)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        total = (
            ranking_weights.get("size", 0.20) * size_norm[i]
            + ranking_weights.get("quality", 0.50) * quality_scores[i]
            + ranking_weights.get("encode", 0.10) * encode_norm[i]
        )
        rows.append({
            "result": results[i],
            "size_score": round(size_norm[i], 4),
            "quality_score": round(quality_scores[i], 4),
            "encode_score": round(encode_norm[i], 4),
            "total_score": round(total, 4),
        })

    # Sort best first
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows


# ── Pareto frontier ──────────────────────────────────────────────────────


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

    # X = size (lower better), Y = quality (higher better)
    xs = [r.compression_ratio for r in results]
    ys: list[float] = []
    for r in results:
        if y_metric and y_metric in r.scores:
            ys.append(r.scores[y_metric])
        elif r.scores:
            # Use first available score
            ys.append(next(iter(r.scores.values())))
        else:
            ys.append(0.5)

    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i?
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]
