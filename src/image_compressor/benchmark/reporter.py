"""Reporter — data aggregation, CSV/JSON export, and Rich console tables.

Generates per-image breakdowns, cross-config summaries, and
a final ranked recommendation report.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from image_compressor.evaluators.base import EncodeResult

logger = logging.getLogger(__name__)

# Forward reference for type hints (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_compressor.benchmark.scorer import AnchorRanking, CategoryAggregate


# ── Helpers ──────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except ModuleNotFoundError:
    _HAS_RICH = False


def _rich_table(title: str, columns: list[str]) -> Any:
    if _HAS_RICH:
        return Table(title=title, show_header=True, header_style="bold magenta")
    # Fallback: plain list of dicts
    return []


def _add_row(table: Any, row: list[Any]) -> None:
    if _HAS_RICH:
        table.add_row(*[str(v) for v in row])


def _print_table(table: Any) -> None:
    if _HAS_RICH:
        Console().print(table)


# ── Per-result summary row ───────────────────────────────────────────────


def result_summary_row(result: EncodeResult, rank: int | None = None) -> dict[str, Any]:
    """Convert one EncodeResult to a flat dict for CSV/JSON."""
    row: dict[str, Any] = {
        "config_id": result.config_id,
        "format": result.format,
        "original_size": result.original_size,
        "compressed_size": result.compressed_size,
        "ratio_pct": round(result.compression_ratio * 100, 2),
        "savings_pct": round(result.savings_pct, 2),
        "encode_time_s": round(result.encode_time, 4),
        "decode_time_s": round(result.decode_time, 4),
    }
    if rank is not None:
        row["rank"] = rank
    for k, v in result.scores.items():
        row[f"score_{k}"] = round(v, 6) if v == v else None
    return row


# ── Export functions ─────────────────────────────────────────────────────


def export_csv(
    all_results: list[list[EncodeResult]],       # per-image list of results
    image_paths: list[Path],
    output_path: Path,
) -> None:
    """Export full results as CSV (one row per config per image)."""
    if not all_results:
        logger.warning("No results to export.")
        return

    # Collect all metric names for columns
    all_metrics: set[str] = set()
    for results in all_results:
        for r in results:
            all_metrics.update(r.scores.keys())

    fieldnames = [
        "image", "config_id", "format", "rank",
        "original_size", "compressed_size", "ratio_pct", "savings_pct",
        "encode_time_s",
    ] + [f"score_{m}" for m in sorted(all_metrics)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for img_path, results in zip(image_paths, all_results):
            for i, r in enumerate(results):
                row = result_summary_row(r, rank=i + 1)
                row["image"] = img_path.name
                writer.writerow(row)

    logger.info("CSV exported → %s", output_path)


def export_json(
    all_results: list[list[EncodeResult]],
    image_paths: list[Path],
    ranked_scores: list[list[dict[str, Any]]],
    categories: list[str],
    output_path: Path,
) -> None:
    """Export full results as structured JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: list[dict[str, Any]] = []
    for img_path, results, ranked, cat in zip(
        image_paths, all_results, ranked_scores, categories
    ):
        entry: dict[str, Any] = {
            "image": str(img_path),
            "category": cat,
            "results": [],
        }
        for idx, r in enumerate(results):
            item = r.to_dict()
            if idx < len(ranked):
                item["rank"] = idx + 1
                item["total_score"] = ranked[idx].get("total_score")
                item["size_score"] = ranked[idx].get("size_score")
                item["quality_score"] = ranked[idx].get("quality_score")
                item["encode_score"] = ranked[idx].get("encode_score")
            entry["results"].append(item)
        payload.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("JSON exported → %s", output_path)


# ── Console reports ──────────────────────────────────────────────────────


def print_per_image_report(
    image_path: Path,
    category: str,
    ranked: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> None:
    """Print a Rich table of ranked results for one image."""
    title = f"Benchmark: {image_path.name} [{category}]"
    if _HAS_RICH:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Config", width=18)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Savings%", justify="right", width=8)
        table.add_column("Encode", justify="right", width=8)
        table.add_column("Quality", justify="right", width=8)
        table.add_column("Total ★", justify="right", width=10)
    else:
        table = []
        print(f"\n{title}")
        print(f"{'#':>3} {'Config':<18} {'Size':>10} {'Savings%':>8} {'Encode':>8} {'Quality':>8} {'Total ★':>10}")
        print("-" * 67)

    for i, row in enumerate(ranked):
        r = row["result"]
        size_str = f"{r.compressed_size:,}"
        savings_str = f"{r.savings_pct:.1f}%"
        encode_str = f"{r.encode_time:.2f}s"
        quality_str = f"{row['quality_score']:.3f}"
        total_str = f"{row['total_score']:.4f}"

        if _HAS_RICH:
            style = "bold green" if i == 0 else ""
            table.add_row(
                str(i + 1), r.config_id, size_str, savings_str,
                encode_str, quality_str, total_str,
                style=style,
            )
        else:
            print(f"{i+1:>3} {r.config_id:<18} {size_str:>10} {ratio_str:>8} {encode_str:>8} {quality_str:>8} {total_str:>10}")

    if _HAS_RICH:
        Console().print(table)
    else:
        print()

    # Top recommendation
    if ranked:
        best = ranked[0]
        r = best["result"]
        print(f"  → 推荐: {r.config_id}  (总分 {best['total_score']:.4f}, "
              f"节省 {r.savings_pct:.1f}%, 编码 {r.encode_time:.2f}s)")


# ── Anchor-based reports ────────────────────────────────────────────────


def print_anchor_report(anchor: Any) -> None:
    """Print anchor-based ranking for one image."""
    if _HAS_RICH:
        table = Table(
            title=f"Anchor: {anchor.image_name} [{anchor.category}]  gate={anchor.gate_metric}",
            show_header=True, header_style="bold cyan",
        )
        table.add_column("Config", width=18)
        table.add_column("Pass", width=6)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Ratio", justify="right", width=8)
        table.add_column("Savings%", justify="right", width=8)
        table.add_column("Encode", justify="right", width=8)

        for e in anchor.passing:
            table.add_row(
                e["config_id"], "✓",
                f"{e['compressed_size']:,}",
                f"{e['compression_ratio']:.2%}",
                f"{e['savings_pct']:.1f}%",
                f"{e['encode_time']:.2f}s",
                style="green",
            )
        for e in anchor.failing:
            table.add_row(
                e["config_id"], "✗",
                f"{e['compressed_size']:,}",
                f"{e['compression_ratio']:.2%}",
                f"{e['savings_pct']:.1f}%",
                f"{e['encode_time']:.2f}s",
                style="dim",
            )
        Console().print(table)
    else:
        gate_dir = ">=" if anchor.gate_higher_is_better else "<="
        print(f"\nAnchor: {anchor.image_name} [{anchor.category}]  "
              f"gate={anchor.gate_metric} {gate_dir} {anchor.gate_threshold}")
        print(f"  PASS ({len(anchor.passing)}):")
        for e in anchor.passing:
            print(f"    {e['config_id']:<18} size={e['compressed_size']:,}  "
                  f"savings={e['savings_pct']:.1f}%  enc={e['encode_time']:.2f}s")
        if anchor.failing:
            print(f"  FAIL ({len(anchor.failing)}):")
            for e in anchor.failing[:3]:
                print(f"    {e['config_id']:<18} size={e['compressed_size']:,}")

    if anchor.best:
        b = anchor.best
        print(f"  → 推荐: {b['config_id']}  (节省 {b['savings_pct']:.1f}%, "
              f"体积 {b['compressed_size']:,} bytes, 编码 {b['encode_time']:.2f}s)")


def print_category_aggregate_report(
    category: str,
    aggregates: list[Any],
    top_n: int = 10,
) -> None:
    """Print category-level aggregate statistics."""
    if not aggregates:
        return

    if _HAS_RICH:
        table = Table(
            title=f"Category Aggregates: {category} ({aggregates[0].image_count} images)",
            show_header=True, header_style="bold magenta",
        )
        table.add_column("Config", width=18)
        table.add_column("Mean Ratio", justify="right", width=10)
        table.add_column("Mean Save%", justify="right", width=10)
        table.add_column("Mean Enc(s)", justify="right", width=10)
        table.add_column("GB/h", justify="right", width=8)

        for a in aggregates[:top_n]:
            table.add_row(
                a.config_id,
                f"{a.mean_compression_ratio:.2%}",
                f"{a.mean_savings_pct:.1f}%",
                f"{a.mean_encode_time:.2f}",
                f"{a.gb_saved_per_hour:.2f}",
            )
        Console().print(table)
    else:
        print(f"\nCategory Aggregates: {category} ({aggregates[0].image_count} images)")
        print(f"{'Config':<18} {'Ratio':>10} {'Save%':>10} {'Enc(s)':>10} {'GB/h':>8}")
        for a in aggregates[:top_n]:
            print(f"{a.config_id:<18} {a.mean_compression_ratio:>9.2%} "
                  f"{a.mean_savings_pct:>9.1f}% {a.mean_encode_time:>9.2f} "
                  f"{a.gb_saved_per_hour:>7.2f}")


def print_pareto_report(
    category: str,
    all_aggs: list[Any],
    pareto: list[Any],
    gate: tuple[str, float, bool],
) -> None:
    """Print category-level Pareto frontier."""
    if not pareto:
        return

    gate_metric, gate_threshold, higher_is_better = gate
    qual_key = gate_metric

    if _HAS_RICH:
        table = Table(
            title=f"Pareto Frontier: {category} (size vs {qual_key})",
            show_header=True, header_style="bold green",
        )
        table.add_column("#", width=3)
        table.add_column("Config", width=18)
        table.add_column("Mean Ratio", justify="right", width=10)
        table.add_column("Mean Save%", justify="right", width=10)
        table.add_column(f"Mean {qual_key}", justify="right", width=14)
        table.add_column("GB/h", justify="right", width=8)
        table.add_column("Images", justify="right", width=6)

        for i, a in enumerate(pareto):
            qual_mean = a.quality_stats.get(qual_key, {}).get("mean", 0)
            table.add_row(
                str(i + 1), a.config_id,
                f"{a.mean_compression_ratio:.2%}",
                f"{a.mean_savings_pct:.1f}%",
                f"{qual_mean:.3f}",
                f"{a.gb_saved_per_hour:.2f}",
                str(a.image_count),
                style="bold green" if i == 0 else "",
            )
        Console().print(table)
    else:
        print(f"\nPareto Frontier: {category} (size vs {qual_key})")
        print(f"{'#':>3} {'Config':<18} {'Ratio':>10} {'Save%':>10} {'Qual':>14} {'GB/h':>8}")
        for i, a in enumerate(pareto):
            qual_mean = a.quality_stats.get(qual_key, {}).get("mean", 0)
            print(f"{i+1:>3} {a.config_id:<18} {a.mean_compression_ratio:>9.2%} "
                  f"{a.mean_savings_pct:>9.1f}% {qual_mean:>13.3f} "
                  f"{a.gb_saved_per_hour:>7.2f}")

    if pareto:
        best = pareto[0]
        print(f"  → 帕累托最优推荐: {best.config_id}  "
              f"(平均节省 {best.mean_savings_pct:.1f}%, "
              f"吞吐 {best.gb_saved_per_hour:.2f} GB/h)")


# ── Summary report ──────────────────────────────────────────────────────


def print_summary_report(
    all_ranked: list[list[dict[str, Any]]],
    image_paths: list[Path],
    categories: list[str],
) -> None:
    """Print a summary table of best config per image."""
    title = "Best Config per Image — Summary"
    if _HAS_RICH:
        table = Table(title=title, show_header=True, header_style="bold yellow")
        table.add_column("Image", width=30)
        table.add_column("Category", width=12)
        table.add_column("Best Config", width=18)
        table.add_column("Savings%", justify="right", width=8)
        table.add_column("Encode", justify="right", width=8)
        table.add_column("Total ★", justify="right", width=10)
    else:
        print(f"\n{title}")
        print(f"{'Image':<30} {'Category':<12} {'Best Config':<18} {'Savings%':>8} {'Encode':>8} {'Total ★':>10}")
        print("-" * 92)

    for img_path, ranked, cat in zip(image_paths, all_ranked, categories):
        if not ranked:
            continue
        best = ranked[0]
        r = best["result"]
        if _HAS_RICH:
            table.add_row(
                img_path.name[:28], cat,
                r.config_id,
                f"{r.savings_pct:.1f}%",
                f"{r.encode_time:.2f}s",
                f"{best['total_score']:.4f}",
            )
        else:
            print(f"{img_path.name[:28]:<30} {cat:<12} {r.config_id:<18} {r.savings_pct:>7.1f}% {r.encode_time:>7.2f}s {best['total_score']:>10.4f}")

    if _HAS_RICH:
        Console().print(table)
