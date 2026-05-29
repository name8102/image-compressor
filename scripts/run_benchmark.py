#!/usr/bin/env python3
"""Multi-Dimensional Image Compression Benchmark — entry script.

Usage:
    # Benchmark all images in a directory (auto-detect categories)
    python scripts/run_benchmark.py samples/

    # Benchmark with custom output directory
    python scripts/run_benchmark.py samples/ --output results/

    # Benchmark specific images with explicit category
    python scripts/run_benchmark.py img1.png img2.jpg --category manga_bw

    # Use a custom codec config subset
    python scripts/run_benchmark.py samples/ --codecs webp,jxl

    # Keep compressed outputs (don't clean up)
    python scripts/run_benchmark.py samples/ --keep-outputs
"""

import argparse
import logging
import sys
from pathlib import Path

# Make sure the project is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_compressor.benchmark import BenchmarkPipeline, load_all_configs

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dimensional Image Compression Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py samples/
  python scripts/run_benchmark.py samples/manga/ --category manga_bw
  python scripts/run_benchmark.py img.png --codecs webp,jxl,q80
        """,
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Image files or directories to benchmark",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("benchmark_results"),
        help="Output directory for reports (default: benchmark_results/)",
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        choices=["photo", "manga_bw", "manga_color", "animated"],
        help="Force a category for all inputs (default: auto-detect)",
    )
    parser.add_argument(
        "--keep-outputs", action="store_true",
        help="Keep compressed output files (default: clean up after evaluation)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to metrics_routing.yaml (default: config/metrics_routing.yaml)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce logging verbosity",
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # ── Build codec configs ──
    codec_configs = load_all_configs()
    log.info("Codec configs loaded: %d", len(codec_configs))

    # ── Initialize pipeline ──
    pipeline = BenchmarkPipeline(
        config_path=args.config,
        codec_configs=codec_configs,
        output_dir=args.output,
        clean_outputs=not args.keep_outputs,
    )

    # ── Register inputs ──
    for p in args.inputs:
        p = p.resolve()
        if not p.exists():
            log.warning("Path not found, skipping: %s", p)
            continue
        if p.is_dir():
            n = pipeline.add_directory(p, category=args.category)
            log.info("Added %d images from %s", n, p)
        elif p.is_file():
            pipeline.add_image(p, category=args.category)

    # ── Run ──
    pipeline.run_all()

    # ── Export reports ──
    paths = pipeline.export_reports()
    print(f"\nReports exported:")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
