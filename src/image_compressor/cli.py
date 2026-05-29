"""CLI entry point for image-compressor."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .cbz_handler import process_cbz
from .checkpoint import Checkpoint, FileStatus
from .compressor import process_image
from .config import get_workers, load_config, get_preset
from .presets import Preset, load_presets
from .reporter import CompressionReporter, print_stats
from .scanner import scan_directory, is_archive

app = typer.Typer(
    name="image-compressor",
    help="TB 级图片批量压缩工具",
    add_completion=True,
)
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging.

    Args:
        level: Log level string.
        log_file: Optional log file path.
    """
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True)
    ]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


@app.command()
def scan(
    directory: str = typer.Argument(..., help="要扫描的目录路径"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="配置文件路径"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r", help="是否递归扫描子目录"
    ),
) -> None:
    """扫描目录，将图片和 CBZ 文件录入数据库。"""
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]错误:[/red] {e}")
        raise typer.Exit(1)

    db_path = config.get("database", {}).get("path", "image_compressor.db")
    checkpoint = Checkpoint(db_path)

    console.print(f"[blue]扫描目录:[/blue] {directory}")

    files_found = 0
    files_added = 0

    with console.status("[bold green]扫描中..."):
        for file_info in scan_directory(directory, recursive=recursive):
            files_found += 1
            added = checkpoint.add_file(
                path=file_info["path"],
                source_type="file",
                original_size=file_info["size"],
            )
            if added:
                files_added += 1

    console.print(f"[green]扫描完成![/green]")
    console.print(f"  发现文件: {files_found}")
    console.print(f"  新增文件: {files_added}")
    console.print(f"  已存在: {files_found - files_added}")


@app.command()
def compress(
    preset_name: str = typer.Option(..., "--preset", "-p", help="压缩预设名称"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="配置文件路径"
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="并行工作进程数"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="测试模式，只打印不执行"
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", "-b", help="每批处理文件数"
    ),
) -> None:
    """开始压缩文件（断点续传）。"""
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]错误:[/red] {e}")
        raise typer.Exit(1)

    # Load preset
    try:
        presets = load_presets(config)
        preset = presets[preset_name]
    except KeyError as e:
        console.print(f"[red]错误:[/red] {e}")
        raise typer.Exit(1)

    db_path = config.get("database", {}).get("path", "image_compressor.db")
    checkpoint = Checkpoint(db_path)

    # Reset any files stuck in processing state
    reset_count = checkpoint.reset_processing()
    if reset_count > 0:
        console.print(f"[yellow]重置了 {reset_count} 个中断的文件[/yellow]")

    # Get workers
    num_workers = workers or get_workers(config)

    # Get pending files
    pending_files = checkpoint.get_pending_files(limit=batch_size)
    if not pending_files:
        console.print("[green]没有待处理的文件！[/green]")
        return

    console.print(f"[blue]预设:[/blue] {preset.name} - {preset.description}")
    console.print(f"[blue]待处理:[/blue] {len(pending_files)} 个文件")
    console.print(f"[blue]并行数:[/blue] {num_workers}")

    if dry_run:
        console.print("\n[yellow]测试模式 - 以下文件将被处理:[/yellow]")
        for f in pending_files[:20]:  # Show first 20
            console.print(f"  {f['path']}")
        if len(pending_files) > 20:
            console.print(f"  ... 还有 {len(pending_files) - 20} 个文件")
        return

    # Get backup config
    backup_config = config.get("backup", {})
    backup_enabled = backup_config.get("enabled", True)
    backup_subdir = backup_config.get("original_dir", ".original")

    # Process files
    reporter = CompressionReporter(len(pending_files))

    with reporter:
        if num_workers <= 1:
            # Sequential processing
            for file_info in pending_files:
                result = _process_single_file(file_info, preset, backup_subdir, backup_enabled)
                reporter.update(
                    original_size=result.get("original_size", 0),
                    compressed_size=result.get("compressed_size", 0),
                    success=result.get("success", False),
                    skipped=result.get("skipped", False),
                )
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _process_single_file,
                        file_info,
                        preset,
                        backup_subdir,
                        backup_enabled,
                    ): file_info
                    for file_info in pending_files
                }

                for future in as_completed(futures):
                    result = future.result()
                    reporter.update(
                        original_size=result.get("original_size", 0),
                        compressed_size=result.get("compressed_size", 0),
                        success=result.get("success", False),
                        skipped=result.get("skipped", False),
                    )

    reporter.print_summary()


def _process_single_file(
    file_info: dict,
    preset: Preset,
    backup_subdir: str,
    backup_enabled: bool,
) -> dict:
    """Process a single file (for use in worker processes).

    Args:
        file_info: File information dictionary from database.
        preset: Compression preset.
        backup_subdir: Backup subdirectory name.
        backup_enabled: Whether backup is enabled.

    Returns:
        Result dictionary.
    """
    # Re-initialize checkpoint in worker process
    # (SQLite connections can't be shared across processes)
    from .config import load_config
    config = load_config()
    db_path = config.get("database", {}).get("path", "image_compressor.db")
    checkpoint = Checkpoint(db_path)

    path = file_info["path"]
    parent_cbz = file_info.get("parent_cbz")

    try:
        checkpoint.mark_processing(path)

        if parent_cbz:
            # This is a CBZ entry - skip individual processing
            # CBZ should be processed as a whole
            checkpoint.mark_skipped(path, "Part of CBZ archive")
            return {"success": True, "skipped": True}
        elif is_archive(path):
            # Process CBZ archive
            backup_dir = str(Path(path).parent / backup_subdir) if backup_enabled else None
            if backup_dir:
                Path(backup_dir).mkdir(parents=True, exist_ok=True)

            # Get compression params
            params = preset.get_compression_params("jpeg")  # Default for CBZ
            result = process_cbz(path, params, backup_dir)

            if result.success:
                checkpoint.mark_completed(
                    path,
                    compressed_size=result.compressed_size,
                    processing_time=result.processing_time,
                    preset=preset.name,
                )
                return {
                    "success": True,
                    "original_size": result.original_size,
                    "compressed_size": result.compressed_size,
                }
            else:
                checkpoint.mark_failed(path, result.error or "Unknown error")
                return {"success": False}
        else:
            # Process regular image
            ext = Path(path).suffix.lower().lstrip(".")
            if ext in ("jpg", "jpeg"):
                source_format = "jpeg"
            elif ext == "png":
                source_format = "png"
            elif ext == "webp":
                source_format = "webp"
            else:
                source_format = "jpeg"

            params = preset.get_compression_params(source_format)
            backup_dir = str(Path(path).parent / backup_subdir) if backup_enabled else None
            if backup_dir:
                Path(backup_dir).mkdir(parents=True, exist_ok=True)

            result = process_image(path, params, backup_dir)

            if result.success:
                checkpoint.mark_completed(
                    result.output_path or path,
                    compressed_size=result.compressed_size,
                    processing_time=result.processing_time,
                    preset=preset.name,
                )
                return {
                    "success": True,
                    "original_size": result.original_size,
                    "compressed_size": result.compressed_size,
                }
            else:
                checkpoint.mark_failed(path, result.error or "Unknown error")
                return {"success": False}

    except Exception as e:
        checkpoint.mark_failed(path, str(e))
        return {"success": False}


@app.command()
def status(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="配置文件路径"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="显示详细信息"
    ),
) -> None:
    """查看处理进度统计。"""
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]错误:[/red] {e}")
        raise typer.Exit(1)

    db_path = config.get("database", {}).get("path", "image_compressor.db")
    checkpoint = Checkpoint(db_path)

    stats = checkpoint.get_stats()
    print_stats(stats)

    if detailed and stats["failed"] > 0:
        console.print("\n[red]失败文件:[/red]")
        # Show first 10 failed files
        with checkpoint._connect() as conn:
            rows = conn.execute(
                "SELECT path, error_message FROM files WHERE status = 'failed' LIMIT 10"
            ).fetchall()
            for row in rows:
                console.print(f"  {row['path']}: {row['error_message']}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
