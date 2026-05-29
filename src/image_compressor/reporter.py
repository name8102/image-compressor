"""Progress reporting and statistics display using rich."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def format_bytes(size: int) -> str:
    """Format bytes to human readable string.

    Args:
        size: Size in bytes.

    Returns:
        Human readable size string.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0  # type: ignore[assignment]
    return f"{size:.1f} PB"


def create_progress() -> Progress:
    """Create a rich progress bar for compression tasks.

    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )


class CompressionReporter:
    """Reporter for compression progress and statistics."""

    def __init__(self, total_files: int) -> None:
        """Initialize reporter.

        Args:
            total_files: Total number of files to process.
        """
        self.total_files = total_files
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.total_original = 0
        self.total_compressed = 0
        self.start_time = time.perf_counter()
        self.progress = create_progress()
        self.task_id = self.progress.add_task("压缩进度", total=total_files)

    def update(
        self,
        original_size: int = 0,
        compressed_size: int = 0,
        success: bool = True,
        skipped: bool = False,
    ) -> None:
        """Update progress with a completed file.

        Args:
            original_size: Original file size.
            compressed_size: Compressed file size.
            success: Whether compression succeeded.
            skipped: Whether file was skipped.
        """
        self.progress.update(self.task_id, advance=1)

        if skipped:
            self.skipped += 1
        elif success:
            self.completed += 1
            self.total_original += original_size
            self.total_compressed += compressed_size
        else:
            self.failed += 1

    def __enter__(self) -> CompressionReporter:
        """Enter context manager."""
        self.progress.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self.progress.stop()

    def print_summary(self) -> None:
        """Print final summary."""
        elapsed = time.perf_counter() - self.start_time
        savings = self.total_original - self.total_compressed
        ratio = (self.total_compressed / self.total_original * 100) if self.total_original > 0 else 0

        table = Table(title="压缩完成", show_header=False)
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")

        table.add_row("总文件数", str(self.total_files))
        table.add_row("成功", str(self.completed))
        table.add_row("失败", str(self.failed))
        table.add_row("跳过", str(self.skipped))
        table.add_row("原始大小", format_bytes(self.total_original))
        table.add_row("压缩后大小", format_bytes(self.total_compressed))
        table.add_row("节省空间", format_bytes(savings))
        table.add_row("压缩率", f"{ratio:.1f}%")
        table.add_row("总耗时", f"{elapsed:.1f} 秒")

        if self.completed > 0:
            avg_time = elapsed / self.completed
            table.add_row("平均速度", f"{avg_time:.2f} 秒/文件")

        console.print()
        console.print(Panel(table))


def print_stats(stats: dict[str, Any]) -> None:
    """Print statistics from database.

    Args:
        stats: Statistics dictionary from Checkpoint.get_stats().
    """
    table = Table(title="处理统计", show_header=False)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("总文件数", str(stats["total"]))
    table.add_row("已完成", str(stats["completed"]))
    table.add_row("失败", str(stats["failed"]))
    table.add_row("待处理", str(stats["pending"]))
    table.add_row("处理中", str(stats["processing"]))

    if stats["completed"] > 0:
        table.add_row("原始大小", format_bytes(stats["total_original_bytes"]))
        table.add_row("压缩后大小", format_bytes(stats["total_compressed_bytes"]))
        table.add_row("节省空间", format_bytes(stats["savings_bytes"]))
        table.add_row("压缩率", f"{stats['compression_ratio']:.1f}%")

        elapsed = stats["total_time_seconds"]
        if elapsed > 0:
            table.add_row("总处理时间", f"{elapsed:.1f} 秒")
            avg = elapsed / stats["completed"]
            table.add_row("平均速度", f"{avg:.2f} 秒/文件")

    console.print(Panel(table))
