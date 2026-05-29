"""SQLite-based state management for tracking file processing status."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator


class FileStatus(str, Enum):
    """File processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


DEFAULT_DB_PATH = Path("image_compressor.db")


class Checkpoint:
    """SQLite-based checkpoint manager for tracking file processing."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        """Initialize checkpoint manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'file',
                    parent_cbz TEXT,
                    original_size INTEGER,
                    compressed_size INTEGER,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_time REAL,
                    preset TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_parent_cbz ON files(parent_cbz)
            """)
            conn.commit()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_file(
        self,
        path: str,
        source_type: str = "file",
        parent_cbz: str | None = None,
        original_size: int | None = None,
    ) -> bool:
        """Add a file to the processing queue.

        Args:
            path: File path (or entry path within CBZ).
            source_type: 'file' or 'cbz_entry'.
            parent_cbz: Parent CBZ path if this is a CBZ entry.
            original_size: Original file size in bytes.

        Returns:
            True if file was added, False if already exists.
        """
        with self._connect() as conn:
            try:
                conn.execute(
                    """INSERT INTO files (path, source_type, parent_cbz, original_size)
                       VALUES (?, ?, ?, ?)""",
                    (path, source_type, parent_cbz, original_size),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def add_files_batch(self, files: list[dict[str, Any]]) -> int:
        """Add multiple files to the processing queue.

        Args:
            files: List of file dictionaries with keys: path, source_type,
                   parent_cbz (optional), original_size (optional).

        Returns:
            Number of files actually added (skips duplicates).
        """
        added = 0
        with self._connect() as conn:
            for f in files:
                try:
                    conn.execute(
                        """INSERT INTO files (path, source_type, parent_cbz, original_size)
                           VALUES (?, ?, ?, ?)""",
                        (
                            f["path"],
                            f.get("source_type", "file"),
                            f.get("parent_cbz"),
                            f.get("original_size"),
                        ),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        return added

    def get_pending_files(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get files pending processing.

        Args:
            limit: Maximum number of files to return.

        Returns:
            List of file dictionaries.
        """
        with self._connect() as conn:
            query = "SELECT * FROM files WHERE status = ? ORDER BY id"
            params: tuple[Any, ...] = (FileStatus.PENDING.value,)
            if limit:
                query += " LIMIT ?"
                params = (FileStatus.PENDING.value, limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_files_by_cbz(self, cbz_path: str) -> list[dict[str, Any]]:
        """Get all entries for a CBZ file.

        Args:
            cbz_path: Path to the CBZ file.

        Returns:
            List of file dictionaries.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE parent_cbz = ? ORDER BY id",
                (cbz_path,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_status(
        self,
        path: str,
        status: FileStatus,
        error_message: str | None = None,
        compressed_size: int | None = None,
        processing_time: float | None = None,
        preset: str | None = None,
    ) -> None:
        """Update file processing status.

        Args:
            path: File path.
            status: New status.
            error_message: Error message if failed.
            compressed_size: Compressed file size in bytes.
            processing_time: Processing time in seconds.
            preset: Preset used for compression.
        """
        with self._connect() as conn:
            conn.execute(
                """UPDATE files SET
                    status = ?,
                    error_message = ?,
                    compressed_size = ?,
                    processing_time = ?,
                    preset = ?,
                    updated_at = CURRENT_TIMESTAMP
                   WHERE path = ?""",
                (status.value, error_message, compressed_size, processing_time, preset, path),
            )
            conn.commit()

    def mark_processing(self, path: str) -> None:
        """Mark a file as currently being processed.

        Args:
            path: File path.
        """
        self.update_status(path, FileStatus.PROCESSING)

    def mark_completed(
        self,
        path: str,
        compressed_size: int,
        processing_time: float,
        preset: str,
    ) -> None:
        """Mark a file as successfully processed.

        Args:
            path: File path.
            compressed_size: Compressed file size in bytes.
            processing_time: Processing time in seconds.
            preset: Preset used for compression.
        """
        self.update_status(
            path,
            FileStatus.COMPLETED,
            compressed_size=compressed_size,
            processing_time=processing_time,
            preset=preset,
        )

    def mark_failed(self, path: str, error_message: str) -> None:
        """Mark a file as failed.

        Args:
            path: File path.
            error_message: Error description.
        """
        self.update_status(path, FileStatus.FAILED, error_message=error_message)

    def mark_skipped(self, path: str, reason: str = "") -> None:
        """Mark a file as skipped.

        Args:
            path: File path.
            reason: Reason for skipping.
        """
        self.update_status(path, FileStatus.SKIPPED, error_message=reason)

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._connect() as conn:
            # Count by status
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM files GROUP BY status"
            ).fetchall()
            counts = {row["status"]: row["cnt"] for row in rows}

            total = sum(counts.values())
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            pending = counts.get("pending", 0)
            processing = counts.get("processing", 0)

            # Size savings
            size_row = conn.execute(
                """SELECT
                    SUM(original_size) as total_original,
                    SUM(compressed_size) as total_compressed,
                    SUM(processing_time) as total_time
                   FROM files WHERE status = 'completed'"""
            ).fetchone()

            total_original = size_row["total_original"] or 0
            total_compressed = size_row["total_compressed"] or 0
            total_time = size_row["total_time"] or 0.0

            savings = total_original - total_compressed if total_original > 0 else 0
            ratio = (total_compressed / total_original * 100) if total_original > 0 else 0

            return {
                "total": total,
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "processing": processing,
                "total_original_bytes": total_original,
                "total_compressed_bytes": total_compressed,
                "savings_bytes": savings,
                "compression_ratio": ratio,
                "total_time_seconds": total_time,
            }

    def reset_processing(self) -> int:
        """Reset files stuck in 'processing' status back to 'pending'.

        This is useful after a crash or interruption.

        Returns:
            Number of files reset.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE files SET status = ? WHERE status = ?",
                (FileStatus.PENDING.value, FileStatus.PROCESSING.value),
            )
            conn.commit()
            return cursor.rowcount
