"""File exporter for writing spans to JSONL files."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from prela.core.span import Span
from prela.exporters.base import BaseExporter, ExportResult


class FileExporter(BaseExporter):
    """
    Export spans to JSONL files with rotation and trace management.

    Features:
    - Thread-safe writes using a lock
    - Automatic directory creation
    - Date-based file naming with sequence numbers
    - Optional file rotation based on size
    - Trace retrieval by trace_id
    - Trace listing by date range
    - Old trace cleanup

    File naming: traces-{date}-{sequence}.jsonl
    Example: traces-2025-01-26-001.jsonl

    The JSONL format writes one JSON object per line, making it easy to
    stream and process large trace files.

    Example:
        ```python
        from prela.core.tracer import Tracer
        from prela.exporters.file import FileExporter

        tracer = Tracer(
            service_name="my-app",
            exporter=FileExporter(
                directory="./traces",
                max_file_size_mb=100,
                rotate=True
            )
        )

        with tracer.span("operation") as span:
            span.set_attribute("key", "value")
        # Span is automatically written to ./traces/traces-2025-01-26-001.jsonl
        ```
    """

    def __init__(
        self,
        directory: str | Path = "./traces",
        format: str = "jsonl",
        max_file_size_mb: int = 100,
        rotate: bool = True,
    ):
        """
        Initialize file exporter.

        Args:
            directory: Directory to store trace files (e.g., "./traces")
            format: File format - "jsonl" or "ndjson" (both are equivalent)
            max_file_size_mb: Maximum file size in MB before rotation
            rotate: Whether to rotate files when size exceeded
        """
        self.directory = Path(directory)
        self.format = format if format in ("jsonl", "ndjson") else "jsonl"
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.rotate = rotate
        self._lock = threading.Lock()
        self._current_file: Path | None = None
        self._current_sequence = 1

        # Create directory if needed
        self.directory.mkdir(parents=True, exist_ok=True)

        # Initialize current file path
        self._update_current_file()

    def _update_current_file(self) -> None:
        """Update the current file path based on date and sequence."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Find the next available sequence number for today
        self._current_sequence = 1
        while True:
            filename = f"traces-{date_str}-{self._current_sequence:03d}.{self.format}"
            file_path = self.directory / filename

            # If file doesn't exist or is under size limit, use it
            if not file_path.exists():
                self._current_file = file_path
                break

            if self.rotate and file_path.stat().st_size >= self.max_file_size_bytes:
                # File is full, try next sequence
                self._current_sequence += 1
            else:
                # File exists and has space
                self._current_file = file_path
                break

    def _check_rotation(self) -> None:
        """Check if current file needs rotation and update if needed."""
        if not self.rotate or not self._current_file:
            return

        # Check if date changed
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_date = self._current_file.stem.split("-")[1:4]  # ["2025", "01", "26"]
        file_date_str = "-".join(file_date)

        if current_date != file_date_str:
            # Date changed, reset to sequence 1
            self._update_current_file()
            return

        # Check file size
        if self._current_file.exists() and self._current_file.stat().st_size >= self.max_file_size_bytes:
            self._current_sequence += 1
            self._update_current_file()

    def export(self, spans: list[Span]) -> ExportResult:
        """
        Export spans to file.

        Args:
            spans: List of spans to export

        Returns:
            ExportResult.SUCCESS if successful, ExportResult.FAILURE otherwise
        """
        if not spans:
            return ExportResult.SUCCESS

        try:
            with self._lock:
                # Check if rotation needed
                self._check_rotation()

                # Append spans
                with open(self._current_file, "a", encoding="utf-8") as f:
                    for span in spans:
                        span_dict = span.to_dict()
                        json_line = json.dumps(span_dict)
                        f.write(json_line + "\n")

            return ExportResult.SUCCESS
        except Exception:
            # Silently fail - don't crash user code due to export failures
            return ExportResult.FAILURE

    def get_trace_file(self, trace_id: str) -> Path | None:
        """
        Find the file containing a specific trace.

        Args:
            trace_id: Trace ID to search for

        Returns:
            Path to the file containing the trace, or None if not found
        """
        # Search through all trace files
        pattern = f"traces-*.{self.format}"
        for file_path in sorted(self.directory.glob(pattern)):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("trace_id") == trace_id:
                                return file_path
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

        return None

    def read_traces(self, trace_id: str | None = None) -> Iterator[Span]:
        """
        Read traces from files.

        Args:
            trace_id: Optional trace ID to filter by. If None, reads all traces.

        Yields:
            Span objects from the trace files
        """
        pattern = f"traces-*.{self.format}"

        # If trace_id provided, only read from that file
        if trace_id:
            file_path = self.get_trace_file(trace_id)
            if file_path:
                files_to_read = [file_path]
            else:
                files_to_read = []
        else:
            files_to_read = sorted(self.directory.glob(pattern))

        for file_path in files_to_read:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if trace_id is None or data.get("trace_id") == trace_id:
                                span = Span.from_dict(data)
                                yield span
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # Skip malformed lines
                            continue
            except Exception:
                # Skip files that can't be read
                continue

    def list_traces(self, start: datetime, end: datetime) -> list[str]:
        """
        List trace IDs within a date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of unique trace IDs found in the date range
        """
        trace_ids = set()
        pattern = f"traces-*.{self.format}"

        for file_path in sorted(self.directory.glob(pattern)):
            # Extract date from filename: traces-2025-01-26-001.jsonl
            try:
                parts = file_path.stem.split("-")
                if len(parts) < 5:
                    continue

                file_date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

                # Check if file date is in range (add 1 day to end for inclusive search)
                if not (start.date() <= file_date.date() <= end.date()):
                    continue

                # Read traces from this file
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            # Check if span timestamp is in range (inclusive)
                            span_time = datetime.fromisoformat(data["started_at"])
                            if start <= span_time <= end:
                                trace_ids.add(data["trace_id"])
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except Exception:
                continue

        return sorted(trace_ids)

    def cleanup_old_traces(self, days: int) -> int:
        """
        Delete trace files older than specified days.

        Args:
            days: Delete files older than this many days (0 means keep today and delete all older)

        Returns:
            Number of files deleted
        """
        if days < 0:
            raise ValueError("days must be non-negative")

        # Calculate cutoff date (start of day)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        deleted_count = 0
        pattern = f"traces-*.{self.format}"

        with self._lock:
            for file_path in self.directory.glob(pattern):
                try:
                    # Extract date from filename
                    parts = file_path.stem.split("-")
                    if len(parts) < 5:
                        continue

                    file_date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()

                    # Delete if older than cutoff (strictly less than)
                    if file_date < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1

                        # Reset current file if we deleted it
                        if file_path == self._current_file:
                            self._update_current_file()
                except Exception:
                    # Skip files that can't be processed
                    continue

        return deleted_count

    def shutdown(self) -> None:
        """
        Shutdown the exporter.

        No cleanup needed for file exporter - file handle is closed
        after each write.
        """
        pass
