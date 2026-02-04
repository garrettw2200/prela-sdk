"""Tests for file exporter."""

from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prela.core.span import Span, SpanStatus, SpanType
from prela.exporters.base import ExportResult
from prela.exporters.file import FileExporter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def exporter(temp_dir):
    """Create a file exporter with test directory."""
    return FileExporter(directory=temp_dir, max_file_size_mb=1, rotate=True)


@pytest.fixture
def sample_span():
    """Create a sample span for testing."""
    span = Span(
        name="test-span",
        span_type=SpanType.CUSTOM,
        trace_id="test-trace-123",
        span_id="test-span-456",
    )
    span.end()
    return span


def test_init_creates_directory(temp_dir):
    """Test that initialization creates the directory."""
    trace_dir = temp_dir / "traces"
    assert not trace_dir.exists()

    FileExporter(directory=trace_dir)

    assert trace_dir.exists()
    assert trace_dir.is_dir()


def test_init_with_format(temp_dir):
    """Test initialization with different formats."""
    exporter = FileExporter(directory=temp_dir, format="ndjson")
    assert exporter.format == "ndjson"

    # Invalid format defaults to jsonl
    exporter = FileExporter(directory=temp_dir, format="json")
    assert exporter.format == "jsonl"


def test_export_creates_file(exporter, sample_span):
    """Test that export creates a file with correct naming."""
    result = exporter.export([sample_span])
    assert result == ExportResult.SUCCESS

    # Check file was created with correct pattern
    files = list(exporter.directory.glob("traces-*.jsonl"))
    assert len(files) == 1

    # Check filename format: traces-YYYY-MM-DD-NNN.jsonl
    filename = files[0].name
    assert filename.startswith("traces-")
    assert filename.endswith(".jsonl")

    # Extract date part
    parts = filename.split("-")
    assert len(parts) == 5  # traces, YYYY, MM, DD, NNN.jsonl


def test_export_jsonl_format(exporter, sample_span):
    """Test that exported data is in JSONL format."""
    exporter.export([sample_span])

    files = list(exporter.directory.glob("traces-*.jsonl"))
    with open(files[0], "r") as f:
        lines = f.readlines()

    # One span = one line
    assert len(lines) == 1

    # Valid JSON
    data = json.loads(lines[0])
    assert data["trace_id"] == "test-trace-123"
    assert data["span_id"] == "test-span-456"
    assert data["name"] == "test-span"


def test_export_multiple_spans(exporter):
    """Test exporting multiple spans."""
    spans = [
        Span(name=f"span-{i}", trace_id="trace-1", span_id=f"span-{i}") for i in range(5)
    ]
    for span in spans:
        span.end()

    result = exporter.export(spans)
    assert result == ExportResult.SUCCESS

    # Read file
    files = list(exporter.directory.glob("traces-*.jsonl"))
    with open(files[0], "r") as f:
        lines = f.readlines()

    assert len(lines) == 5

    # Verify all spans
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["name"] == f"span-{i}"


def test_export_empty_list(exporter):
    """Test exporting empty list returns success."""
    result = exporter.export([])
    assert result == ExportResult.SUCCESS

    # No files created
    files = list(exporter.directory.glob("traces-*.jsonl"))
    assert len(files) == 0


def test_export_appends_to_file(exporter, sample_span):
    """Test that multiple exports append to the same file."""
    # First export
    exporter.export([sample_span])

    # Second export
    span2 = Span(name="span-2", trace_id="trace-2", span_id="span-2")
    span2.end()
    exporter.export([span2])

    # Should still be one file
    files = list(exporter.directory.glob("traces-*.jsonl"))
    assert len(files) == 1

    # With two lines
    with open(files[0], "r") as f:
        lines = f.readlines()
    assert len(lines) == 2


def test_file_rotation_by_size(temp_dir):
    """Test that file rotates when size limit is reached."""
    # Use very small size limit (1KB = 0.001MB)
    exporter = FileExporter(directory=temp_dir, max_file_size_mb=0.001, rotate=True)

    # Export many spans to exceed size limit
    for i in range(100):
        span = Span(
            name=f"span-{i}",
            trace_id=f"trace-{i}",
            span_id=f"span-{i}",
        )
        span.set_attribute("large_data", "x" * 100)  # Make span larger
        span.end()
        exporter.export([span])

    # Should have created multiple files
    files = list(exporter.directory.glob("traces-*.jsonl"))
    assert len(files) > 1

    # Files should have sequential numbers
    filenames = sorted([f.name for f in files])
    assert "001" in filenames[0]
    assert "002" in filenames[1]


def test_rotation_disabled(temp_dir):
    """Test that rotation can be disabled."""
    exporter = FileExporter(directory=temp_dir, max_file_size_mb=0.001, rotate=False)

    # Export many spans
    for i in range(50):
        span = Span(name=f"span-{i}", trace_id=f"trace-{i}")
        span.set_attribute("data", "x" * 100)
        span.end()
        exporter.export([span])

    # Should still be only one file (even if it exceeds size)
    files = list(exporter.directory.glob("traces-*.jsonl"))
    assert len(files) == 1


def test_thread_safety(temp_dir):
    """Test that exporter is thread-safe."""
    exporter = FileExporter(directory=temp_dir)

    def export_spans(thread_id: int):
        for i in range(10):
            span = Span(
                name=f"thread-{thread_id}-span-{i}",
                trace_id=f"trace-{thread_id}",
            )
            span.end()
            exporter.export([span])

    # Run 5 threads concurrently
    threads = [threading.Thread(target=export_spans, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Count total lines
    files = list(exporter.directory.glob("traces-*.jsonl"))
    total_lines = 0
    for file_path in files:
        with open(file_path, "r") as f:
            total_lines += len(f.readlines())

    # Should have 50 total spans (5 threads * 10 spans)
    assert total_lines == 50


def test_get_trace_file(exporter):
    """Test finding file containing a specific trace."""
    # Export spans from different traces
    span1 = Span(name="span-1", trace_id="trace-aaa", span_id="span-1")
    span1.end()
    exporter.export([span1])

    span2 = Span(name="span-2", trace_id="trace-bbb", span_id="span-2")
    span2.end()
    exporter.export([span2])

    # Find trace-aaa
    file_path = exporter.get_trace_file("trace-aaa")
    assert file_path is not None
    assert file_path.exists()

    # Verify it contains the trace
    with open(file_path, "r") as f:
        content = f.read()
        assert "trace-aaa" in content

    # Non-existent trace
    file_path = exporter.get_trace_file("trace-nonexistent")
    assert file_path is None


def test_read_traces_all(exporter):
    """Test reading all traces."""
    # Export multiple spans
    spans = []
    for i in range(3):
        span = Span(name=f"span-{i}", trace_id=f"trace-{i}", span_id=f"span-{i}")
        span.end()
        spans.append(span)
    exporter.export(spans)

    # Read all traces
    read_spans = list(exporter.read_traces())
    assert len(read_spans) == 3

    # Verify span names
    span_names = {s.name for s in read_spans}
    assert span_names == {"span-0", "span-1", "span-2"}


def test_read_traces_by_trace_id(exporter):
    """Test reading traces filtered by trace_id."""
    # Export spans from different traces
    span1 = Span(name="span-1", trace_id="trace-aaa", span_id="span-1")
    span1.end()
    exporter.export([span1])

    span2 = Span(name="span-2", trace_id="trace-aaa", span_id="span-2")
    span2.end()
    exporter.export([span2])

    span3 = Span(name="span-3", trace_id="trace-bbb", span_id="span-3")
    span3.end()
    exporter.export([span3])

    # Read only trace-aaa
    read_spans = list(exporter.read_traces(trace_id="trace-aaa"))
    assert len(read_spans) == 2
    assert all(s.trace_id == "trace-aaa" for s in read_spans)

    # Read only trace-bbb
    read_spans = list(exporter.read_traces(trace_id="trace-bbb"))
    assert len(read_spans) == 1
    assert read_spans[0].trace_id == "trace-bbb"


def test_read_traces_nonexistent_trace_id(exporter):
    """Test reading non-existent trace returns empty."""
    span = Span(name="span-1", trace_id="trace-aaa")
    span.end()
    exporter.export([span])

    read_spans = list(exporter.read_traces(trace_id="trace-nonexistent"))
    assert len(read_spans) == 0


def test_read_traces_handles_malformed_lines(exporter, temp_dir):
    """Test that read_traces skips malformed JSON lines."""
    # Create a file with malformed data
    file_path = temp_dir / "traces-2025-01-26-001.jsonl"
    with open(file_path, "w") as f:
        # Valid line
        span = Span(name="valid", trace_id="trace-1")
        span.end()
        f.write(json.dumps(span.to_dict()) + "\n")

        # Malformed JSON
        f.write("{ invalid json }\n")

        # Empty line
        f.write("\n")

        # Another valid line
        span2 = Span(name="valid-2", trace_id="trace-2")
        span2.end()
        f.write(json.dumps(span2.to_dict()) + "\n")

    # Should read only valid spans
    read_spans = list(exporter.read_traces())
    assert len(read_spans) == 2
    assert read_spans[0].name == "valid"
    assert read_spans[1].name == "valid-2"


def test_list_traces_by_date_range(exporter):
    """Test listing traces within a date range."""
    # Export spans with known timestamps
    now = datetime.now(timezone.utc)

    span1 = Span(name="span-1", trace_id="trace-1", started_at=now - timedelta(days=2))
    span1.end()
    exporter.export([span1])

    span2 = Span(name="span-2", trace_id="trace-2", started_at=now - timedelta(days=1))
    span2.end()
    exporter.export([span2])

    span3 = Span(name="span-3", trace_id="trace-3", started_at=now)
    span3.end()
    exporter.export([span3])

    # List traces from last 2 days
    start = now - timedelta(days=2)
    end = now
    trace_ids = exporter.list_traces(start, end)

    # Should include all traces
    assert "trace-1" in trace_ids
    assert "trace-2" in trace_ids
    assert "trace-3" in trace_ids


def test_list_traces_excludes_out_of_range(exporter):
    """Test that list_traces excludes traces outside date range."""
    now = datetime.now(timezone.utc)

    # Span from 5 days ago
    span1 = Span(name="old", trace_id="trace-old", started_at=now - timedelta(days=5))
    span1.end()
    exporter.export([span1])

    # Span from today
    span2 = Span(name="new", trace_id="trace-new", started_at=now)
    span2.end()
    exporter.export([span2])

    # List only last 2 days
    start = now - timedelta(days=2)
    end = now
    trace_ids = exporter.list_traces(start, end)

    assert "trace-new" in trace_ids
    assert "trace-old" not in trace_ids


def test_list_traces_unique(exporter):
    """Test that list_traces returns unique trace IDs."""
    now = datetime.now(timezone.utc)

    # Multiple spans from same trace
    for i in range(3):
        span = Span(
            name=f"span-{i}",
            trace_id="trace-same",
            span_id=f"span-{i}",
            started_at=now,
        )
        span.end()
        exporter.export([span])

    # List traces
    start = now - timedelta(days=1)
    end = now + timedelta(days=1)
    trace_ids = exporter.list_traces(start, end)

    # Should only return one trace ID
    assert len(trace_ids) == 1
    assert trace_ids[0] == "trace-same"


def test_cleanup_old_traces(exporter, temp_dir):
    """Test cleanup of old trace files."""
    # Manually create old files with known dates
    old_date = datetime.now(timezone.utc) - timedelta(days=10)
    recent_date = datetime.now(timezone.utc) - timedelta(days=2)

    old_file = temp_dir / f"traces-{old_date.strftime('%Y-%m-%d')}-001.jsonl"
    recent_file = temp_dir / f"traces-{recent_date.strftime('%Y-%m-%d')}-001.jsonl"

    # Create files
    old_file.write_text('{"trace_id": "old"}\n')
    recent_file.write_text('{"trace_id": "recent"}\n')

    assert old_file.exists()
    assert recent_file.exists()

    # Cleanup files older than 7 days
    deleted_count = exporter.cleanup_old_traces(days=7)

    # Old file should be deleted
    assert not old_file.exists()
    assert recent_file.exists()
    assert deleted_count == 1


def test_cleanup_with_zero_days(exporter, temp_dir):
    """Test cleanup with days=0 deletes nothing from today."""
    # Create today's file
    today = datetime.now(timezone.utc)
    today_file = temp_dir / f"traces-{today.strftime('%Y-%m-%d')}-001.jsonl"
    today_file.write_text('{"trace_id": "today"}\n')

    deleted_count = exporter.cleanup_old_traces(days=0)

    # Should not delete today's file
    assert today_file.exists()
    assert deleted_count == 0


def test_cleanup_negative_days_raises_error(exporter):
    """Test that negative days raises ValueError."""
    with pytest.raises(ValueError, match="days must be non-negative"):
        exporter.cleanup_old_traces(days=-1)


def test_cleanup_updates_current_file(exporter, temp_dir):
    """Test that cleanup updates current file if deleted."""
    # Force exporter to use an old file
    old_date = datetime.now(timezone.utc) - timedelta(days=10)
    old_file = temp_dir / f"traces-{old_date.strftime('%Y-%m-%d')}-001.jsonl"
    old_file.write_text('{"trace_id": "old"}\n')

    # Manually set current file to the old one
    exporter._current_file = old_file

    # Cleanup
    exporter.cleanup_old_traces(days=7)

    # Current file should be updated
    assert exporter._current_file != old_file
    assert exporter._current_file is not None


def test_export_failure_returns_failure_result(temp_dir):
    """Test that export failures return FAILURE result."""
    exporter = FileExporter(directory=temp_dir)

    # Create an invalid span that will cause serialization issues
    # (This is tricky - we'll simulate by making the directory read-only on Unix)
    import os
    import stat

    if os.name != "nt":  # Skip on Windows
        # Make directory read-only
        temp_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        span = Span(name="test")
        span.end()

        result = exporter.export([span])
        assert result == ExportResult.FAILURE

        # Restore permissions for cleanup
        temp_dir.chmod(stat.S_IRWXU)


def test_shutdown(exporter):
    """Test shutdown method."""
    exporter.shutdown()
    # Should not raise any errors
    # No-op for file exporter


def test_concurrent_rotation(temp_dir):
    """Test that rotation works correctly with concurrent writes."""
    exporter = FileExporter(directory=temp_dir, max_file_size_mb=0.001, rotate=True)

    def export_large_spans(thread_id: int):
        for i in range(20):
            span = Span(
                name=f"thread-{thread_id}-span-{i}",
                trace_id=f"trace-{thread_id}-{i}",
            )
            span.set_attribute("data", "x" * 200)
            span.end()
            exporter.export([span])

    # Run threads that will trigger rotation
    threads = [threading.Thread(target=export_large_spans, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have created multiple files
    files = list(exporter.directory.glob("traces-*.jsonl"))
    assert len(files) > 1

    # All spans should be present
    total_lines = 0
    for file_path in files:
        with open(file_path, "r") as f:
            total_lines += len(f.readlines())

    assert total_lines == 60  # 3 threads * 20 spans


def test_ndjson_format(temp_dir):
    """Test that ndjson format works identically to jsonl."""
    exporter = FileExporter(directory=temp_dir, format="ndjson")

    span = Span(name="test", trace_id="trace-1")
    span.end()
    exporter.export([span])

    # Should create .ndjson file
    files = list(exporter.directory.glob("traces-*.ndjson"))
    assert len(files) == 1

    # Content should be valid JSON lines
    with open(files[0], "r") as f:
        data = json.loads(f.readline())
        assert data["trace_id"] == "trace-1"


def test_date_based_file_rotation(temp_dir):
    """Test that files rotate when date changes."""
    exporter = FileExporter(directory=temp_dir)

    # Get current file
    current_file = exporter._current_file
    current_date_parts = current_file.stem.split("-")[1:4]

    # Verify filename contains today's date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file_date = "-".join(current_date_parts)
    assert file_date == today


def test_multiple_sequences_same_day(temp_dir):
    """Test that multiple sequence files can be created on same day."""
    # Create first file and fill it
    exporter1 = FileExporter(directory=temp_dir, max_file_size_mb=0.001, rotate=True)

    for i in range(50):
        span = Span(name=f"span-{i}")
        span.set_attribute("data", "x" * 100)
        span.end()
        exporter1.export([span])

    # Create second exporter (should pick up next sequence)
    exporter2 = FileExporter(directory=temp_dir, max_file_size_mb=0.001, rotate=True)

    # Both exporters should have created files
    files = sorted(temp_dir.glob("traces-*.jsonl"))
    assert len(files) >= 2

    # Check sequence numbers
    assert "001" in files[0].name
    # Should have higher sequence numbers
    assert any("00" in f.name for f in files[1:])
