"""Tests for ConsoleExporter."""

from __future__ import annotations

import time
from io import StringIO
from unittest.mock import patch

import pytest

from prela.core.clock import now
from prela.core.span import Span, SpanEvent, SpanStatus, SpanType
from prela.exporters.base import ExportResult
from prela.exporters.console import RICH_AVAILABLE, ConsoleExporter


def test_console_exporter_initialization():
    """Test console exporter can be initialized."""
    exporter = ConsoleExporter()
    assert exporter.verbosity == "normal"
    assert exporter.color is True or not RICH_AVAILABLE
    assert exporter.show_timestamps is True


def test_console_exporter_initialization_custom():
    """Test console exporter with custom settings."""
    exporter = ConsoleExporter(
        verbosity="verbose", color=False, show_timestamps=False
    )
    assert exporter.verbosity == "verbose"
    assert exporter.color is False
    assert exporter.show_timestamps is False


def test_console_exporter_invalid_verbosity():
    """Test console exporter with invalid verbosity."""
    with pytest.raises(ValueError, match="Invalid verbosity"):
        ConsoleExporter(verbosity="invalid")


def test_console_exporter_export_single_span():
    """Test exporting a single span."""
    exporter = ConsoleExporter(color=False, show_timestamps=False)
    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.CUSTOM,
        started_at=now(),
        attributes={"key": "value"},
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([span])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    assert "test-operation" in output
    assert "custom" in output.lower()
    assert "✓" in output  # Success indicator


def test_console_exporter_export_nested_spans():
    """Test exporting nested spans (parent-child relationship)."""
    exporter = ConsoleExporter(color=False, show_timestamps=False, verbosity="normal")

    # Create parent span
    parent = Span(
        span_id="parent-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="parent-operation",
        span_type=SpanType.AGENT,
        started_at=now(),
        attributes={},
    )

    # Create child span
    child = Span(
        span_id="child-1",
        trace_id="trace-1",
        parent_span_id="parent-1",
        name="child-operation",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={"llm.model": "gpt-4"},
    )

    # End spans
    time.sleep(0.01)
    child.end()
    parent.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([parent, child])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    assert "parent-operation" in output
    assert "child-operation" in output
    assert "agent" in output.lower()
    assert "llm" in output.lower()

    # Check tree structure indicators
    assert "└─" in output or "├─" in output


def test_console_exporter_verbosity_minimal():
    """Test minimal verbosity (no attributes)."""
    exporter = ConsoleExporter(
        verbosity="minimal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={"llm.model": "gpt-4", "llm.input_tokens": 100},
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([span])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    assert "test-operation" in output
    # Minimal mode should NOT show attributes
    assert "gpt-4" not in output
    assert "100" not in output


def test_console_exporter_verbosity_normal():
    """Test normal verbosity (key attributes)."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={
            "llm.model": "gpt-4",
            "llm.input_tokens": 100,
            "llm.output_tokens": 50,
            "irrelevant.attribute": "should-not-show",
        },
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([span])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    assert "test-operation" in output
    # Normal mode should show key attributes
    assert "gpt-4" in output
    assert "100" in output and "50" in output
    # But not irrelevant attributes
    assert "should-not-show" not in output


def test_console_exporter_verbosity_verbose():
    """Test verbose verbosity (all attributes + events)."""
    exporter = ConsoleExporter(
        verbosity="verbose", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={
            "llm.model": "gpt-4",
            "llm.input_tokens": 100,
            "custom.attribute": "custom-value",
        },
    )
    span.add_event(
        SpanEvent(
            timestamp=now(),
            name="test-event",
            attributes={"event_key": "event_value"},
        )
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([span])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    assert "test-operation" in output
    # Verbose mode should show ALL attributes
    assert "gpt-4" in output
    assert "100" in output
    assert "custom-value" in output
    # And events
    assert "test-event" in output
    assert "event_value" in output


def test_console_exporter_llm_span_attributes():
    """Test LLM span key attribute extraction."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="llm-call",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={
            "llm.model": "gpt-4",
            "llm.input_tokens": 150,
            "llm.output_tokens": 89,
        },
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show model and tokens in format "150 → 89"
    assert "gpt-4" in output
    assert "150" in output
    assert "89" in output
    assert "→" in output


def test_console_exporter_tool_span_attributes():
    """Test TOOL span key attribute extraction."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="tool-call",
        span_type=SpanType.TOOL,
        started_at=now(),
        attributes={"tool.name": "web_search", "tool.input": "AI news 2024"},
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show tool name and input
    assert "web_search" in output
    assert "AI news 2024" in output


def test_console_exporter_retrieval_span_attributes():
    """Test RETRIEVAL span key attribute extraction."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="retriever-call",
        span_type=SpanType.RETRIEVAL,
        started_at=now(),
        attributes={
            "retriever.query": "What is the capital of France?",
            "retriever.document_count": 5,
        },
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show query and document count
    assert "capital of France" in output
    assert "5" in output


def test_console_exporter_embedding_span_attributes():
    """Test EMBEDDING span key attribute extraction."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="embedding-call",
        span_type=SpanType.EMBEDDING,
        started_at=now(),
        attributes={
            "embedding.model": "text-embedding-ada-002",
            "embedding.dimensions": 1536,
        },
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show model and dimensions
    assert "text-embedding-ada-002" in output
    assert "1536" in output


def test_console_exporter_error_span():
    """Test error span display."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="failed-operation",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={},
    )
    span.set_status(SpanStatus.ERROR, "Test error message")
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([span])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    assert "failed-operation" in output
    assert "✗" in output  # Error indicator
    assert "Test error message" in output


def test_console_exporter_duration_formatting():
    """Test duration formatting (µs, ms, s)."""
    exporter = ConsoleExporter(color=False, show_timestamps=False)

    # Create span with known duration
    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="timed-operation",
        span_type=SpanType.CUSTOM,
        started_at=now(),
        attributes={},
    )
    time.sleep(0.05)  # 50ms
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show duration in ms
    assert "ms" in output or "s" in output


def test_console_exporter_multiple_traces():
    """Test exporting spans from multiple traces."""
    exporter = ConsoleExporter(color=False, show_timestamps=False)

    spans = [
        Span(
            span_id="span-1",
            trace_id="trace-1",
            parent_span_id=None,
            name="operation-1",
            span_type=SpanType.CUSTOM,
            started_at=now(),
            attributes={},
        ),
        Span(
            span_id="span-2",
            trace_id="trace-2",
            parent_span_id=None,
            name="operation-2",
            span_type=SpanType.CUSTOM,
            started_at=now(),
            attributes={},
        ),
    ]

    for span in spans:
        span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export(spans)
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    # Should show both traces
    assert "trace-1" in output
    assert "trace-2" in output
    assert "operation-1" in output
    assert "operation-2" in output


def test_console_exporter_with_timestamps():
    """Test exporting with timestamps enabled."""
    exporter = ConsoleExporter(color=False, show_timestamps=True)

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.CUSTOM,
        started_at=now(),
        attributes={},
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show timestamp (HH:MM:SS.mmm format)
    assert "@" in output  # Timestamp separator


def test_console_exporter_shutdown():
    """Test console exporter shutdown (no-op)."""
    exporter = ConsoleExporter()
    exporter.shutdown()  # Should not raise


def test_console_exporter_empty_spans():
    """Test exporting empty span list."""
    exporter = ConsoleExporter()
    result = exporter.export([])
    assert result == ExportResult.SUCCESS


def test_console_exporter_truncate_long_attributes():
    """Test that long attribute values are truncated."""
    exporter = ConsoleExporter(
        verbosity="normal", color=False, show_timestamps=False
    )

    long_input = "x" * 100  # Very long input

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.TOOL,
        started_at=now(),
        attributes={"tool.name": "test", "tool.input": long_input},
    )
    span.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        exporter.export([span])
        output = fake_stdout.getvalue()

    # Should show truncated version with "..."
    assert "..." in output


def test_console_exporter_with_rich_if_available():
    """Test exporting with rich library if available."""
    exporter = ConsoleExporter(color=True)

    span = Span(
        span_id="span-1",
        trace_id="trace-1",
        parent_span_id=None,
        name="test-operation",
        span_type=SpanType.CUSTOM,
        started_at=now(),
        attributes={"key": "value"},
    )
    span.end()

    # This test will use rich if available, otherwise fall back to plain
    result = exporter.export([span])
    assert result == ExportResult.SUCCESS

    # If rich is available, console should be initialized
    if RICH_AVAILABLE:
        assert hasattr(exporter, "console")
    else:
        assert not hasattr(exporter, "console") or exporter.color is False


def test_console_exporter_complex_tree():
    """Test exporting complex span tree."""
    exporter = ConsoleExporter(color=False, show_timestamps=False, verbosity="normal")

    # Create complex tree:
    # root
    #   ├─ child1
    #   │   └─ grandchild1
    #   └─ child2

    root = Span(
        span_id="root",
        trace_id="trace-1",
        parent_span_id=None,
        name="root",
        span_type=SpanType.AGENT,
        started_at=now(),
        attributes={},
    )

    child1 = Span(
        span_id="child1",
        trace_id="trace-1",
        parent_span_id="root",
        name="child1",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={"llm.model": "gpt-4"},
    )

    grandchild1 = Span(
        span_id="grandchild1",
        trace_id="trace-1",
        parent_span_id="child1",
        name="grandchild1",
        span_type=SpanType.TOOL,
        started_at=now(),
        attributes={"tool.name": "search"},
    )

    child2 = Span(
        span_id="child2",
        trace_id="trace-1",
        parent_span_id="root",
        name="child2",
        span_type=SpanType.LLM,
        started_at=now(),
        attributes={"llm.model": "gpt-3.5"},
    )

    # End spans
    time.sleep(0.01)
    grandchild1.end()
    child1.end()
    child2.end()
    root.end()

    with patch("sys.stdout", new=StringIO()) as fake_stdout:
        result = exporter.export([root, child1, child2, grandchild1])
        output = fake_stdout.getvalue()

    assert result == ExportResult.SUCCESS
    # All spans should be present
    assert "root" in output
    assert "child1" in output
    assert "child2" in output
    assert "grandchild1" in output
    # Should show models
    assert "gpt-4" in output
    assert "gpt-3.5" in output
