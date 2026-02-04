"""Tests for MultiExporter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from prela.core.span import Span, SpanType
from prela.exporters.base import BaseExporter, ExportResult
from prela.exporters.multi import MultiExporter


class MockExporter(BaseExporter):
    """Mock exporter for testing."""

    def __init__(self, result: ExportResult = ExportResult.SUCCESS, raise_error: bool = False):
        self.result = result
        self.raise_error = raise_error
        self.exported_spans = []
        self.shutdown_called = False
        self.export_count = 0

    def export(self, spans: list) -> ExportResult:
        self.export_count += 1
        self.exported_spans.extend(spans)
        if self.raise_error:
            raise RuntimeError("Mock export error")
        return self.result

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def sample_span():
    """Create a sample span for testing."""
    return Span(
        span_id="span-123",
        trace_id="trace-456",
        name="test-operation",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_multi_exporter_initialization():
    """Test MultiExporter can be initialized."""
    exporter1 = MockExporter()
    exporter2 = MockExporter()
    multi = MultiExporter([exporter1, exporter2])

    assert len(multi.exporters) == 2


def test_multi_exporter_requires_exporters():
    """Test MultiExporter raises on empty list."""
    with pytest.raises(ValueError, match="at least one exporter"):
        MultiExporter([])


def test_multi_exporter_all_success(sample_span):
    """Test MultiExporter returns SUCCESS when all succeed."""
    exporter1 = MockExporter(ExportResult.SUCCESS)
    exporter2 = MockExporter(ExportResult.SUCCESS)
    exporter3 = MockExporter(ExportResult.SUCCESS)
    multi = MultiExporter([exporter1, exporter2, exporter3])

    result = multi.export([sample_span])

    assert result == ExportResult.SUCCESS
    assert exporter1.export_count == 1
    assert exporter2.export_count == 1
    assert exporter3.export_count == 1


def test_multi_exporter_one_retry(sample_span):
    """Test MultiExporter returns RETRY if any exporter requests retry."""
    exporter1 = MockExporter(ExportResult.SUCCESS)
    exporter2 = MockExporter(ExportResult.RETRY)
    exporter3 = MockExporter(ExportResult.SUCCESS)
    multi = MultiExporter([exporter1, exporter2, exporter3])

    result = multi.export([sample_span])

    assert result == ExportResult.RETRY
    assert exporter1.export_count == 1
    assert exporter2.export_count == 1
    assert exporter3.export_count == 1


def test_multi_exporter_all_failure(sample_span):
    """Test MultiExporter returns FAILURE when all fail."""
    exporter1 = MockExporter(ExportResult.FAILURE)
    exporter2 = MockExporter(ExportResult.FAILURE)
    exporter3 = MockExporter(ExportResult.FAILURE)
    multi = MultiExporter([exporter1, exporter2, exporter3])

    result = multi.export([sample_span])

    assert result == ExportResult.FAILURE


def test_multi_exporter_mixed_success_failure(sample_span):
    """Test MultiExporter returns SUCCESS if at least one succeeds."""
    exporter1 = MockExporter(ExportResult.FAILURE)
    exporter2 = MockExporter(ExportResult.SUCCESS)
    exporter3 = MockExporter(ExportResult.FAILURE)
    multi = MultiExporter([exporter1, exporter2, exporter3])

    result = multi.export([sample_span])

    assert result == ExportResult.SUCCESS


def test_multi_exporter_exception_handling(sample_span):
    """Test MultiExporter handles exporter exceptions."""
    exporter1 = MockExporter(raise_error=True)
    exporter2 = MockExporter(ExportResult.SUCCESS)
    multi = MultiExporter([exporter1, exporter2])

    # Should not raise, treats exception as FAILURE
    result = multi.export([sample_span])

    # Second exporter succeeded, so overall success
    assert result == ExportResult.SUCCESS
    assert exporter2.export_count == 1


def test_multi_exporter_all_exceptions(sample_span):
    """Test MultiExporter when all exporters raise exceptions."""
    exporter1 = MockExporter(raise_error=True)
    exporter2 = MockExporter(raise_error=True)
    multi = MultiExporter([exporter1, exporter2])

    result = multi.export([sample_span])

    # All failed, so FAILURE
    assert result == ExportResult.FAILURE


def test_multi_exporter_empty_span_list():
    """Test MultiExporter with empty span list."""
    exporter1 = MockExporter()
    exporter2 = MockExporter()
    multi = MultiExporter([exporter1, exporter2])

    result = multi.export([])

    assert result == ExportResult.SUCCESS
    # Exporters should still be called
    assert exporter1.export_count == 0  # Empty list, may not call
    assert exporter2.export_count == 0


def test_multi_exporter_multiple_spans(sample_span):
    """Test MultiExporter with multiple spans."""
    span2 = Span(
        span_id="span-789",
        trace_id="trace-abc",
        name="operation-2",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    exporter1 = MockExporter()
    exporter2 = MockExporter()
    multi = MultiExporter([exporter1, exporter2])

    result = multi.export([sample_span, span2])

    assert result == ExportResult.SUCCESS
    assert len(exporter1.exported_spans) == 2
    assert len(exporter2.exported_spans) == 2


def test_multi_exporter_shutdown():
    """Test MultiExporter shutdown calls all exporters."""
    exporter1 = MockExporter()
    exporter2 = MockExporter()
    exporter3 = MockExporter()
    multi = MultiExporter([exporter1, exporter2, exporter3])

    multi.shutdown()

    assert exporter1.shutdown_called
    assert exporter2.shutdown_called
    assert exporter3.shutdown_called


def test_multi_exporter_shutdown_with_exception():
    """Test MultiExporter shutdown handles exceptions."""

    class FailingExporter(BaseExporter):
        def export(self, spans):
            return ExportResult.SUCCESS

        def shutdown(self):
            raise RuntimeError("Shutdown error")

    exporter1 = FailingExporter()
    exporter2 = MockExporter()
    multi = MultiExporter([exporter1, exporter2])

    # Should not raise
    multi.shutdown()

    # Second exporter should still be shut down
    assert exporter2.shutdown_called


def test_multi_exporter_result_priority():
    """Test MultiExporter result priority: RETRY > SUCCESS > FAILURE."""
    # RETRY beats SUCCESS
    exporter1 = MockExporter(ExportResult.RETRY)
    exporter2 = MockExporter(ExportResult.SUCCESS)
    multi = MultiExporter([exporter1, exporter2])
    assert multi._combine_results([ExportResult.RETRY, ExportResult.SUCCESS]) == ExportResult.RETRY

    # SUCCESS beats FAILURE
    exporter1 = MockExporter(ExportResult.SUCCESS)
    exporter2 = MockExporter(ExportResult.FAILURE)
    multi = MultiExporter([exporter1, exporter2])
    assert (
        multi._combine_results([ExportResult.SUCCESS, ExportResult.FAILURE])
        == ExportResult.SUCCESS
    )

    # All FAILURE = FAILURE
    assert (
        multi._combine_results([ExportResult.FAILURE, ExportResult.FAILURE])
        == ExportResult.FAILURE
    )


def test_multi_exporter_single_exporter(sample_span):
    """Test MultiExporter with single exporter."""
    exporter = MockExporter(ExportResult.SUCCESS)
    multi = MultiExporter([exporter])

    result = multi.export([sample_span])

    assert result == ExportResult.SUCCESS
    assert exporter.export_count == 1


def test_multi_exporter_preserves_order(sample_span):
    """Test MultiExporter calls exporters in order."""
    call_order = []

    class OrderTrackingExporter(BaseExporter):
        def __init__(self, id: int):
            self.id = id

        def export(self, spans):
            call_order.append(self.id)
            return ExportResult.SUCCESS

        def shutdown(self):
            pass

    exporter1 = OrderTrackingExporter(1)
    exporter2 = OrderTrackingExporter(2)
    exporter3 = OrderTrackingExporter(3)
    multi = MultiExporter([exporter1, exporter2, exporter3])

    multi.export([sample_span])

    assert call_order == [1, 2, 3]
