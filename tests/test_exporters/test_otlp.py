"""Tests for OTLP exporter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from prela.core.span import Span, SpanStatus, SpanType
from prela.exporters.base import ExportResult

# Check if OTLP exporter is available
try:
    from prela.exporters.otlp import OTLPExporter

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not OTLP_AVAILABLE, reason="OTLP exporter not available")


@pytest.fixture
def sample_span():
    """Create a sample span for testing."""
    return Span(
        span_id="span-123",
        trace_id="trace-456",
        name="test-operation",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        attributes={"service.name": "test-service", "key1": "value1"},
    )


def test_otlp_exporter_initialization():
    """Test OTLP exporter can be initialized with defaults."""
    exporter = OTLPExporter()
    assert exporter.endpoint == "http://localhost:4318/v1/traces"
    assert exporter.headers["Content-Type"] == "application/json"
    assert exporter.timeout == 10
    assert not exporter.insecure


def test_otlp_exporter_custom_endpoint():
    """Test OTLP exporter with custom endpoint."""
    exporter = OTLPExporter(endpoint="http://tempo:4318/v1/traces")
    assert exporter.endpoint == "http://tempo:4318/v1/traces"


def test_otlp_exporter_custom_headers():
    """Test OTLP exporter with custom headers."""
    headers = {"X-Scope-OrgID": "tenant1", "Authorization": "Bearer token123"}
    exporter = OTLPExporter(headers=headers)
    assert exporter.headers["X-Scope-OrgID"] == "tenant1"
    assert exporter.headers["Authorization"] == "Bearer token123"
    assert exporter.headers["Content-Type"] == "application/json"


def test_otlp_exporter_invalid_endpoint():
    """Test OTLP exporter raises on invalid endpoint."""
    with pytest.raises(ValueError, match="Invalid endpoint URL"):
        OTLPExporter(endpoint="not-a-url")


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_success(mock_post, sample_span):
    """Test successful OTLP export."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.SUCCESS
    assert mock_post.called
    assert mock_post.call_args[0][0] == "http://localhost:4318/v1/traces"


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_empty_list(mock_post):
    """Test exporting empty span list."""
    exporter = OTLPExporter()
    result = exporter.export([])

    assert result == ExportResult.SUCCESS
    assert not mock_post.called


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_retry_on_429(mock_post, sample_span):
    """Test OTLP export returns RETRY on 429."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_post.return_value = mock_response

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.RETRY


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_retry_on_503(mock_post, sample_span):
    """Test OTLP export returns RETRY on 503."""
    mock_response = Mock()
    mock_response.status_code = 503
    mock_post.return_value = mock_response

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.RETRY


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_failure_on_4xx(mock_post, sample_span):
    """Test OTLP export returns FAILURE on other 4xx."""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"
    mock_post.return_value = mock_response

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.FAILURE


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_retry_on_timeout(mock_post, sample_span):
    """Test OTLP export returns RETRY on timeout."""
    import requests

    mock_post.side_effect = requests.exceptions.Timeout()

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.RETRY


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_retry_on_connection_error(mock_post, sample_span):
    """Test OTLP export returns RETRY on connection error."""
    import requests

    mock_post.side_effect = requests.exceptions.ConnectionError()

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.RETRY


@patch("prela.exporters.otlp.requests.post")
def test_otlp_export_failure_on_exception(mock_post, sample_span):
    """Test OTLP export returns FAILURE on unexpected exception."""
    mock_post.side_effect = ValueError("Unexpected error")

    exporter = OTLPExporter()
    result = exporter.export([sample_span])

    assert result == ExportResult.FAILURE


def test_otlp_span_conversion(sample_span):
    """Test Prela span to OTLP conversion."""
    exporter = OTLPExporter()
    otlp_data = exporter._spans_to_otlp([sample_span])

    assert "resourceSpans" in otlp_data
    assert len(otlp_data["resourceSpans"]) == 1

    resource_span = otlp_data["resourceSpans"][0]
    assert "resource" in resource_span
    assert "scopeSpans" in resource_span

    # Check resource attributes
    resource_attrs = resource_span["resource"]["attributes"]
    assert any(
        attr["key"] == "service.name" and attr["value"]["stringValue"] == "test-service"
        for attr in resource_attrs
    )

    # Check scope spans
    scope_spans = resource_span["scopeSpans"][0]
    assert scope_spans["scope"]["name"] == "prela"
    assert len(scope_spans["spans"]) == 1


def test_otlp_span_fields(sample_span):
    """Test OTLP span has all required fields."""
    exporter = OTLPExporter()
    otlp_span = exporter._span_to_otlp(sample_span)

    assert "traceId" in otlp_span
    assert "spanId" in otlp_span
    assert "name" in otlp_span
    assert otlp_span["name"] == "test-operation"
    assert "kind" in otlp_span
    assert "startTimeUnixNano" in otlp_span
    assert "endTimeUnixNano" in otlp_span
    assert "attributes" in otlp_span
    assert "events" in otlp_span
    assert "status" in otlp_span


def test_otlp_span_with_parent():
    """Test OTLP span with parent includes parentSpanId."""
    span = Span(
        span_id="child-123",
        trace_id="trace-456",
        parent_span_id="parent-789",
        name="child-operation",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    exporter = OTLPExporter()
    otlp_span = exporter._span_to_otlp(span)

    assert "parentSpanId" in otlp_span
    assert otlp_span["parentSpanId"]  # Not empty


def test_otlp_span_status_success():
    """Test OTLP span status for success."""
    span = Span(
        span_id="span-123",
        trace_id="trace-456",
        name="test",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
    )

    exporter = OTLPExporter()
    otlp_span = exporter._span_to_otlp(span)

    assert otlp_span["status"]["code"] == 1  # OK


def test_otlp_span_status_error():
    """Test OTLP span status for error."""
    span = Span(
        span_id="span-123",
        trace_id="trace-456",
        name="test",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        status=SpanStatus.ERROR,
        status_message="Error message",
    )

    exporter = OTLPExporter()
    otlp_span = exporter._span_to_otlp(span)

    assert otlp_span["status"]["code"] == 2  # ERROR
    assert otlp_span["status"]["message"] == "Error message"


def test_otlp_span_kind_mapping():
    """Test span type to OTLP kind mapping."""
    exporter = OTLPExporter()

    assert exporter._span_type_to_kind("agent") == 1  # INTERNAL
    assert exporter._span_type_to_kind("llm") == 3  # CLIENT
    assert exporter._span_type_to_kind("tool") == 1  # INTERNAL
    assert exporter._span_type_to_kind("retrieval") == 3  # CLIENT
    assert exporter._span_type_to_kind("embedding") == 3  # CLIENT
    assert exporter._span_type_to_kind("custom") == 1  # INTERNAL
    assert exporter._span_type_to_kind("unknown") == 0  # UNSPECIFIED


def test_otlp_attribute_types():
    """Test OTLP attribute conversion for different types."""
    exporter = OTLPExporter()

    # String
    attr = exporter._attribute_to_otlp("key", "value")
    assert attr["value"]["stringValue"] == "value"

    # Integer
    attr = exporter._attribute_to_otlp("key", 42)
    assert attr["value"]["intValue"] == "42"

    # Float
    attr = exporter._attribute_to_otlp("key", 3.14)
    assert attr["value"]["doubleValue"] == 3.14

    # Boolean
    attr = exporter._attribute_to_otlp("key", True)
    assert attr["value"]["boolValue"] is True

    # List
    attr = exporter._attribute_to_otlp("key", ["a", "b", "c"])
    assert "arrayValue" in attr["value"]
    assert len(attr["value"]["arrayValue"]["values"]) == 3


def test_otlp_span_with_events():
    """Test OTLP span includes events."""
    span = Span(
        span_id="span-123",
        trace_id="trace-456",
        name="test",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    span.add_event("event1", {"key": "value"})
    span.add_event("event2")

    exporter = OTLPExporter()
    otlp_span = exporter._span_to_otlp(span)

    assert len(otlp_span["events"]) == 2
    assert otlp_span["events"][0]["name"] == "event1"
    assert otlp_span["events"][1]["name"] == "event2"


def test_otlp_multiple_traces():
    """Test OTLP export with multiple traces."""
    span1 = Span(
        span_id="span-1",
        trace_id="trace-1",
        name="op1",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        attributes={"service.name": "service1"},
    )
    span2 = Span(
        span_id="span-2",
        trace_id="trace-2",
        name="op2",
        span_type=SpanType.CUSTOM,
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        attributes={"service.name": "service2"},
    )

    exporter = OTLPExporter()
    otlp_data = exporter._spans_to_otlp([span1, span2])

    # Should have 2 resource spans (one per trace)
    assert len(otlp_data["resourceSpans"]) == 2


def test_otlp_exporter_shutdown():
    """Test OTLP exporter shutdown."""
    exporter = OTLPExporter()
    exporter.shutdown()  # Should not raise
