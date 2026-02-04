"""Tests for HTTP exporter."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
import requests

from prela.core.span import Span, SpanStatus, SpanType
from prela.exporters.base import ExportResult
from prela.exporters.http import HTTPExporter


@pytest.fixture
def mock_session():
    """Mock requests.Session."""
    with patch("requests.Session") as mock:
        session_instance = Mock()
        mock.return_value = session_instance

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "accepted"}'
        session_instance.post.return_value = mock_response
        session_instance.headers = {}

        yield session_instance


@pytest.fixture
def sample_spans():
    """Create sample spans for testing."""
    trace_id = "trace-123"
    now = datetime.now(timezone.utc)

    spans = [
        Span(
            span_id="span-1",
            trace_id=trace_id,
            name="root_operation",
            span_type=SpanType.AGENT,
            started_at=now,
        ),
        Span(
            span_id="span-2",
            trace_id=trace_id,
            parent_span_id="span-1",
            name="llm_call",
            span_type=SpanType.LLM,
            started_at=now + timedelta(milliseconds=10),
        ),
    ]

    # Set attributes BEFORE ending spans
    for span in spans:
        span.set_attribute("service.name", "test-service")
        span.end()

    return spans


class TestHTTPExporterInit:
    """Test HTTPExporter initialization."""

    def test_init_minimal(self, mock_session):
        """Test initialization with minimal parameters."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        assert exporter.endpoint == "https://api.example.com/v1/traces"
        assert exporter.api_key is None
        assert exporter.bearer_token is None
        assert exporter.compress is False
        assert exporter.max_retries == 3
        assert exporter.timeout_ms == 30000.0

    def test_init_with_api_key(self, mock_session):
        """Test initialization with API key."""
        exporter = HTTPExporter(
            endpoint="https://api.example.com/v1/traces", api_key="test-key"
        )

        assert exporter.api_key == "test-key"

    def test_init_with_bearer_token(self, mock_session):
        """Test initialization with Bearer token."""
        exporter = HTTPExporter(
            endpoint="https://api.example.com/v1/traces", bearer_token="test-token"
        )

        assert exporter.bearer_token == "test-token"

    def test_init_with_compression(self, mock_session):
        """Test initialization with compression enabled."""
        exporter = HTTPExporter(
            endpoint="https://api.example.com/v1/traces", compress=True
        )

        assert exporter.compress is True

    def test_init_strips_trailing_slash(self, mock_session):
        """Test that trailing slash is removed from endpoint."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces/")

        assert exporter.endpoint == "https://api.example.com/v1/traces"

    def test_init_empty_endpoint_raises(self, mock_session):
        """Test that empty endpoint raises ValueError."""
        with pytest.raises(ValueError, match="endpoint cannot be empty"):
            HTTPExporter(endpoint="")

    def test_init_both_auth_raises(self, mock_session):
        """Test that providing both api_key and bearer_token raises."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            HTTPExporter(
                endpoint="https://api.example.com/v1/traces",
                api_key="key",
                bearer_token="token",
            )

    def test_init_custom_headers(self, mock_session):
        """Test initialization with custom headers."""
        exporter = HTTPExporter(
            endpoint="https://api.example.com/v1/traces",
            headers={"X-Custom": "value"},
        )

        assert exporter.headers == {"X-Custom": "value"}

    def test_init_custom_retry_config(self, mock_session):
        """Test initialization with custom retry configuration."""
        exporter = HTTPExporter(
            endpoint="https://api.example.com/v1/traces",
            max_retries=5,
            initial_backoff_ms=200.0,
            max_backoff_ms=5000.0,
            timeout_ms=10000.0,
        )

        assert exporter.max_retries == 5
        assert exporter.initial_backoff_ms == 200.0
        assert exporter.max_backoff_ms == 5000.0
        assert exporter.timeout_ms == 10000.0


class TestHTTPExporterExport:
    """Test HTTPExporter export functionality."""

    def test_export_empty_spans(self, mock_session):
        """Test exporting empty span list."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        result = exporter._do_export([])

        assert result == ExportResult.SUCCESS
        # No HTTP request should be made
        mock_session.post.assert_not_called()

    def test_export_single_trace(self, mock_session, sample_spans):
        """Test exporting a single trace."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.SUCCESS
        mock_session.post.assert_called_once()

        # Verify request
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://api.example.com/v1/traces"
        assert call_args[1]["timeout"] == 30.0  # 30000ms / 1000

        # Verify payload structure
        payload = json.loads(call_args[1]["data"])
        assert payload["trace_id"] == "trace-123"
        assert payload["service_name"] == "test-service"
        assert len(payload["spans"]) == 2
        assert payload["status"] == "SUCCESS"

    def test_export_with_compression(self, mock_session, sample_spans):
        """Test exporting with gzip compression."""
        exporter = HTTPExporter(
            endpoint="https://api.example.com/v1/traces", compress=True
        )

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.SUCCESS

        # Verify compressed payload
        call_args = mock_session.post.call_args
        compressed_data = call_args[1]["data"]
        assert isinstance(compressed_data, bytes)

        # Decompress and verify
        decompressed = gzip.decompress(compressed_data)
        payload = json.loads(decompressed)
        assert payload["trace_id"] == "trace-123"

    def test_export_multiple_traces(self, mock_session):
        """Test exporting multiple traces in one batch."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        # Create spans from two different traces
        now = datetime.now(timezone.utc)
        spans = [
            Span(
                span_id="span-1",
                trace_id="trace-1",
                name="op1",
                span_type=SpanType.AGENT,
                started_at=now,
            ),
            Span(
                span_id="span-2",
                trace_id="trace-2",
                name="op2",
                span_type=SpanType.AGENT,
                started_at=now,
            ),
        ]
        for span in spans:
            span.set_attribute("service.name", "test")
            span.end()

        result = exporter._do_export(spans)

        assert result == ExportResult.SUCCESS
        # Should make 2 HTTP requests (one per trace)
        assert mock_session.post.call_count == 2

    def test_export_with_error_span(self, mock_session):
        """Test exporting trace with error span."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        # Create span with error
        now = datetime.now(timezone.utc)
        span = Span(
            span_id="span-1",
            trace_id="trace-1",
            name="failed_op",
            span_type=SpanType.AGENT,
            started_at=now,
        )
        span.set_attribute("service.name", "test")
        span.set_status(SpanStatus.ERROR, "Something went wrong")
        span.end()

        result = exporter._do_export([span])

        assert result == ExportResult.SUCCESS

        # Verify status is ERROR
        call_args = mock_session.post.call_args
        payload = json.loads(call_args[1]["data"])
        assert payload["status"] == "ERROR"

    def test_export_202_accepted(self, mock_session, sample_spans):
        """Test that 202 Accepted is treated as success."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_response = mock_session.post.return_value
        mock_response.status_code = 202

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.SUCCESS

    def test_export_400_client_error(self, mock_session, sample_spans):
        """Test that 4xx errors return FAILURE (don't retry)."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_response = mock_session.post.return_value
        mock_response.status_code = 400
        mock_response.text = "Bad request"

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.FAILURE

    def test_export_401_unauthorized(self, mock_session, sample_spans):
        """Test that 401 Unauthorized returns FAILURE."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_response = mock_session.post.return_value
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.FAILURE

    def test_export_500_server_error(self, mock_session, sample_spans):
        """Test that 5xx errors return RETRY."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_response = mock_session.post.return_value
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.RETRY

    def test_export_503_service_unavailable(self, mock_session, sample_spans):
        """Test that 503 Service Unavailable returns RETRY."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_response = mock_session.post.return_value
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.RETRY

    def test_export_timeout(self, mock_session, sample_spans):
        """Test that timeout returns RETRY."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_session.post.side_effect = requests.exceptions.Timeout()

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.RETRY

    def test_export_connection_error(self, mock_session, sample_spans):
        """Test that connection error returns RETRY."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_session.post.side_effect = requests.exceptions.ConnectionError()

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.RETRY

    def test_export_unexpected_exception(self, mock_session, sample_spans):
        """Test that unexpected exceptions return RETRY."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        mock_session.post.side_effect = RuntimeError("Unexpected error")

        result = exporter._do_export(sample_spans)

        assert result == ExportResult.RETRY

    def test_export_serialization_error(self, mock_session):
        """Test that JSON serialization errors return FAILURE."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        # Create span with non-serializable attribute
        now = datetime.now(timezone.utc)
        span = Span(
            span_id="span-1",
            trace_id="trace-1",
            name="op",
            span_type=SpanType.AGENT,
            started_at=now,
        )
        span.set_attribute("service.name", "test")
        span.set_attribute("bad_value", object())  # Non-serializable
        span.end()

        result = exporter._do_export([span])

        # Should return FAILURE because serialization is permanent failure
        assert result == ExportResult.FAILURE


class TestHTTPExporterIntegration:
    """Test HTTPExporter with full export() method (retry logic)."""

    def test_export_with_retry_success(self, mock_session, sample_spans):
        """Test successful export with retry wrapper."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        exporter.export(sample_spans)

        # Should not raise exception
        assert mock_session.post.call_count == 1

    def test_export_after_shutdown_raises(self, mock_session, sample_spans):
        """Test that export after shutdown raises error."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")
        exporter.shutdown()

        with pytest.raises(RuntimeError, match="exporter is shutdown"):
            exporter.export(sample_spans)


class TestHTTPExporterShutdown:
    """Test HTTPExporter shutdown."""

    def test_shutdown(self, mock_session):
        """Test shutdown closes session."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        exporter.shutdown()

        mock_session.close.assert_called_once()
        assert exporter._shutdown is True

    def test_shutdown_idempotent(self, mock_session):
        """Test that shutdown can be called multiple times."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        exporter.shutdown()
        exporter.shutdown()

        # Should not raise exception
        assert exporter._shutdown is True


class TestSpanSerialization:
    """Test span serialization for HTTP export."""

    def test_span_to_dict_basic(self, mock_session):
        """Test basic span serialization."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        now = datetime.now(timezone.utc)
        span = Span(
            span_id="span-1",
            trace_id="trace-1",
            name="operation",
            span_type=SpanType.LLM,
            started_at=now,
        )
        span.set_attribute("key", "value")
        span.end()

        span_dict = exporter._span_to_dict(span)

        assert span_dict["span_id"] == "span-1"
        assert span_dict["trace_id"] == "trace-1"
        assert span_dict["name"] == "operation"
        assert span_dict["span_type"] == "llm"
        assert span_dict["status"] == "success"
        assert span_dict["attributes"] == {"key": "value"}
        assert span_dict["started_at"].endswith("Z")
        assert span_dict["ended_at"].endswith("Z")
        assert span_dict["duration_ms"] > 0

    def test_span_to_dict_with_parent(self, mock_session):
        """Test span serialization with parent."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        now = datetime.now(timezone.utc)
        span = Span(
            span_id="span-2",
            trace_id="trace-1",
            parent_span_id="span-1",
            name="child_op",
            span_type=SpanType.TOOL,
            started_at=now,
        )
        span.end()

        span_dict = exporter._span_to_dict(span)

        assert span_dict["parent_span_id"] == "span-1"

    def test_span_to_dict_with_events(self, mock_session):
        """Test span serialization with events."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        now = datetime.now(timezone.utc)
        span = Span(
            span_id="span-1",
            trace_id="trace-1",
            name="operation",
            span_type=SpanType.AGENT,
            started_at=now,
        )
        span.add_event("event1", {"key": "value"})
        span.end()

        span_dict = exporter._span_to_dict(span)

        assert len(span_dict["events"]) == 1
        assert span_dict["events"][0]["name"] == "event1"
        assert span_dict["events"][0]["attributes"] == {"key": "value"}
        assert span_dict["events"][0]["timestamp"].endswith("Z")

    def test_span_to_dict_without_end(self, mock_session):
        """Test span serialization without end time."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        now = datetime.now(timezone.utc)
        span = Span(
            span_id="span-1",
            trace_id="trace-1",
            name="operation",
            span_type=SpanType.AGENT,
            started_at=now,
        )

        span_dict = exporter._span_to_dict(span)

        assert span_dict["ended_at"] is None
        assert span_dict["duration_ms"] == 0.0


class TestStatusAggregation:
    """Test trace status aggregation."""

    def test_aggregate_status_all_success(self, mock_session):
        """Test status aggregation when all spans succeed."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        now = datetime.now(timezone.utc)
        spans = [
            Span(
                span_id=f"span-{i}",
                trace_id="trace-1",
                name=f"op-{i}",
                span_type=SpanType.AGENT,
                started_at=now,
            )
            for i in range(3)
        ]
        for span in spans:
            span.end()

        status = exporter._aggregate_status(spans)

        assert status == "SUCCESS"

    def test_aggregate_status_with_error(self, mock_session):
        """Test status aggregation when one span has error."""
        exporter = HTTPExporter(endpoint="https://api.example.com/v1/traces")

        now = datetime.now(timezone.utc)
        spans = [
            Span(
                span_id=f"span-{i}",
                trace_id="trace-1",
                name=f"op-{i}",
                span_type=SpanType.AGENT,
                started_at=now,
            )
            for i in range(3)
        ]

        # Set one span to error
        spans[1].set_status(SpanStatus.ERROR)

        for span in spans:
            span.end()

        status = exporter._aggregate_status(spans)

        assert status == "ERROR"
