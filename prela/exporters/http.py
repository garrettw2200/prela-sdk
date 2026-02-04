"""HTTP exporter for sending spans to remote observability backends.

This module provides an HTTP-based exporter that sends spans to remote
Prela backends (e.g., Ingest Gateway on Railway) using JSON over HTTP.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any, Optional

from prela.core.span import Span
from prela.exporters.base import BatchExporter, ExportResult
from prela.license import set_tier

logger = logging.getLogger(__name__)


class HTTPExporter(BatchExporter):
    """
    Export spans to a remote HTTP endpoint using JSON.

    Features:
    - Batching with configurable size limits
    - Retry with exponential backoff
    - Optional gzip compression
    - Authentication via API key or Bearer token
    - Timeout handling

    The exporter sends spans to the configured endpoint as JSON with the format:
    ```json
    {
        "trace_id": "...",
        "service_name": "...",
        "started_at": "2025-01-27T10:00:00.000000Z",
        "completed_at": "2025-01-27T10:00:01.000000Z",
        "duration_ms": 1000.0,
        "status": "SUCCESS",
        "spans": [...]
    }
    ```

    Example:
        ```python
        from prela import init

        # Simple usage with Railway deployment
        init(
            service_name="my-agent",
            exporter="http",
            http_endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces"
        )

        # Advanced usage with authentication and compression
        from prela.exporters.http import HTTPExporter
        from prela import Tracer

        exporter = HTTPExporter(
            endpoint="https://api.prela.io/v1/traces",
            api_key="your-api-key",
            compress=True,
            timeout_ms=10000
        )

        tracer = Tracer(service_name="my-agent", exporter=exporter)
        ```
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        compress: bool = False,
        headers: Optional[dict[str, str]] = None,
        max_retries: int = 3,
        initial_backoff_ms: float = 100.0,
        max_backoff_ms: float = 10000.0,
        timeout_ms: float = 30000.0,
    ) -> None:
        """
        Initialize HTTP exporter.

        Args:
            endpoint: The HTTP endpoint URL (e.g., "https://api.prela.io/v1/traces")
            api_key: Optional API key for authentication (sent as X-API-Key header)
            bearer_token: Optional Bearer token for authentication
            compress: Enable gzip compression for request body
            headers: Additional HTTP headers to include
            max_retries: Maximum number of retry attempts (default: 3)
            initial_backoff_ms: Initial backoff delay in milliseconds (default: 100)
            max_backoff_ms: Maximum backoff delay in milliseconds (default: 10000)
            timeout_ms: Timeout for HTTP requests in milliseconds (default: 30000)

        Raises:
            ValueError: If endpoint is empty or both api_key and bearer_token are provided
        """
        if not endpoint:
            raise ValueError("endpoint cannot be empty")

        if api_key and bearer_token:
            raise ValueError("Cannot specify both api_key and bearer_token")

        super().__init__(
            max_retries=max_retries,
            initial_backoff_ms=initial_backoff_ms,
            max_backoff_ms=max_backoff_ms,
            timeout_ms=timeout_ms,
        )

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.compress = compress
        self.headers = headers or {}

        # Import requests here to make it optional dependency
        try:
            import requests

            self._requests = requests
            self._session = requests.Session()

            # Set default headers
            self._session.headers.update(
                {
                    "Content-Type": "application/json",
                    "User-Agent": "prela-sdk/0.1.0",
                }
            )

            # Add authentication headers
            if self.api_key:
                self._session.headers["X-API-Key"] = self.api_key
            elif self.bearer_token:
                self._session.headers["Authorization"] = f"Bearer {self.bearer_token}"

            # Add custom headers
            self._session.headers.update(self.headers)

            # Add compression header if enabled
            if self.compress:
                self._session.headers["Content-Encoding"] = "gzip"

        except ImportError:
            raise ImportError(
                "requests library is required for HTTPExporter. "
                "Install it with: pip install requests"
            )

        logger.debug(
            "Initialized HTTPExporter: endpoint=%s, compress=%s",
            self.endpoint,
            self.compress,
        )

    def _do_export(self, spans: list[Span]) -> ExportResult:
        """
        Perform the actual HTTP export operation.

        Sends spans to the configured endpoint as JSON. Groups spans by trace_id
        and sends each trace as a separate request.

        Args:
            spans: List of spans to export

        Returns:
            ExportResult.SUCCESS if export succeeded
            ExportResult.RETRY if export should be retried (5xx errors, network errors)
            ExportResult.FAILURE if export failed permanently (4xx errors)
        """
        if not spans:
            return ExportResult.SUCCESS

        # Group spans by trace_id
        traces: dict[str, list[Span]] = {}
        for span in spans:
            if span.trace_id not in traces:
                traces[span.trace_id] = []
            traces[span.trace_id].append(span)

        # Send each trace
        for trace_id, trace_spans in traces.items():
            try:
                result = self._send_trace(trace_id, trace_spans)
                if result != ExportResult.SUCCESS:
                    return result
            except Exception as e:
                logger.error("Failed to export trace %s: %s", trace_id, str(e))
                # Network errors should be retried
                if self._is_retryable_error(e):
                    return ExportResult.RETRY
                return ExportResult.FAILURE

        return ExportResult.SUCCESS

    def _send_trace(self, trace_id: str, spans: list[Span]) -> ExportResult:
        """
        Send a single trace to the HTTP endpoint.

        Args:
            trace_id: The trace ID
            spans: List of spans in this trace

        Returns:
            ExportResult indicating success, retry, or failure
        """
        # Build trace payload
        # Sort spans by started_at to find root span
        sorted_spans = sorted(spans, key=lambda s: s.started_at)
        root_span = sorted_spans[0]

        # Get service name from root span or first span with the attribute
        service_name = None
        for span in sorted_spans:
            if "service.name" in span.attributes:
                service_name = span.attributes["service.name"]
                break

        if not service_name:
            service_name = "unknown"

        payload = {
            "trace_id": trace_id,
            "service_name": service_name,
            "started_at": root_span.started_at.isoformat() + "Z",
            "completed_at": sorted_spans[-1].ended_at.isoformat() + "Z"
            if sorted_spans[-1].ended_at
            else root_span.started_at.isoformat() + "Z",
            "duration_ms": sum(
                (s.ended_at - s.started_at).total_seconds() * 1000
                for s in spans
                if s.ended_at
            ),
            "status": self._aggregate_status(spans),
            "spans": [self._span_to_dict(span) for span in spans],
        }

        # Serialize to JSON
        try:
            json_data = json.dumps(payload)
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize trace %s: %s", trace_id, str(e))
            return ExportResult.FAILURE

        # Compress if enabled
        if self.compress:
            json_bytes = json_data.encode("utf-8")
            body = gzip.compress(json_bytes)
        else:
            body = json_data

        # Send HTTP request
        try:
            timeout_seconds = self.timeout_ms / 1000
            response = self._session.post(
                self.endpoint, data=body, timeout=timeout_seconds
            )

            # Check response status
            if response.status_code == 200 or response.status_code == 202:
                logger.debug(
                    "Successfully exported trace %s (%d spans)",
                    trace_id[:8],
                    len(spans),
                )

                # Extract tier from response if available
                # The ingest gateway can return tier info in custom headers
                if "X-Prela-Tier" in response.headers:
                    tier = response.headers["X-Prela-Tier"]
                    set_tier(tier)
                    logger.debug(f"Subscription tier detected: {tier}")

                return ExportResult.SUCCESS

            elif 400 <= response.status_code < 500:
                # Client errors (bad request, auth, etc.) - don't retry
                logger.error(
                    "Export failed with client error %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return ExportResult.FAILURE

            elif 500 <= response.status_code < 600:
                # Server errors - retry
                logger.warning(
                    "Export failed with server error %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return ExportResult.RETRY

            else:
                # Unexpected status code
                logger.error(
                    "Export failed with unexpected status %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return ExportResult.FAILURE

        except self._requests.exceptions.Timeout:
            logger.warning("Export request timed out after %.2fs", timeout_seconds)
            return ExportResult.RETRY

        except self._requests.exceptions.ConnectionError as e:
            logger.warning("Export connection error: %s", str(e))
            return ExportResult.RETRY

        except Exception as e:
            logger.error("Export request failed: %s", str(e))
            return ExportResult.RETRY

    def _span_to_dict(self, span: Span) -> dict[str, Any]:
        """
        Convert a span to a dictionary for JSON serialization.

        Args:
            span: The span to convert

        Returns:
            Dictionary representation of the span
        """
        return {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "span_type": span.span_type.value,
            "started_at": span.started_at.isoformat() + "Z",
            "ended_at": span.ended_at.isoformat() + "Z" if span.ended_at else None,
            "duration_ms": (span.ended_at - span.started_at).total_seconds() * 1000
            if span.ended_at
            else 0.0,
            "status": span.status.value,
            "status_message": span.status_message,
            "attributes": span.attributes,
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp.isoformat() + "Z",
                    "attributes": event.attributes,
                }
                for event in span.events
            ],
        }

    def _aggregate_status(self, spans: list[Span]) -> str:
        """
        Aggregate span statuses into a single trace status.

        Args:
            spans: List of spans in the trace

        Returns:
            "ERROR" if any span has ERROR status, otherwise "SUCCESS"
        """
        from prela.core.span import SpanStatus

        for span in spans:
            if span.status == SpanStatus.ERROR:
                return "ERROR"
        return "SUCCESS"

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error should be retried, False otherwise
        """
        # Network errors, timeouts, and server errors are retryable
        retryable_types = (
            self._requests.exceptions.Timeout,
            self._requests.exceptions.ConnectionError,
            self._requests.exceptions.HTTPError,
        )
        return isinstance(error, retryable_types)

    def shutdown(self) -> None:
        """
        Shutdown the exporter and close the HTTP session.
        """
        super().shutdown()
        if hasattr(self, "_session"):
            self._session.close()
            logger.debug("HTTP session closed")
