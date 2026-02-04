"""
OTLP exporter for sending traces to OpenTelemetry Protocol endpoints.

This exporter sends spans to any OTLP-compatible backend (e.g., Jaeger,
Tempo, Honeycomb, New Relic, etc.) using the OpenTelemetry Protocol.

The exporter uses HTTP/protobuf by default but can be configured for gRPC.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

from prela.core.span import Span
from prela.exporters.base import BaseExporter, ExportResult

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class OTLPExporter(BaseExporter):
    """
    Exporter for sending traces to OTLP-compatible backends.

    Supports:
    - Jaeger (v1.35+)
    - Grafana Tempo
    - Honeycomb
    - New Relic
    - Any OpenTelemetry Collector
    - Any OTLP-compatible endpoint

    The exporter uses HTTP/JSON by default for maximum compatibility.
    Protobuf encoding is available but requires additional dependencies.

    Example:
        ```python
        from prela import init
        from prela.exporters.otlp import OTLPExporter

        # Simple usage with defaults (localhost collector)
        init(
            service_name="my-app",
            exporter=OTLPExporter()
        )

        # Jaeger
        init(
            service_name="my-app",
            exporter=OTLPExporter(endpoint="http://localhost:4318")
        )

        # Tempo
        init(
            service_name="my-app",
            exporter=OTLPExporter(
                endpoint="http://tempo:4318",
                headers={"X-Scope-OrgID": "tenant1"}
            )
        )

        # Honeycomb
        init(
            service_name="my-app",
            exporter=OTLPExporter(
                endpoint="https://api.honeycomb.io:443",
                headers={
                    "x-honeycomb-team": "YOUR_API_KEY",
                    "x-honeycomb-dataset": "my-dataset"
                }
            )
        )
        ```
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        headers: dict[str, str] | None = None,
        timeout: int = 10,
        insecure: bool = False,
        compression: str | None = None,
    ):
        """
        Initialize OTLP exporter.

        Args:
            endpoint: OTLP endpoint URL (default: http://localhost:4318/v1/traces)
            headers: Additional HTTP headers (e.g., authentication)
            timeout: Request timeout in seconds (default: 10)
            insecure: Allow insecure HTTPS connections (default: False)
            compression: Compression algorithm ("gzip" or None)

        Raises:
            ImportError: If requests library is not installed
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library is required for OTLP exporter. "
                "Install with: pip install 'prela[otlp]' or 'pip install requests'"
            )

        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.insecure = insecure
        self.compression = compression

        # Set default headers
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

        # Validate endpoint
        parsed = urlparse(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"Invalid endpoint URL: {endpoint}. "
                f"Must include scheme and host (e.g., http://localhost:4318)"
            )

        logger.debug(f"OTLPExporter initialized: {endpoint}")

    def export(self, spans: list[Span]) -> ExportResult:
        """
        Export spans to OTLP endpoint.

        Args:
            spans: List of spans to export

        Returns:
            ExportResult indicating success, failure, or retry
        """
        if not spans:
            return ExportResult.SUCCESS

        try:
            # Convert spans to OTLP format
            otlp_data = self._spans_to_otlp(spans)

            # Send to endpoint
            response = requests.post(
                self.endpoint,
                json=otlp_data,
                headers=self.headers,
                timeout=self.timeout,
                verify=not self.insecure,
            )

            # Check response
            if response.status_code == 200:
                logger.debug(f"Exported {len(spans)} spans to OTLP endpoint")
                return ExportResult.SUCCESS
            elif response.status_code in (429, 503):
                # Rate limit or service unavailable - retry
                logger.warning(
                    f"OTLP endpoint returned {response.status_code}, will retry"
                )
                return ExportResult.RETRY
            else:
                # Other errors - failure
                logger.error(
                    f"OTLP export failed: {response.status_code} {response.text}"
                )
                return ExportResult.FAILURE

        except requests.exceptions.Timeout:
            logger.warning(f"OTLP export timeout after {self.timeout}s, will retry")
            return ExportResult.RETRY
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"OTLP connection error: {e}, will retry")
            return ExportResult.RETRY
        except Exception as e:
            logger.error(f"OTLP export error: {e}", exc_info=True)
            return ExportResult.FAILURE

    def _spans_to_otlp(self, spans: list[Span]) -> dict[str, Any]:
        """
        Convert Prela spans to OTLP JSON format.

        OTLP format spec:
        https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto

        Args:
            spans: List of Prela spans

        Returns:
            OTLP-formatted dictionary
        """
        # Group spans by trace_id
        traces_by_id: dict[str, list[Span]] = {}
        for span in spans:
            if span.trace_id not in traces_by_id:
                traces_by_id[span.trace_id] = []
            traces_by_id[span.trace_id].append(span)

        # Build OTLP structure
        resource_spans = []

        for trace_id, trace_spans in traces_by_id.items():
            # Extract service name from first span
            service_name = "unknown"
            for span in trace_spans:
                if "service.name" in span.attributes:
                    service_name = span.attributes["service.name"]
                    break

            # Build scope spans (one scope for all spans)
            otlp_spans = []
            for span in trace_spans:
                otlp_span = self._span_to_otlp(span)
                otlp_spans.append(otlp_span)

            resource_span = {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}}
                    ]
                },
                "scopeSpans": [{"scope": {"name": "prela"}, "spans": otlp_spans}],
            }
            resource_spans.append(resource_span)

        return {"resourceSpans": resource_spans}

    def _span_to_otlp(self, span: Span) -> dict[str, Any]:
        """
        Convert a single Prela span to OTLP format.

        Args:
            span: Prela span

        Returns:
            OTLP span dictionary
        """
        # Convert trace_id and span_id to hex (OTLP uses hex strings)
        trace_id_hex = span.trace_id.replace("-", "")[:32].ljust(32, "0")
        span_id_hex = span.span_id.replace("-", "")[:16].ljust(16, "0")
        parent_span_id_hex = ""
        if span.parent_span_id:
            parent_span_id_hex = span.parent_span_id.replace("-", "")[:16].ljust(
                16, "0"
            )

        # Convert timestamps to nanoseconds
        start_time_ns = int(span.started_at.timestamp() * 1_000_000_000)
        end_time_ns = (
            int(span.ended_at.timestamp() * 1_000_000_000) if span.ended_at else start_time_ns
        )

        # Convert span type to OTLP span kind
        span_kind = self._span_type_to_kind(span.span_type.value)

        # Convert attributes
        attributes = []
        for key, value in span.attributes.items():
            if key == "service.name":
                continue  # Already in resource attributes
            attr = self._attribute_to_otlp(key, value)
            if attr:
                attributes.append(attr)

        # Convert events
        events = []
        for event in span.events:
            event_time_ns = int(event.timestamp.timestamp() * 1_000_000_000)
            event_attrs = []
            for key, value in event.attributes.items():
                attr = self._attribute_to_otlp(key, value)
                if attr:
                    event_attrs.append(attr)

            events.append(
                {
                    "timeUnixNano": str(event_time_ns),
                    "name": event.name,
                    "attributes": event_attrs,
                }
            )

        # Build status
        status = {"code": 0}  # UNSET
        if span.status:
            if span.status.value == "success":
                status["code"] = 1  # OK
            elif span.status.value == "error":
                status["code"] = 2  # ERROR
                if span.status_message:
                    status["message"] = span.status_message

        otlp_span = {
            "traceId": trace_id_hex,
            "spanId": span_id_hex,
            "name": span.name,
            "kind": span_kind,
            "startTimeUnixNano": str(start_time_ns),
            "endTimeUnixNano": str(end_time_ns),
            "attributes": attributes,
            "events": events,
            "status": status,
        }

        if parent_span_id_hex:
            otlp_span["parentSpanId"] = parent_span_id_hex

        return otlp_span

    def _span_type_to_kind(self, span_type: str) -> int:
        """
        Convert Prela span type to OTLP span kind.

        OTLP span kinds:
        - 0: UNSPECIFIED
        - 1: INTERNAL
        - 2: SERVER
        - 3: CLIENT
        - 4: PRODUCER
        - 5: CONSUMER

        Args:
            span_type: Prela span type

        Returns:
            OTLP span kind integer
        """
        # Map Prela types to OTLP kinds
        type_map = {
            "agent": 1,  # INTERNAL
            "llm": 3,  # CLIENT (calling external LLM service)
            "tool": 1,  # INTERNAL
            "retrieval": 3,  # CLIENT (calling external retrieval service)
            "embedding": 3,  # CLIENT (calling external embedding service)
            "custom": 1,  # INTERNAL
        }
        return type_map.get(span_type.lower(), 0)  # Default: UNSPECIFIED

    def _attribute_to_otlp(self, key: str, value: Any) -> dict[str, Any] | None:
        """
        Convert a Prela attribute to OTLP format.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            OTLP attribute dictionary, or None if value type unsupported
        """
        # Determine value type and convert
        if isinstance(value, bool):
            return {"key": key, "value": {"boolValue": value}}
        elif isinstance(value, int):
            return {"key": key, "value": {"intValue": str(value)}}
        elif isinstance(value, float):
            return {"key": key, "value": {"doubleValue": value}}
        elif isinstance(value, str):
            return {"key": key, "value": {"stringValue": value}}
        elif isinstance(value, (list, tuple)):
            # OTLP supports array values
            array_values = []
            for item in value:
                if isinstance(item, str):
                    array_values.append({"stringValue": item})
                elif isinstance(item, int):
                    array_values.append({"intValue": str(item)})
                elif isinstance(item, float):
                    array_values.append({"doubleValue": item})
                elif isinstance(item, bool):
                    array_values.append({"boolValue": item})
            if array_values:
                return {"key": key, "value": {"arrayValue": {"values": array_values}}}
        else:
            # Convert other types to string
            return {"key": key, "value": {"stringValue": str(value)}}

        return None

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        logger.debug("OTLPExporter shutdown")
