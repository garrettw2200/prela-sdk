"""Span implementation for distributed tracing of AI agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SpanType(Enum):
    """Type of span in the trace."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class SpanStatus(Enum):
    """Status of a span."""

    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class SpanEvent:
    """An event that occurred during a span's execution."""

    __slots__ = ("timestamp", "name", "attributes")

    timestamp: datetime
    name: str
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanEvent:
        """Create event from dictionary representation."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            name=data["name"],
            attributes=data["attributes"],
        )


class Span:
    """A span represents a unit of work in a distributed trace.

    Spans are immutable after being ended. Any attempt to modify an ended span
    will raise a RuntimeError.
    """

    __slots__ = (
        "span_id",
        "trace_id",
        "parent_span_id",
        "name",
        "span_type",
        "started_at",
        "ended_at",
        "status",
        "status_message",
        "attributes",
        "events",
        "_ended",
        "replay_snapshot",
        "_tracer",
        "_context_token",
        "_sampled",
    )

    def __init__(
        self,
        span_id: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        name: str = "",
        span_type: SpanType = SpanType.CUSTOM,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        status: SpanStatus = SpanStatus.PENDING,
        status_message: str | None = None,
        attributes: dict[str, Any] | None = None,
        events: list[SpanEvent] | None = None,
        _ended: bool = False,
        replay_snapshot: Any = None,
    ) -> None:
        """Initialize a new span.

        Args:
            span_id: Unique identifier for this span (generates UUID if not provided)
            trace_id: Trace ID this span belongs to (generates UUID if not provided)
            parent_span_id: Parent span ID if this is a child span
            name: Human-readable name for this span
            span_type: Type of operation this span represents
            started_at: When the span started (uses current time if not provided)
            ended_at: When the span ended (None if still running)
            status: Current status of the span
            status_message: Optional message describing the status
            attributes: Key-value pairs of metadata
            events: List of events that occurred during span execution
            _ended: Internal flag for immutability (do not set manually)
            replay_snapshot: Optional replay data for deterministic re-execution
        """
        object.__setattr__(self, "span_id", span_id or str(uuid.uuid4()))
        object.__setattr__(self, "trace_id", trace_id or str(uuid.uuid4()))
        object.__setattr__(self, "parent_span_id", parent_span_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "span_type", span_type)
        object.__setattr__(self, "started_at", started_at or datetime.now(timezone.utc))
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "status_message", status_message)
        object.__setattr__(self, "attributes", attributes or {})
        object.__setattr__(self, "events", events or [])
        object.__setattr__(self, "_ended", _ended)
        object.__setattr__(self, "replay_snapshot", replay_snapshot)

    def _check_ended(self) -> None:
        """Raise error if span has been ended."""
        if self._ended:  # type: ignore[has-type]
            raise RuntimeError(
                f"Cannot modify span '{self.name}' (ID: {self.span_id}) after it has ended"
            )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        Args:
            key: Attribute key
            value: Attribute value

        Raises:
            RuntimeError: If the span has already ended
        """
        self._check_ended()
        self.attributes[key] = value  # type: ignore[has-type]

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span.

        Args:
            name: Event name
            attributes: Optional event attributes

        Raises:
            RuntimeError: If the span has already ended
        """
        self._check_ended()
        event = SpanEvent(
            timestamp=datetime.now(timezone.utc),
            name=name,
            attributes=attributes or {},
        )
        self.events.append(event)  # type: ignore[has-type]

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        """Set the status of the span.

        Args:
            status: Span status
            message: Optional status message

        Raises:
            RuntimeError: If the span has already ended
        """
        self._check_ended()
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "status_message", message)

    def end(self, end_time: datetime | None = None) -> None:
        """End the span.

        Args:
            end_time: When the span ended (uses current time if not provided)

        Raises:
            RuntimeError: If the span has already ended
        """
        self._check_ended()
        object.__setattr__(self, "ended_at", end_time or datetime.now(timezone.utc))
        object.__setattr__(self, "_ended", True)

        # Set status to SUCCESS if still PENDING
        if self.status == SpanStatus.PENDING:  # type: ignore[has-type]
            object.__setattr__(self, "status", SpanStatus.SUCCESS)

        # Handle cleanup for spans created via start_span()
        if hasattr(self, "_tracer"):
            from prela.core.context import get_current_context, reset_context

            ctx = get_current_context()
            if ctx:
                # Pop span from context
                ctx.pop_span()

                # Add to completed spans collection
                ctx.add_completed_span(self)

                # Export if sampled and this is a root span
                tracer = getattr(self, "_tracer")
                sampled = getattr(self, "_sampled", False)
                if sampled and self.parent_span_id is None and tracer.exporter:  # type: ignore[has-type]
                    # Export ALL spans in the trace, not just the root
                    tracer.exporter.export(ctx.all_spans)

                # Reset context if we created it
                token = getattr(self, "_context_token", None)
                if token is not None:
                    reset_context(token)

    @property
    def duration_ms(self) -> float | None:
        """Get the duration of the span in milliseconds.

        Returns:
            Duration in milliseconds, or None if span hasn't ended
        """
        if self.ended_at is None:  # type: ignore[has-type]
            return None
        delta = self.ended_at - self.started_at  # type: ignore[has-type]
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary representation.

        Returns:
            Dictionary containing all span data
        """
        result = {
            "span_id": self.span_id,  # type: ignore[has-type]
            "trace_id": self.trace_id,  # type: ignore[has-type]
            "parent_span_id": self.parent_span_id,  # type: ignore[has-type]
            "name": self.name,  # type: ignore[has-type]
            "span_type": self.span_type.value,  # type: ignore[has-type]
            "started_at": self.started_at.isoformat(),  # type: ignore[has-type]
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,  # type: ignore[has-type]
            "status": self.status.value,  # type: ignore[has-type]
            "status_message": self.status_message,  # type: ignore[has-type]
            "attributes": self.attributes,  # type: ignore[has-type]
            "events": [event.to_dict() for event in self.events],  # type: ignore[has-type]
            "duration_ms": self.duration_ms,
        }

        # Include replay data if present
        if self.replay_snapshot is not None:  # type: ignore[has-type]
            result["replay_snapshot"] = self.replay_snapshot.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        """Create span from dictionary representation.

        Args:
            data: Dictionary containing span data

        Returns:
            Reconstructed Span instance
        """
        # Deserialize replay snapshot if present
        replay_snapshot = None
        if "replay_snapshot" in data:
            from prela.core.replay import ReplaySnapshot

            replay_snapshot = ReplaySnapshot.from_dict(data["replay_snapshot"])

        return cls(
            span_id=data["span_id"],
            trace_id=data["trace_id"],
            parent_span_id=data.get("parent_span_id"),
            name=data["name"],
            span_type=SpanType(data["span_type"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=(datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None),
            status=SpanStatus(data.get("status", "pending")),
            status_message=data.get("status_message"),
            attributes=data.get("attributes", {}),
            events=[SpanEvent.from_dict(e) for e in data.get("events", [])],
            _ended=data.get("ended_at") is not None,
            replay_snapshot=replay_snapshot,
        )
