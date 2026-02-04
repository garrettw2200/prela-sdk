"""Tests for span implementation."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from prela.core.span import Span, SpanEvent, SpanStatus, SpanType


class TestSpanCreation:
    """Tests for creating spans."""

    def test_create_span_with_defaults(self) -> None:
        """Test creating a span with default values."""
        span = Span(name="test-span")

        assert span.name == "test-span"
        assert span.span_id is not None
        assert span.trace_id is not None
        assert span.parent_span_id is None
        assert span.span_type == SpanType.CUSTOM
        assert span.status == SpanStatus.PENDING
        assert span.status_message is None
        assert isinstance(span.started_at, datetime)
        assert span.ended_at is None
        assert span.attributes == {}
        assert span.events == []
        assert span._ended is False

    def test_create_span_with_custom_values(self) -> None:
        """Test creating a span with custom values."""
        span_id = "custom-span-id"
        trace_id = "custom-trace-id"
        parent_id = "parent-span-id"
        started_at = datetime.now(timezone.utc)
        attributes = {"key": "value"}

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_id,
            name="custom-span",
            span_type=SpanType.LLM,
            started_at=started_at,
            attributes=attributes,
        )

        assert span.span_id == span_id
        assert span.trace_id == trace_id
        assert span.parent_span_id == parent_id
        assert span.name == "custom-span"
        assert span.span_type == SpanType.LLM
        assert span.started_at == started_at
        assert span.attributes == attributes

    def test_span_types(self) -> None:
        """Test all span types can be created."""
        for span_type in SpanType:
            span = Span(name=f"{span_type.value}-span", span_type=span_type)
            assert span.span_type == span_type

    def test_span_generates_unique_ids(self) -> None:
        """Test that each span gets unique IDs."""
        span1 = Span(name="span1")
        span2 = Span(name="span2")

        assert span1.span_id != span2.span_id
        assert span1.trace_id != span2.trace_id


class TestSpanAttributes:
    """Tests for span attributes."""

    def test_set_attribute(self) -> None:
        """Test setting attributes on a span."""
        span = Span(name="test-span")

        span.set_attribute("model", "gpt-4")
        span.set_attribute("temperature", 0.7)
        span.set_attribute("max_tokens", 100)

        assert span.attributes["model"] == "gpt-4"
        assert span.attributes["temperature"] == 0.7
        assert span.attributes["max_tokens"] == 100

    def test_set_attribute_updates_existing(self) -> None:
        """Test that setting an attribute updates existing value."""
        span = Span(name="test-span")

        span.set_attribute("count", 1)
        assert span.attributes["count"] == 1

        span.set_attribute("count", 2)
        assert span.attributes["count"] == 2

    def test_set_attribute_after_end_raises_error(self) -> None:
        """Test that setting attributes after end raises error."""
        span = Span(name="test-span")
        span.end()

        with pytest.raises(RuntimeError, match="Cannot modify span.*after it has ended"):
            span.set_attribute("key", "value")


class TestSpanEvents:
    """Tests for span events."""

    def test_add_event(self) -> None:
        """Test adding an event to a span."""
        span = Span(name="test-span")

        span.add_event("api_call_started")

        assert len(span.events) == 1
        assert span.events[0].name == "api_call_started"
        assert isinstance(span.events[0].timestamp, datetime)
        assert span.events[0].attributes == {}

    def test_add_event_with_attributes(self) -> None:
        """Test adding an event with attributes."""
        span = Span(name="test-span")

        span.add_event("tool_called", {"tool_name": "calculator", "input": "2+2"})

        assert len(span.events) == 1
        assert span.events[0].name == "tool_called"
        assert span.events[0].attributes == {"tool_name": "calculator", "input": "2+2"}

    def test_add_multiple_events(self) -> None:
        """Test adding multiple events."""
        span = Span(name="test-span")

        span.add_event("event1")
        span.add_event("event2")
        span.add_event("event3")

        assert len(span.events) == 3
        assert span.events[0].name == "event1"
        assert span.events[1].name == "event2"
        assert span.events[2].name == "event3"

    def test_add_event_after_end_raises_error(self) -> None:
        """Test that adding events after end raises error."""
        span = Span(name="test-span")
        span.end()

        with pytest.raises(RuntimeError, match="Cannot modify span.*after it has ended"):
            span.add_event("event")


class TestSpanStatus:
    """Tests for span status."""

    def test_set_status_success(self) -> None:
        """Test setting status to success."""
        span = Span(name="test-span")

        span.set_status(SpanStatus.SUCCESS, "Operation completed")

        assert span.status == SpanStatus.SUCCESS
        assert span.status_message == "Operation completed"

    def test_set_status_error(self) -> None:
        """Test setting status to error."""
        span = Span(name="test-span")

        span.set_status(SpanStatus.ERROR, "API request failed")

        assert span.status == SpanStatus.ERROR
        assert span.status_message == "API request failed"

    def test_set_status_without_message(self) -> None:
        """Test setting status without a message."""
        span = Span(name="test-span")

        span.set_status(SpanStatus.SUCCESS)

        assert span.status == SpanStatus.SUCCESS
        assert span.status_message is None

    def test_set_status_after_end_raises_error(self) -> None:
        """Test that setting status after end raises error."""
        span = Span(name="test-span")
        span.end()

        with pytest.raises(RuntimeError, match="Cannot modify span.*after it has ended"):
            span.set_status(SpanStatus.ERROR)


class TestSpanEnd:
    """Tests for ending spans."""

    def test_end_span(self) -> None:
        """Test ending a span."""
        span = Span(name="test-span")
        assert span.ended_at is None
        assert span._ended is False

        span.end()

        assert span.ended_at is not None
        assert isinstance(span.ended_at, datetime)
        assert span._ended is True

    def test_end_span_with_custom_time(self) -> None:
        """Test ending a span with custom time."""
        span = Span(name="test-span")
        end_time = datetime.now(timezone.utc) + timedelta(seconds=5)

        span.end(end_time)

        assert span.ended_at == end_time

    def test_end_span_sets_success_if_pending(self) -> None:
        """Test that ending a pending span sets status to SUCCESS."""
        span = Span(name="test-span")
        assert span.status == SpanStatus.PENDING

        span.end()

        assert span.status == SpanStatus.SUCCESS

    def test_end_span_preserves_error_status(self) -> None:
        """Test that ending a span preserves ERROR status."""
        span = Span(name="test-span")
        span.set_status(SpanStatus.ERROR, "Failed")

        span.end()

        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Failed"

    def test_end_span_twice_raises_error(self) -> None:
        """Test that ending a span twice raises error."""
        span = Span(name="test-span")
        span.end()

        with pytest.raises(RuntimeError, match="Cannot modify span.*after it has ended"):
            span.end()


class TestSpanDuration:
    """Tests for span duration calculation."""

    def test_duration_ms_before_end(self) -> None:
        """Test duration is None before span is ended."""
        span = Span(name="test-span")
        assert span.duration_ms is None

    def test_duration_ms_after_end(self) -> None:
        """Test duration calculation after span is ended."""
        started_at = datetime.now(timezone.utc)
        ended_at = started_at + timedelta(milliseconds=150)

        span = Span(name="test-span", started_at=started_at)
        span.end(ended_at)

        assert span.duration_ms == pytest.approx(150.0, rel=0.01)

    def test_duration_ms_realistic(self) -> None:
        """Test duration with realistic timing."""
        span = Span(name="test-span")
        time.sleep(0.01)  # Sleep for ~10ms
        span.end()

        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms


class TestSpanImmutability:
    """Tests for span immutability after ending."""

    def test_immutability_comprehensive(self) -> None:
        """Test all modification methods raise errors after ending."""
        span = Span(name="test-span")
        span.end()

        # Test all modification methods
        with pytest.raises(RuntimeError):
            span.set_attribute("key", "value")

        with pytest.raises(RuntimeError):
            span.add_event("event")

        with pytest.raises(RuntimeError):
            span.set_status(SpanStatus.ERROR)

        with pytest.raises(RuntimeError):
            span.end()

    def test_reading_after_end_works(self) -> None:
        """Test that reading span data after end still works."""
        span = Span(name="test-span")
        span.set_attribute("model", "gpt-4")
        span.add_event("test_event")
        span.end()

        # All read operations should work
        assert span.name == "test-span"
        assert span.span_id is not None
        assert span.status == SpanStatus.SUCCESS
        assert span.attributes["model"] == "gpt-4"
        assert len(span.events) == 1
        assert span.duration_ms is not None


class TestSpanSerialization:
    """Tests for span serialization."""

    def test_to_dict_basic(self) -> None:
        """Test converting span to dictionary."""
        span = Span(name="test-span", span_type=SpanType.AGENT)
        span.set_attribute("model", "gpt-4")
        span.add_event("started")
        span.end()

        data = span.to_dict()

        assert data["name"] == "test-span"
        assert data["span_type"] == "agent"
        assert data["span_id"] == span.span_id
        assert data["trace_id"] == span.trace_id
        assert data["parent_span_id"] is None
        assert data["status"] == "success"
        assert data["attributes"] == {"model": "gpt-4"}
        assert len(data["events"]) == 1
        assert data["events"][0]["name"] == "started"
        assert data["duration_ms"] is not None

    def test_to_dict_with_parent(self) -> None:
        """Test to_dict includes parent span ID."""
        span = Span(name="child-span", parent_span_id="parent-123")
        data = span.to_dict()

        assert data["parent_span_id"] == "parent-123"

    def test_from_dict_basic(self) -> None:
        """Test creating span from dictionary."""
        data = {
            "span_id": "span-123",
            "trace_id": "trace-456",
            "parent_span_id": None,
            "name": "test-span",
            "span_type": "llm",
            "started_at": "2024-01-01T12:00:00+00:00",
            "ended_at": "2024-01-01T12:00:01+00:00",
            "status": "success",
            "status_message": "Completed",
            "attributes": {"model": "gpt-4"},
            "events": [
                {
                    "timestamp": "2024-01-01T12:00:00.500000+00:00",
                    "name": "api_call",
                    "attributes": {"endpoint": "/chat"},
                }
            ],
        }

        span = Span.from_dict(data)

        assert span.span_id == "span-123"
        assert span.trace_id == "trace-456"
        assert span.name == "test-span"
        assert span.span_type == SpanType.LLM
        assert span.status == SpanStatus.SUCCESS
        assert span.status_message == "Completed"
        assert span.attributes == {"model": "gpt-4"}
        assert len(span.events) == 1
        assert span.events[0].name == "api_call"
        assert span._ended is True

    def test_serialization_roundtrip(self) -> None:
        """Test that serialization and deserialization preserves data."""
        original = Span(
            name="test-span",
            span_type=SpanType.TOOL,
            parent_span_id="parent-456",
        )
        original.set_attribute("tool_name", "calculator")
        original.set_attribute("input", "2+2")
        original.add_event("computation_started", {"timestamp": "now"})
        original.add_event("computation_completed", {"result": 4})
        original.set_status(SpanStatus.SUCCESS, "Calculation complete")
        original.end()

        # Serialize and deserialize
        data = original.to_dict()
        reconstructed = Span.from_dict(data)

        # Verify all fields match
        assert reconstructed.span_id == original.span_id
        assert reconstructed.trace_id == original.trace_id
        assert reconstructed.parent_span_id == original.parent_span_id
        assert reconstructed.name == original.name
        assert reconstructed.span_type == original.span_type
        assert reconstructed.started_at == original.started_at
        assert reconstructed.ended_at == original.ended_at
        assert reconstructed.status == original.status
        assert reconstructed.status_message == original.status_message
        assert reconstructed.attributes == original.attributes
        assert len(reconstructed.events) == len(original.events)
        assert reconstructed._ended == original._ended
        assert reconstructed.duration_ms == original.duration_ms

    def test_serialization_roundtrip_with_no_end(self) -> None:
        """Test serialization roundtrip for span that hasn't ended."""
        original = Span(name="active-span", span_type=SpanType.AGENT)
        original.set_attribute("state", "running")

        data = original.to_dict()
        reconstructed = Span.from_dict(data)

        assert reconstructed.ended_at is None
        assert reconstructed._ended is False
        assert reconstructed.duration_ms is None
        assert reconstructed.status == SpanStatus.PENDING

        # Should be able to modify reconstructed span
        reconstructed.set_attribute("new_key", "new_value")
        assert reconstructed.attributes["new_key"] == "new_value"


class TestSpanEvent:
    """Tests for SpanEvent."""

    def test_span_event_creation(self) -> None:
        """Test creating a span event."""
        timestamp = datetime.now(timezone.utc)
        event = SpanEvent(
            timestamp=timestamp,
            name="test_event",
            attributes={"key": "value"},
        )

        assert event.timestamp == timestamp
        assert event.name == "test_event"
        assert event.attributes == {"key": "value"}

    def test_span_event_to_dict(self) -> None:
        """Test converting span event to dictionary."""
        timestamp = datetime.now(timezone.utc)
        event = SpanEvent(
            timestamp=timestamp,
            name="test_event",
            attributes={"key": "value"},
        )

        data = event.to_dict()

        assert data["timestamp"] == timestamp.isoformat()
        assert data["name"] == "test_event"
        assert data["attributes"] == {"key": "value"}

    def test_span_event_from_dict(self) -> None:
        """Test creating span event from dictionary."""
        data = {
            "timestamp": "2024-01-01T12:00:00+00:00",
            "name": "test_event",
            "attributes": {"key": "value"},
        }

        event = SpanEvent.from_dict(data)

        assert event.name == "test_event"
        assert event.attributes == {"key": "value"}
        assert isinstance(event.timestamp, datetime)

    def test_span_event_roundtrip(self) -> None:
        """Test span event serialization roundtrip."""
        original = SpanEvent(
            timestamp=datetime.now(timezone.utc),
            name="test_event",
            attributes={"complex": {"nested": "data"}},
        )

        data = original.to_dict()
        reconstructed = SpanEvent.from_dict(data)

        assert reconstructed.name == original.name
        assert reconstructed.attributes == original.attributes
        # Timestamps should be equal (after roundtrip through ISO format)
        assert reconstructed.timestamp.isoformat() == original.timestamp.isoformat()
