"""Tests for Anthropic instrumentation with replay capture."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from prela.core.replay import ReplaySnapshot
from prela.core.tracer import Tracer
from prela.instrumentation.anthropic import AnthropicInstrumentor


# Mock classes to simulate anthropic SDK structures
class MockUsage:
    """Mock for anthropic usage object."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockTextBlock:
    """Mock for anthropic text content block."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockMessage:
    """Mock for anthropic Message object."""

    def __init__(
        self,
        content: list[Any],
        model: str = "claude-sonnet-4-20250514",
        stop_reason: str = "end_turn",
        usage: MockUsage | None = None,
        msg_id: str = "msg_123",
    ) -> None:
        self.id = msg_id
        self.model = model
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or MockUsage()


class MockStreamEvent:
    """Mock for streaming event."""

    def __init__(self, event_type: str, **kwargs: Any) -> None:
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockTextDelta:
    """Mock for text delta in streaming."""

    def __init__(self, text: str) -> None:
        self.type = "text_delta"
        self.text = text


class CapturingExporter:
    """Exporter that captures spans for testing."""

    def __init__(self):
        self.exported_spans = []

    def export(self, spans):
        self.exported_spans.extend(spans)

    def shutdown(self):
        pass


@pytest.fixture
def tracer_with_replay():
    """Create a real tracer with replay capture enabled."""
    exporter = CapturingExporter()
    tracer = Tracer(service_name="test", exporter=exporter, capture_for_replay=True)
    tracer.exporter = exporter  # Keep reference for tests
    return tracer


@pytest.fixture
def tracer_without_replay():
    """Create a real tracer with replay capture disabled."""
    exporter = CapturingExporter()
    tracer = Tracer(service_name="test", exporter=exporter, capture_for_replay=False)
    tracer.exporter = exporter  # Keep reference for tests
    return tracer


class TestAnthropicReplayCapture:
    """Test replay capture for Anthropic instrumentation."""

    def test_sync_messages_create_with_replay(self, tracer_with_replay):
        """Test sync messages.create captures replay data."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = tracer_with_replay
        instrumentor._anthropic_module = MagicMock()
        instrumentor._messages_module = MagicMock()

        # Create mock response
        mock_response = MockMessage(
            content=[MockTextBlock("Hello! How can I help you today?")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=150, output_tokens=89),
            msg_id="msg_123",
        )

        # Create a mock original function
        def mock_create(self_obj, *args, **kwargs):
            return mock_response

        # Call the traced function
        result = instrumentor._trace_messages_create_sync(
            mock_create,
            MagicMock(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful",
            max_tokens=1024,
            temperature=0.7,
        )

        # Verify response
        assert result == mock_response

        # Get the span from exporter
        spans = tracer_with_replay.exporter.exported_spans
        assert len(spans) == 1
        span = spans[0]

        # Verify span has replay snapshot
        assert hasattr(span, "replay_snapshot")
        assert span.replay_snapshot is not None
        assert isinstance(span.replay_snapshot, ReplaySnapshot)

        # Verify request data
        assert span.replay_snapshot.llm_request is not None
        assert span.replay_snapshot.llm_request["model"] == "claude-sonnet-4-20250514"
        assert span.replay_snapshot.llm_request["messages"] == [
            {"role": "user", "content": "Hello"}
        ]
        assert span.replay_snapshot.llm_request["system"] == "You are helpful"
        assert span.replay_snapshot.llm_request["max_tokens"] == 1024
        assert span.replay_snapshot.llm_request["temperature"] == 0.7

        # Verify response data
        assert span.replay_snapshot.llm_response is not None
        assert span.replay_snapshot.llm_response["text"] == "Hello! How can I help you today?"
        assert span.replay_snapshot.llm_response["finish_reason"] == "end_turn"
        assert span.replay_snapshot.llm_response["model"] == "claude-sonnet-4-20250514"
        assert span.replay_snapshot.llm_response["prompt_tokens"] == 150
        assert span.replay_snapshot.llm_response["completion_tokens"] == 89

        # Verify model info
        assert span.replay_snapshot.model_info is not None
        assert span.replay_snapshot.model_info["model"] == "claude-sonnet-4-20250514"
        assert span.replay_snapshot.model_info["id"] == "msg_123"

    def test_sync_messages_create_without_replay(self, tracer_without_replay):
        """Test sync messages.create does not capture replay when disabled."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = tracer_without_replay
        instrumentor._anthropic_module = MagicMock()
        instrumentor._messages_module = MagicMock()

        # Create mock response
        mock_response = MockMessage(
            content=[MockTextBlock("Response text")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=150, output_tokens=89),
            msg_id="msg_456",
        )

        # Create a mock original function
        def mock_create(self_obj, *args, **kwargs):
            return mock_response

        # Call the traced function
        result = instrumentor._trace_messages_create_sync(
            mock_create,
            MagicMock(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify response
        assert result == mock_response

        # Get the span from exporter
        spans = tracer_without_replay.exporter.exported_spans
        assert len(spans) == 1
        span = spans[0]

        # Verify span does NOT have replay snapshot
        assert not hasattr(span, "replay_snapshot") or span.replay_snapshot is None

    @pytest.mark.asyncio
    async def test_async_messages_create_with_replay(self, tracer_with_replay):
        """Test async messages.create captures replay data."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = tracer_with_replay
        instrumentor._anthropic_module = MagicMock()
        instrumentor._async_messages_module = MagicMock()

        # Create mock response
        mock_response = MockMessage(
            content=[MockTextBlock("Async response text")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=200, output_tokens=120),
            msg_id="msg_456",
        )

        # Create a mock async original function
        async def mock_create_async(self_obj, *args, **kwargs):
            return mock_response

        # Call the traced function
        result = await instrumentor._trace_messages_create_async(
            mock_create_async,
            MagicMock(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Async test"}],
            max_tokens=2048,
        )

        # Verify response
        assert result == mock_response

        # Get the span from exporter
        spans = tracer_with_replay.exporter.exported_spans
        assert len(spans) == 1
        span = spans[0]

        # Verify span has replay snapshot
        assert hasattr(span, "replay_snapshot")
        assert span.replay_snapshot is not None
        assert isinstance(span.replay_snapshot, ReplaySnapshot)

        # Verify request data
        assert span.replay_snapshot.llm_request["model"] == "claude-sonnet-4-20250514"
        assert span.replay_snapshot.llm_request["messages"] == [{"role": "user", "content": "Async test"}]
        assert span.replay_snapshot.llm_request["max_tokens"] == 2048

        # Verify response data
        assert span.replay_snapshot.llm_response["text"] == "Async response text"
        assert span.replay_snapshot.llm_response["prompt_tokens"] == 200
        assert span.replay_snapshot.llm_response["completion_tokens"] == 120

    def test_streaming_with_replay(self, tracer_with_replay):
        """Test streaming messages captures replay data with chunks."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = tracer_with_replay
        instrumentor._anthropic_module = MagicMock()
        instrumentor._messages_module = MagicMock()

        # Create mock stream events
        mock_events = []

        # Event 1: content_block_delta with text
        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.delta = MagicMock(type="text_delta", text="Hello ")
        mock_events.append(event1)

        # Event 2: content_block_delta with text
        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta = MagicMock(type="text_delta", text="world!")
        mock_events.append(event2)

        # Event 3: message_delta with usage
        event3 = MagicMock()
        event3.type = "message_delta"
        event3.usage = MagicMock(output_tokens=50)
        event3.delta = MagicMock(stop_reason="end_turn")
        mock_events.append(event3)

        # Create mock stream
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_events))

        # Create a mock original function
        def mock_stream_func(self_obj, *args, **kwargs):
            return mock_stream

        # Call the traced function
        result = instrumentor._trace_messages_stream_sync(
            mock_stream_func,
            MagicMock(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Stream test"}],
            temperature=0.5,
        )

        # Consume the stream
        with result as stream:
            events = list(stream)

        # Verify events were yielded
        assert len(events) == 3

        # Get the span from exporter
        spans = tracer_with_replay.exporter.exported_spans
        assert len(spans) == 1
        span = spans[0]

        # Verify span has replay snapshot
        assert hasattr(span, "replay_snapshot")
        assert span.replay_snapshot is not None

        # Verify request data
        assert span.replay_snapshot.llm_request["model"] == "claude-sonnet-4-20250514"
        assert span.replay_snapshot.llm_request["messages"] == [{"role": "user", "content": "Stream test"}]
        assert span.replay_snapshot.llm_request["temperature"] == 0.5

        # Verify response data
        assert span.replay_snapshot.llm_response is not None
        assert span.replay_snapshot.llm_response["text"] == "Hello world!"
        assert span.replay_snapshot.llm_response["finish_reason"] == "end_turn"

        # Verify streaming chunks were captured
        assert span.replay_snapshot.llm_streaming_chunks is not None
        assert len(span.replay_snapshot.llm_streaming_chunks) == 3

        # Verify chunk data
        chunk1 = span.replay_snapshot.llm_streaming_chunks[0]
        assert chunk1["type"] == "content_block_delta"
        assert chunk1["delta"]["text"] == "Hello "

        chunk2 = span.replay_snapshot.llm_streaming_chunks[1]
        assert chunk2["type"] == "content_block_delta"
        assert chunk2["delta"]["text"] == "world!"

        chunk3 = span.replay_snapshot.llm_streaming_chunks[2]
        assert chunk3["type"] == "message_delta"
        assert chunk3["usage"]["output_tokens"] == 50

    def test_roundtrip_serialization_with_system_prompt(self, tracer_with_replay):
        """Test that Anthropic-specific fields survive serialization."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = tracer_with_replay
        instrumentor._anthropic_module = MagicMock()
        instrumentor._messages_module = MagicMock()

        # Create mock response
        mock_response = MockMessage(
            content=[MockTextBlock("Test response")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=100, output_tokens=50),
            msg_id="msg_789",
        )

        def mock_create(self_obj, *args, **kwargs):
            return mock_response

        # Call with system prompt (Anthropic-specific)
        instrumentor._trace_messages_create_sync(
            mock_create,
            MagicMock(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful AI assistant",
            max_tokens=512,
        )

        # Get the span from exporter
        span = tracer_with_replay.exporter.exported_spans[0]

        # Serialize
        span_dict = span.to_dict()

        # Verify replay_snapshot in dict
        assert "replay_snapshot" in span_dict
        assert "llm_request" in span_dict["replay_snapshot"]
        assert span_dict["replay_snapshot"]["llm_request"]["system"] == "You are a helpful AI assistant"

        # Deserialize
        from prela.core.span import Span
        restored_span = Span.from_dict(span_dict)

        # Verify system prompt survived roundtrip
        assert restored_span.replay_snapshot is not None
        assert restored_span.replay_snapshot.llm_request["system"] == "You are a helpful AI assistant"
