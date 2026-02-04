"""Tests for Anthropic SDK instrumentation."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any
from datetime import datetime, timezone

from prela.core.span import SpanType, SpanStatus
from prela.instrumentation.anthropic import (
    AnthropicInstrumentor,
    TracedMessageStream,
    TracedAsyncMessageStream,
)


# Mock classes to simulate anthropic SDK structures
class MockUsage:
    """Mock for anthropic usage object."""

    def __init__(
        self, input_tokens: int = 100, output_tokens: int = 50
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockTextBlock:
    """Mock for anthropic text content block."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mock for anthropic tool use content block."""

    def __init__(
        self, tool_id: str, name: str, input_data: dict[str, Any]
    ) -> None:
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = input_data


class MockThinkingBlock:
    """Mock for anthropic thinking content block."""

    def __init__(self, thinking: str) -> None:
        self.type = "thinking"
        self.thinking = thinking


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
        self.type = "message"
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


class MockMessageDelta:
    """Mock for message delta."""

    def __init__(self, stop_reason: str | None = None) -> None:
        if stop_reason:
            self.stop_reason = stop_reason


class MockMessageStream:
    """Mock for MessageStream."""

    def __init__(self, events: list[MockStreamEvent]) -> None:
        self.events = events
        self._entered = False

    def __enter__(self) -> MockMessageStream:
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._entered = False

    def __iter__(self) -> Any:
        return iter(self.events)


class MockAsyncMessageStream:
    """Mock for AsyncMessageStream."""

    def __init__(self, events: list[MockStreamEvent]) -> None:
        self.events = events
        self._entered = False

    async def __aenter__(self) -> MockAsyncMessageStream:
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._entered = False

    async def __aiter__(self) -> Any:
        for event in self.events:
            yield event


class MockMessages:
    """Mock for messages resource."""

    def __init__(self) -> None:
        self.create = Mock()
        self.stream = Mock()


class MockAsyncMessages:
    """Mock for async messages resource."""

    def __init__(self) -> None:
        self.create = AsyncMock()
        self.stream = AsyncMock()


class MockAnthropic:
    """Mock for Anthropic client."""

    def __init__(self) -> None:
        self.messages = MockMessages()


class MockAsyncAnthropic:
    """Mock for AsyncAnthropic client."""

    def __init__(self) -> None:
        self.messages = MockAsyncMessages()


class MockAPIError(Exception):
    """Mock for anthropic APIError."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def mock_tracer() -> Mock:
    """Create a mock tracer."""
    tracer = Mock()
    span = Mock()

    # Set up span methods
    span.set_attribute = Mock()
    span.add_event = Mock()
    span.set_status = Mock()
    span.end = Mock()

    tracer.start_span = Mock(return_value=span)
    return tracer


@pytest.fixture
def mock_anthropic_module() -> Mock:
    """Create a mock anthropic module."""
    module = Mock()
    module.__name__ = "anthropic"
    module.Anthropic = MockAnthropic
    module.AsyncAnthropic = MockAsyncAnthropic

    # Create instances for __new__ calls
    module.Anthropic.__new__ = Mock(return_value=MockAnthropic())
    module.AsyncAnthropic.__new__ = Mock(return_value=MockAsyncAnthropic())

    return module


class TestAnthropicInstrumentor:
    """Tests for AnthropicInstrumentor class."""

    def test_init(self) -> None:
        """Test instrumentor initialization."""
        instrumentor = AnthropicInstrumentor()
        assert instrumentor._tracer is None
        assert instrumentor._anthropic_module is None
        assert instrumentor._messages_module is None
        assert instrumentor._async_messages_module is None

    def test_instrument_without_anthropic_installed(
        self, mock_tracer: Mock
    ) -> None:
        """Test instrumentation fails when anthropic is not installed."""
        instrumentor = AnthropicInstrumentor()

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="anthropic package is not installed"):
                instrumentor.instrument(mock_tracer)

    def test_instrument_success(
        self, mock_tracer: Mock, mock_anthropic_module: Mock
    ) -> None:
        """Test successful instrumentation."""
        instrumentor = AnthropicInstrumentor()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            with patch(
                "prela.instrumentation.anthropic.wrap_function"
            ) as mock_wrap:
                # Mock import
                with patch("builtins.__import__", return_value=mock_anthropic_module):
                    instrumentor.instrument(mock_tracer)

                # Check that wrap_function was called for sync and async methods
                assert mock_wrap.call_count >= 2  # At least sync and async create

        assert instrumentor._tracer is mock_tracer
        assert instrumentor._anthropic_module is mock_anthropic_module

    def test_instrument_idempotent(
        self, mock_tracer: Mock, mock_anthropic_module: Mock
    ) -> None:
        """Test that calling instrument multiple times doesn't double-wrap."""
        instrumentor = AnthropicInstrumentor()

        # Set up as already instrumented
        instrumentor._tracer = mock_tracer
        instrumentor._messages_module = Mock()
        instrumentor._messages_module.__prela_originals__ = {"create": Mock()}

        # Second call should be a no-op
        instrumentor.instrument(mock_tracer)

        # Should still be instrumented
        assert instrumentor.is_instrumented

    def test_uninstrument(
        self, mock_tracer: Mock, mock_anthropic_module: Mock
    ) -> None:
        """Test uninstrumentation."""
        instrumentor = AnthropicInstrumentor()

        # Set up as instrumented
        instrumentor._tracer = mock_tracer
        instrumentor._anthropic_module = mock_anthropic_module

        # Create mock modules with proper structure
        # The modules need to have __prela_originals__ to be considered instrumented
        messages_module = Mock()
        messages_module.__prela_originals__ = {"create": Mock(), "stream": Mock()}

        async_messages_module = Mock()

        instrumentor._messages_module = messages_module
        instrumentor._async_messages_module = async_messages_module

        with patch("prela.instrumentation.anthropic.unwrap_function") as mock_unwrap:
            instrumentor.uninstrument()

            # Check that unwrap_function was called for both sync and async
            # 2 calls for sync (create, stream) + 2 calls for async (create, stream) = 4
            assert mock_unwrap.call_count == 4

            # Verify the calls were for the right methods
            expected_calls = [
                (messages_module, "create"),
                (messages_module, "stream"),
                (async_messages_module, "create"),
                (async_messages_module, "stream"),
            ]

            actual_calls = [
                (call[0][0], call[0][1]) for call in mock_unwrap.call_args_list
            ]

            for expected in expected_calls:
                assert expected in actual_calls

        # Should be cleaned up
        assert instrumentor._tracer is None
        assert instrumentor._anthropic_module is None

    def test_uninstrument_when_not_instrumented(self) -> None:
        """Test uninstrumentation when not instrumented is a no-op."""
        instrumentor = AnthropicInstrumentor()
        instrumentor.uninstrument()  # Should not raise

    def test_is_instrumented(self, mock_tracer: Mock) -> None:
        """Test is_instrumented property."""
        instrumentor = AnthropicInstrumentor()

        # Not instrumented initially
        assert not instrumentor.is_instrumented

        # Set up as instrumented
        instrumentor._tracer = mock_tracer
        instrumentor._messages_module = Mock()
        instrumentor._messages_module.__prela_originals__ = {"create": Mock()}

        # Now should be instrumented
        assert instrumentor.is_instrumented


class TestMessagesCreateSync:
    """Tests for synchronous messages.create instrumentation."""

    def test_basic_request_response(self, mock_tracer: Mock) -> None:
        """Test basic messages.create call."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create mock response
        response = MockMessage(
            content=[MockTextBlock("Hello! How can I help?")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=150, output_tokens=89),
        )

        # Create original function
        original_func = Mock(return_value=response)

        # Create wrapper
        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        # Call the wrapper
        result = wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Check result
        assert result is response

        # Check span was created
        mock_tracer.start_span.assert_called_once()
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["name"] == "anthropic.messages.create"
        assert call_kwargs["span_type"] == SpanType.LLM

        # Check span attributes were set
        span = mock_tracer.start_span.return_value
        assert span.set_attribute.called

        # Verify key attributes
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["llm.vendor"] == "anthropic"
        assert attribute_calls["llm.model"] == "claude-sonnet-4-20250514"
        assert attribute_calls["llm.max_tokens"] == 1024
        assert attribute_calls["llm.input_tokens"] == 150
        assert attribute_calls["llm.output_tokens"] == 89
        assert attribute_calls["llm.stop_reason"] == "end_turn"

        # Check events were added
        assert span.add_event.called

        # Check span was ended and marked successful
        span.set_status.assert_called_with(SpanStatus.SUCCESS)
        span.end.assert_called_once()

    def test_with_system_prompt(self, mock_tracer: Mock) -> None:
        """Test messages.create with system prompt."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockMessage(
            content=[MockTextBlock("I'm a helpful assistant.")],
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Check system prompt was captured
        span = mock_tracer.start_span.return_value
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["llm.system"] == "You are a helpful assistant."

    def test_with_temperature(self, mock_tracer: Mock) -> None:
        """Test messages.create with temperature parameter."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockMessage(
            content=[MockTextBlock("Response text")],
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Check temperature was captured
        span = mock_tracer.start_span.return_value
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["llm.temperature"] == 0.7

    def test_tool_use_response(self, mock_tracer: Mock) -> None:
        """Test messages.create with tool use in response."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create response with tool use
        response = MockMessage(
            content=[
                MockTextBlock("Let me search for that."),
                MockToolUseBlock(
                    tool_id="tool_abc123",
                    name="search",
                    input_data={"query": "test"},
                ),
            ],
            stop_reason="tool_use",
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Search for test"}],
        )

        # Check tool use event was added
        span = mock_tracer.start_span.return_value
        event_calls = [call[1] for call in span.add_event.call_args_list]

        # Find the tool_use event
        tool_use_events = [e for e in event_calls if e.get("name") == "llm.tool_use"]
        assert len(tool_use_events) == 1

        tool_calls = tool_use_events[0]["attributes"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "tool_abc123"
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["input"] == {"query": "test"}

    def test_thinking_blocks(self, mock_tracer: Mock) -> None:
        """Test messages.create with extended thinking blocks."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create response with thinking
        response = MockMessage(
            content=[
                MockThinkingBlock("Let me think about this..."),
                MockTextBlock("Here's my answer."),
            ],
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Question"}],
        )

        # Check thinking event was added
        span = mock_tracer.start_span.return_value
        event_calls = [call[1] for call in span.add_event.call_args_list]

        # Find the thinking event
        thinking_events = [e for e in event_calls if e.get("name") == "llm.thinking"]
        assert len(thinking_events) == 1

        thinking_content = thinking_events[0]["attributes"]["thinking"]
        assert "Let me think about this..." in thinking_content

    def test_error_handling(self, mock_tracer: Mock) -> None:
        """Test error handling during API call."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create original function that raises error
        error = MockAPIError("Rate limit exceeded", status_code=429)
        original_func = Mock(side_effect=error)

        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        # Call should raise the error
        with pytest.raises(MockAPIError):
            wrapper(
                Mock(),
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )

        # Check error was recorded
        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.ERROR, "Rate limit exceeded")

        # Check error event was added
        event_calls = [call[1] for call in span.add_event.call_args_list]
        error_events = [e for e in event_calls if e.get("name") == "error"]
        assert len(error_events) == 1

        error_attrs = error_events[0]["attributes"]
        assert error_attrs["error.type"] == "MockAPIError"
        assert error_attrs["error.message"] == "Rate limit exceeded"
        assert error_attrs["error.status_code"] == 429

        # Span should still be ended
        span.end.assert_called_once()

    def test_without_tracer(self) -> None:
        """Test that wrapper works without tracer (defensive)."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = None

        response = MockMessage(content=[MockTextBlock("Hello")])
        original_func = Mock(return_value=response)

        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=False)

        # Should still work, just without tracing
        result = wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert result is response
        original_func.assert_called_once()


class TestMessagesCreateAsync:
    """Tests for asynchronous messages.create instrumentation."""

    @pytest.mark.asyncio
    async def test_basic_request_response(self, mock_tracer: Mock) -> None:
        """Test basic async messages.create call."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create mock response
        response = MockMessage(
            content=[MockTextBlock("Hello! How can I help?")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=150, output_tokens=89),
        )

        # Create async original function
        async def original_func(*args: Any, **kwargs: Any) -> MockMessage:
            return response

        # Create wrapper
        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=True)

        # Call the wrapper
        result = await wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Check result
        assert result is response

        # Check span was created
        mock_tracer.start_span.assert_called_once()

        # Check span attributes
        span = mock_tracer.start_span.return_value
        assert span.set_attribute.called

        # Check span was ended
        span.set_status.assert_called_with(SpanStatus.SUCCESS)
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_tracer: Mock) -> None:
        """Test error handling during async API call."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create async function that raises error
        async def original_func(*args: Any, **kwargs: Any) -> None:
            raise MockAPIError("Network error", status_code=500)

        wrapper = instrumentor._create_messages_wrapper(original_func, is_async=True)

        # Call should raise the error
        with pytest.raises(MockAPIError):
            await wrapper(
                Mock(),
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )

        # Check error was recorded
        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.ERROR, "Network error")
        span.end.assert_called_once()


class TestMessagesStreamSync:
    """Tests for synchronous streaming instrumentation."""

    def test_basic_streaming(self, mock_tracer: Mock) -> None:
        """Test basic streaming with text deltas."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create stream events
        events = [
            MockStreamEvent("message_start"),
            MockStreamEvent(
                "content_block_start",
                index=0,
                content_block=MockTextBlock(""),
            ),
            MockStreamEvent(
                "content_block_delta",
                index=0,
                delta=MockTextDelta("Hello"),
            ),
            MockStreamEvent(
                "content_block_delta",
                index=0,
                delta=MockTextDelta(" world"),
            ),
            MockStreamEvent(
                "message_delta",
                delta=MockMessageDelta(stop_reason="end_turn"),
                usage=MockUsage(output_tokens=5),
            ),
            MockStreamEvent("message_stop"),
        ]

        stream = MockMessageStream(events)
        original_func = Mock(return_value=stream)

        wrapper = instrumentor._create_stream_wrapper(original_func, is_async=False)

        # Call wrapper
        result = wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Should return a TracedMessageStream
        assert isinstance(result, TracedMessageStream)

        # Check span was created with stream flag
        span = mock_tracer.start_span.return_value
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["llm.stream"] is True

        # Consume the stream
        with result as traced_stream:
            consumed_events = list(traced_stream)

        # Check all events were yielded
        assert len(consumed_events) == len(events)

        # Check span was finalized
        assert span.end.called

    def test_streaming_with_tool_calls(self, mock_tracer: Mock) -> None:
        """Test streaming with tool use blocks."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create stream events with tool use
        tool_block = MockToolUseBlock(
            tool_id="tool_123",
            name="search",
            input_data={},
        )

        events = [
            MockStreamEvent("message_start"),
            MockStreamEvent(
                "content_block_start",
                index=0,
                content_block=tool_block,
            ),
            MockStreamEvent(
                "message_delta",
                delta=MockMessageDelta(stop_reason="tool_use"),
            ),
            MockStreamEvent("message_stop"),
        ]

        stream = MockMessageStream(events)
        original_func = Mock(return_value=stream)

        wrapper = instrumentor._create_stream_wrapper(original_func, is_async=False)
        result = wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Search"}],
        )

        # Consume the stream
        with result as traced_stream:
            list(traced_stream)

        # Check tool use was captured
        span = mock_tracer.start_span.return_value
        # Tool calls should be recorded during finalization

    def test_streaming_error(self, mock_tracer: Mock) -> None:
        """Test error handling during streaming."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create original function that raises error
        original_func = Mock(side_effect=MockAPIError("Connection failed", 503))

        wrapper = instrumentor._create_stream_wrapper(original_func, is_async=False)

        # Call should raise error
        with pytest.raises(MockAPIError):
            wrapper(
                Mock(),
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )

        # Check error was recorded
        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.ERROR, "Connection failed")
        span.end.assert_called_once()


class TestMessagesStreamAsync:
    """Tests for asynchronous streaming instrumentation."""

    @pytest.mark.asyncio
    async def test_basic_streaming(self, mock_tracer: Mock) -> None:
        """Test basic async streaming with text deltas."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create stream events
        events = [
            MockStreamEvent("message_start"),
            MockStreamEvent(
                "content_block_start",
                index=0,
                content_block=MockTextBlock(""),
            ),
            MockStreamEvent(
                "content_block_delta",
                index=0,
                delta=MockTextDelta("Hello"),
            ),
            MockStreamEvent(
                "content_block_delta",
                index=0,
                delta=MockTextDelta(" async"),
            ),
            MockStreamEvent(
                "message_delta",
                delta=MockMessageDelta(stop_reason="end_turn"),
                usage=MockUsage(output_tokens=5),
            ),
            MockStreamEvent("message_stop"),
        ]

        stream = MockAsyncMessageStream(events)

        async def original_func(*args: Any, **kwargs: Any) -> MockAsyncMessageStream:
            return stream

        wrapper = instrumentor._create_stream_wrapper(original_func, is_async=True)

        # Call wrapper
        result = await wrapper(
            Mock(),
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Should return a TracedAsyncMessageStream
        assert isinstance(result, TracedAsyncMessageStream)

        # Consume the stream
        async with result as traced_stream:
            consumed_events = []
            async for event in traced_stream:
                consumed_events.append(event)

        # Check all events were yielded
        assert len(consumed_events) == len(events)

        # Check span was finalized
        span = mock_tracer.start_span.return_value
        assert span.end.called

    @pytest.mark.asyncio
    async def test_streaming_error(self, mock_tracer: Mock) -> None:
        """Test error handling during async streaming."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create async function that raises error
        async def original_func(*args: Any, **kwargs: Any) -> None:
            raise MockAPIError("Timeout", 504)

        wrapper = instrumentor._create_stream_wrapper(original_func, is_async=True)

        # Call should raise error
        with pytest.raises(MockAPIError):
            await wrapper(
                Mock(),
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )

        # Check error was recorded
        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.ERROR, "Timeout")
        span.end.assert_called_once()


class TestTracedMessageStream:
    """Tests for TracedMessageStream wrapper."""

    def test_context_manager(self, mock_tracer: Mock) -> None:
        """Test context manager protocol."""
        span = Mock()
        events = [MockStreamEvent("message_start")]
        stream = MockMessageStream(events)

        traced = TracedMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        with traced:
            assert stream._entered

        assert not stream._entered
        span.end.assert_called_once()

    def test_iteration(self, mock_tracer: Mock) -> None:
        """Test iteration over stream events."""
        span = Mock()
        events = [
            MockStreamEvent("message_start"),
            MockStreamEvent("message_stop"),
        ]
        stream = MockMessageStream(events)

        traced = TracedMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        consumed = []
        with traced:
            for event in traced:
                consumed.append(event)

        assert len(consumed) == 2

    def test_text_aggregation(self, mock_tracer: Mock) -> None:
        """Test text content aggregation."""
        span = Mock()
        events = [
            MockStreamEvent(
                "content_block_delta",
                delta=MockTextDelta("Hello"),
            ),
            MockStreamEvent(
                "content_block_delta",
                delta=MockTextDelta(" world"),
            ),
        ]
        stream = MockMessageStream(events)

        traced = TracedMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        with traced:
            for _ in traced:
                pass

        # Check aggregated text was added as event
        assert span.add_event.called


class TestTracedAsyncMessageStream:
    """Tests for TracedAsyncMessageStream wrapper."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_tracer: Mock) -> None:
        """Test async context manager protocol."""
        span = Mock()
        events = [MockStreamEvent("message_start")]
        stream = MockAsyncMessageStream(events)

        traced = TracedAsyncMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        async with traced:
            assert stream._entered

        assert not stream._entered
        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_iteration(self, mock_tracer: Mock) -> None:
        """Test async iteration over stream events."""
        span = Mock()
        events = [
            MockStreamEvent("message_start"),
            MockStreamEvent("message_stop"),
        ]
        stream = MockAsyncMessageStream(events)

        traced = TracedAsyncMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        consumed = []
        async with traced:
            async for event in traced:
                consumed.append(event)

        assert len(consumed) == 2


class TestErrorHandling:
    """Tests for defensive error handling."""

    def test_extract_response_with_malformed_response(
        self, mock_tracer: Mock
    ) -> None:
        """Test that malformed responses don't crash extraction."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create a response object that raises errors when accessing attributes
        class BrokenResponse:
            @property
            def model(self) -> None:
                raise RuntimeError("Broken!")

            @property
            def usage(self) -> None:
                raise RuntimeError("Broken!")

        response = BrokenResponse()
        span = Mock()

        # Should not raise, just log and continue
        instrumentor._extract_response_attributes(span, response)

        # Span methods should not have been called due to errors
        # but the function should complete without raising

    def test_handle_tool_use_with_malformed_content(
        self, mock_tracer: Mock
    ) -> None:
        """Test that malformed tool use content doesn't crash."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        # Create a response with broken content blocks
        class BrokenBlock:
            @property
            def type(self) -> str:
                return "tool_use"

            @property
            def id(self) -> None:
                raise RuntimeError("Broken!")

        class BrokenResponse:
            @property
            def content(self) -> list[Any]:
                raise RuntimeError("Broken!")

        response = BrokenResponse()
        span = Mock()

        # Should not raise
        instrumentor._handle_tool_use(span, response)

    def test_handle_thinking_with_malformed_content(
        self, mock_tracer: Mock
    ) -> None:
        """Test that malformed thinking blocks don't crash."""
        instrumentor = AnthropicInstrumentor()
        instrumentor._tracer = mock_tracer

        class BrokenResponse:
            @property
            def content(self) -> None:
                raise RuntimeError("Broken!")

        response = BrokenResponse()
        span = Mock()

        # Should not raise
        instrumentor._handle_thinking_blocks(span, response)

    def test_streaming_with_malformed_events(self, mock_tracer: Mock) -> None:
        """Test that malformed streaming events don't crash."""
        span = Mock()

        # Create an event that will raise errors
        class BrokenEvent:
            @property
            def type(self) -> str:
                raise RuntimeError("Broken!")

        events = [BrokenEvent()]
        stream = MockMessageStream(events)

        traced = TracedMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        # Should not raise, just process what it can
        with traced:
            for _ in traced:
                pass

    def test_instrument_failure_cleanup(
        self, mock_tracer: Mock, mock_anthropic_module: Mock
    ) -> None:
        """Test that failed instrumentation cleans up properly."""
        instrumentor = AnthropicInstrumentor()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            with patch(
                "prela.instrumentation.anthropic.wrap_function",
                side_effect=RuntimeError("Wrap failed"),
            ):
                with patch("builtins.__import__", return_value=mock_anthropic_module):
                    with pytest.raises(RuntimeError, match="Failed to instrument"):
                        instrumentor.instrument(mock_tracer)

        # Should be cleaned up
        assert instrumentor._tracer is None
        assert instrumentor._anthropic_module is None

    @pytest.mark.asyncio
    async def test_async_stream_with_exception_during_iteration(
        self, mock_tracer: Mock
    ) -> None:
        """Test async stream handles exceptions during iteration."""
        span = Mock()

        # Create stream that raises during iteration
        class BrokenAsyncStream:
            async def __aenter__(self) -> BrokenAsyncStream:
                return self

            async def __aexit__(
                self, exc_type: Any, exc_val: Any, exc_tb: Any
            ) -> None:
                pass

            def __aiter__(self) -> BrokenAsyncStream:
                return self

            async def __anext__(self) -> Any:
                raise RuntimeError("Stream broke!")

        stream = BrokenAsyncStream()

        traced = TracedAsyncMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        # Should propagate the exception and set error status
        with pytest.raises(RuntimeError, match="Stream broke"):
            async with traced:
                async for _ in traced:
                    pass

        # Error status should be set
        span.set_status.assert_called_with(SpanStatus.ERROR, "Stream broke!")

    def test_sync_stream_with_exception_during_iteration(
        self, mock_tracer: Mock
    ) -> None:
        """Test sync stream handles exceptions during iteration."""
        span = Mock()

        # Create stream that raises during iteration
        class BrokenStream:
            def __enter__(self) -> BrokenStream:
                return self

            def __exit__(
                self, exc_type: Any, exc_val: Any, exc_tb: Any
            ) -> None:
                pass

            def __iter__(self) -> Any:
                raise RuntimeError("Stream broke!")

        stream = BrokenStream()

        traced = TracedMessageStream(
            stream=stream,
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        # Should propagate the exception and set error status
        with pytest.raises(RuntimeError, match="Stream broke"):
            with traced:
                for _ in traced:
                    pass

        # Error status should be set
        span.set_status.assert_called_with(SpanStatus.ERROR, "Stream broke!")
