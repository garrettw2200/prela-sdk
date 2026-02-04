"""Tests for OpenAI SDK instrumentation."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any

from prela.core.span import SpanType, SpanStatus
from prela.instrumentation.openai import (
    OpenAIInstrumentor,
    TracedChatCompletionStream,
    TracedAsyncChatCompletionStream,
)


# Mock classes to simulate OpenAI SDK structures
class MockUsage:
    """Mock for OpenAI usage object."""

    def __init__(
        self,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        total_tokens: int = 150,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockMessage:
    """Mock for chat completion message."""

    def __init__(
        self, content: str, role: str = "assistant", tool_calls: list[Any] | None = None
    ) -> None:
        self.content = content
        self.role = role
        self.tool_calls = tool_calls or []


class MockChoice:
    """Mock for completion choice."""

    def __init__(
        self,
        message: MockMessage | None = None,
        text: str | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.message = message
        self.text = text
        self.finish_reason = finish_reason


class MockChatCompletion:
    """Mock for chat completion response."""

    def __init__(
        self,
        choices: list[MockChoice],
        model: str = "gpt-4",
        usage: MockUsage | None = None,
        completion_id: str = "chatcmpl-123",
    ) -> None:
        self.id = completion_id
        self.model = model
        self.choices = choices
        self.usage = usage or MockUsage()


class MockCompletion:
    """Mock for legacy completion response."""

    def __init__(
        self,
        choices: list[MockChoice],
        model: str = "gpt-3.5-turbo-instruct",
        usage: MockUsage | None = None,
    ) -> None:
        self.id = "cmpl-123"
        self.model = model
        self.choices = choices
        self.usage = usage or MockUsage()


class MockEmbedding:
    """Mock for embedding data."""

    def __init__(self, embedding: list[float], index: int = 0) -> None:
        self.embedding = embedding
        self.index = index


class MockEmbeddingResponse:
    """Mock for embedding response."""

    def __init__(
        self,
        data: list[MockEmbedding],
        model: str = "text-embedding-ada-002",
        usage: MockUsage | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.usage = usage or MockUsage(prompt_tokens=10, total_tokens=10)


class MockStreamChunk:
    """Mock for streaming chunk."""

    def __init__(
        self, content: str | None = None, finish_reason: str | None = None
    ) -> None:
        self.choices = []
        if content is not None or finish_reason is not None:
            delta = Mock()
            delta.content = content
            choice = Mock()
            choice.delta = delta
            choice.finish_reason = finish_reason
            self.choices.append(choice)


class MockFunction:
    """Mock for function call."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class MockToolCall:
    """Mock for tool call."""

    def __init__(self, tool_id: str, function: MockFunction) -> None:
        self.id = tool_id
        self.type = "function"
        self.function = function


class MockChatCompletions:
    """Mock for chat completions resource."""

    def __init__(self) -> None:
        self.create = Mock()


class MockAsyncChatCompletions:
    """Mock for async chat completions resource."""

    def __init__(self) -> None:
        self.create = AsyncMock()


class MockCompletions:
    """Mock for completions resource."""

    def __init__(self) -> None:
        self.create = Mock()


class MockEmbeddings:
    """Mock for embeddings resource."""

    def __init__(self) -> None:
        self.create = Mock()


class MockChat:
    """Mock for chat resource."""

    def __init__(self) -> None:
        self.completions = MockChatCompletions()


class MockAsyncChat:
    """Mock for async chat resource."""

    def __init__(self) -> None:
        self.completions = MockAsyncChatCompletions()


class MockOpenAI:
    """Mock for OpenAI client."""

    def __init__(self) -> None:
        self.chat = MockChat()
        self.completions = MockCompletions()
        self.embeddings = MockEmbeddings()


class MockAsyncOpenAI:
    """Mock for AsyncOpenAI client."""

    def __init__(self) -> None:
        self.chat = MockAsyncChat()


class MockAPIError(Exception):
    """Mock for openai APIError."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def mock_tracer() -> Mock:
    """Create a mock tracer."""
    tracer = Mock()
    span = Mock()

    span.set_attribute = Mock()
    span.add_event = Mock()
    span.set_status = Mock()
    span.end = Mock()

    tracer.start_span = Mock(return_value=span)
    return tracer


@pytest.fixture
def mock_openai_module() -> Mock:
    """Create a mock openai module."""
    module = Mock()
    module.__name__ = "openai"
    module.OpenAI = MockOpenAI
    module.AsyncOpenAI = MockAsyncOpenAI

    module.OpenAI.__new__ = Mock(return_value=MockOpenAI())
    module.AsyncOpenAI.__new__ = Mock(return_value=MockAsyncOpenAI())

    return module


class TestOpenAIInstrumentor:
    """Tests for OpenAIInstrumentor class."""

    def test_init(self) -> None:
        """Test instrumentor initialization."""
        instrumentor = OpenAIInstrumentor()
        assert instrumentor._tracer is None
        assert instrumentor._openai_module is None

    def test_instrument_without_openai_installed(self, mock_tracer: Mock) -> None:
        """Test instrumentation fails when openai is not installed."""
        instrumentor = OpenAIInstrumentor()

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="openai package is not installed"):
                instrumentor.instrument(mock_tracer)

    def test_instrument_success(
        self, mock_tracer: Mock, mock_openai_module: Mock
    ) -> None:
        """Test successful instrumentation."""
        instrumentor = OpenAIInstrumentor()

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with patch("prela.instrumentation.openai.wrap_function") as mock_wrap:
                with patch("builtins.__import__", return_value=mock_openai_module):
                    instrumentor.instrument(mock_tracer)

                # Check that wrap_function was called
                assert mock_wrap.call_count >= 2

        assert instrumentor._tracer is mock_tracer
        assert instrumentor._openai_module is mock_openai_module

    def test_uninstrument(
        self, mock_tracer: Mock, mock_openai_module: Mock
    ) -> None:
        """Test uninstrumentation."""
        instrumentor = OpenAIInstrumentor()

        instrumentor._tracer = mock_tracer
        instrumentor._openai_module = mock_openai_module
        instrumentor._chat_completions_module = Mock()
        instrumentor._chat_completions_module.__prela_originals__ = {"create": Mock()}
        instrumentor._async_chat_completions_module = Mock()
        instrumentor._completions_module = Mock()
        instrumentor._embeddings_module = Mock()

        with patch("prela.instrumentation.openai.unwrap_function") as mock_unwrap:
            instrumentor.uninstrument()
            assert mock_unwrap.call_count >= 2

        assert instrumentor._tracer is None


class TestChatCompletionsSync:
    """Tests for synchronous chat completions instrumentation."""

    def test_basic_chat_completion(self, mock_tracer: Mock) -> None:
        """Test basic chat completion call."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockChatCompletion(
            choices=[MockChoice(message=MockMessage(content="Hello!"))],
            model="gpt-4",
            usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=False
        )

        result = wrapper(
            Mock(),
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result is response

        mock_tracer.start_span.assert_called_once()
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["name"] == "openai.chat.completions.create"
        assert call_kwargs["span_type"] == SpanType.LLM

        span = mock_tracer.start_span.return_value
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["llm.vendor"] == "openai"
        assert attribute_calls["llm.model"] == "gpt-4"
        assert attribute_calls["llm.prompt_tokens"] == 10
        assert attribute_calls["llm.completion_tokens"] == 5
        assert attribute_calls["llm.total_tokens"] == 15

        span.set_status.assert_called_with(SpanStatus.SUCCESS)
        span.end.assert_called_once()

    def test_chat_completion_with_tools(self, mock_tracer: Mock) -> None:
        """Test chat completion with tool calls."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        tool_calls = [
            MockToolCall(
                tool_id="call_123",
                function=MockFunction(name="get_weather", arguments='{"city": "SF"}'),
            )
        ]

        response = MockChatCompletion(
            choices=[
                MockChoice(
                    message=MockMessage(
                        content="Let me check the weather", tool_calls=tool_calls
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=False
        )

        wrapper(Mock(), model="gpt-4", messages=[{"role": "user", "content": "Test"}])

        span = mock_tracer.start_span.return_value
        event_calls = [call[1] for call in span.add_event.call_args_list]

        tool_events = [e for e in event_calls if e.get("name") == "llm.tool_calls"]
        assert len(tool_events) == 1

    def test_chat_completion_streaming(self, mock_tracer: Mock) -> None:
        """Test streaming chat completion."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        chunks = [
            MockStreamChunk(content="Hello"),
            MockStreamChunk(content=" world"),
            MockStreamChunk(finish_reason="stop"),
        ]

        original_func = Mock(return_value=iter(chunks))
        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=False
        )

        result = wrapper(
            Mock(),
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert isinstance(result, TracedChatCompletionStream)

        # Consume the stream
        with result as traced_stream:
            consumed = list(traced_stream)

        assert len(consumed) == 3

    def test_error_handling(self, mock_tracer: Mock) -> None:
        """Test error handling during API call."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        error = MockAPIError("Rate limit exceeded", status_code=429)
        original_func = Mock(side_effect=error)

        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=False
        )

        with pytest.raises(MockAPIError):
            wrapper(Mock(), model="gpt-4", messages=[])

        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.ERROR, "Rate limit exceeded")
        span.end.assert_called_once()


class TestChatCompletionsAsync:
    """Tests for asynchronous chat completions instrumentation."""

    @pytest.mark.asyncio
    async def test_basic_async_chat_completion(self, mock_tracer: Mock) -> None:
        """Test basic async chat completion."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockChatCompletion(
            choices=[MockChoice(message=MockMessage(content="Hello!"))]
        )

        async def original_func(*args: Any, **kwargs: Any) -> MockChatCompletion:
            return response

        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=True
        )

        result = await wrapper(
            Mock(),
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result is response
        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.SUCCESS)


class TestLegacyCompletions:
    """Tests for legacy completions API instrumentation."""

    def test_legacy_completion(self, mock_tracer: Mock) -> None:
        """Test legacy completion call."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockCompletion(
            choices=[MockChoice(text="Hello world!")],
            model="gpt-3.5-turbo-instruct",
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_completions_wrapper(original_func)

        result = wrapper(
            Mock(),
            model="gpt-3.5-turbo-instruct",
            prompt="Say hello",
        )

        assert result is response

        span = mock_tracer.start_span.return_value
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["name"] == "openai.completions.create"

        span.set_status.assert_called_with(SpanStatus.SUCCESS)


class TestEmbeddings:
    """Tests for embeddings API instrumentation."""

    def test_embeddings_creation(self, mock_tracer: Mock) -> None:
        """Test embeddings creation."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockEmbeddingResponse(
            data=[MockEmbedding(embedding=[0.1, 0.2, 0.3])],
            model="text-embedding-ada-002",
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_embeddings_wrapper(original_func)

        result = wrapper(
            Mock(),
            model="text-embedding-ada-002",
            input="Test text",
        )

        assert result is response

        span = mock_tracer.start_span.return_value
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["span_type"] == SpanType.EMBEDDING

        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["embedding.input_count"] == 1
        assert attribute_calls["embedding.count"] == 1
        assert attribute_calls["embedding.dimensions"] == 3


class TestStreamWrappers:
    """Tests for stream wrapper classes."""

    def test_sync_stream_context_manager(self, mock_tracer: Mock) -> None:
        """Test sync stream context manager."""
        span = Mock()
        chunks = [MockStreamChunk(content="test")]

        traced = TracedChatCompletionStream(
            stream=iter(chunks),
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        with traced:
            list(traced)

        span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_stream_context_manager(self, mock_tracer: Mock) -> None:
        """Test async stream context manager."""
        span = Mock()

        async def async_chunks():
            yield MockStreamChunk(content="test")

        traced = TracedAsyncChatCompletionStream(
            stream=async_chunks(),
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        async with traced:
            async for _ in traced:
                pass

        span.end.assert_called_once()


class TestErrorHandling:
    """Tests for defensive error handling."""

    def test_extraction_with_malformed_response(self, mock_tracer: Mock) -> None:
        """Test that malformed responses don't crash extraction."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        class BrokenResponse:
            @property
            def model(self) -> None:
                raise RuntimeError("Broken!")

        response = BrokenResponse()
        span = Mock()

        # Should not raise
        instrumentor._extract_chat_completion_attributes(span, response)

    def test_tool_calls_with_malformed_data(self, mock_tracer: Mock) -> None:
        """Test that malformed tool calls don't crash."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        class BrokenToolCall:
            @property
            def id(self) -> None:
                raise RuntimeError("Broken!")

        span = Mock()
        tool_calls = [BrokenToolCall()]

        # Should not raise
        instrumentor._handle_tool_calls(span, tool_calls)

    def test_chat_completion_with_temperature(self, mock_tracer: Mock) -> None:
        """Test chat completion with temperature parameter."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockChatCompletion(
            choices=[MockChoice(message=MockMessage(content="Response"))]
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=False
        )

        wrapper(
            Mock(),
            model="gpt-4",
            messages=[],
            temperature=0.7,
            max_tokens=100,
        )

        span = mock_tracer.start_span.return_value
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["llm.temperature"] == 0.7
        assert attribute_calls["llm.max_tokens"] == 100

    def test_without_tracer(self) -> None:
        """Test that wrapper works without tracer (defensive)."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = None

        response = MockChatCompletion(
            choices=[MockChoice(message=MockMessage(content="Hello"))]
        )
        original_func = Mock(return_value=response)

        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=False
        )

        result = wrapper(Mock(), model="gpt-4", messages=[])

        assert result is response
        original_func.assert_called_once()

    def test_uninstrument_when_not_instrumented(self) -> None:
        """Test uninstrumentation when not instrumented is a no-op."""
        instrumentor = OpenAIInstrumentor()
        instrumentor.uninstrument()  # Should not raise

    def test_is_instrumented(self, mock_tracer: Mock) -> None:
        """Test is_instrumented property."""
        instrumentor = OpenAIInstrumentor()

        assert not instrumentor.is_instrumented

        instrumentor._tracer = mock_tracer
        instrumentor._chat_completions_module = Mock()
        instrumentor._chat_completions_module.__prela_originals__ = {"create": Mock()}

        assert instrumentor.is_instrumented

    def test_instrument_idempotent(
        self, mock_tracer: Mock, mock_openai_module: Mock
    ) -> None:
        """Test that calling instrument multiple times doesn't double-wrap."""
        instrumentor = OpenAIInstrumentor()

        instrumentor._tracer = mock_tracer
        instrumentor._chat_completions_module = Mock()
        instrumentor._chat_completions_module.__prela_originals__ = {"create": Mock()}

        # Second call should be a no-op
        instrumentor.instrument(mock_tracer)

        assert instrumentor.is_instrumented

    def test_instrument_failure_cleanup(
        self, mock_tracer: Mock, mock_openai_module: Mock
    ) -> None:
        """Test that failed instrumentation cleans up properly."""
        instrumentor = OpenAIInstrumentor()

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with patch(
                "prela.instrumentation.openai.wrap_function",
                side_effect=RuntimeError("Wrap failed"),
            ):
                with patch("builtins.__import__", return_value=mock_openai_module):
                    with pytest.raises(RuntimeError, match="Failed to instrument"):
                        instrumentor.instrument(mock_tracer)

        assert instrumentor._tracer is None
        assert instrumentor._openai_module is None

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_tracer: Mock) -> None:
        """Test error handling during async API call."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        async def original_func(*args: Any, **kwargs: Any) -> None:
            raise MockAPIError("Network error", status_code=500)

        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=True
        )

        with pytest.raises(MockAPIError):
            await wrapper(Mock(), model="gpt-4", messages=[])

        span = mock_tracer.start_span.return_value
        span.set_status.assert_called_with(SpanStatus.ERROR, "Network error")
        span.end.assert_called_once()

    def test_embeddings_with_list_input(self, mock_tracer: Mock) -> None:
        """Test embeddings with multiple inputs."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        response = MockEmbeddingResponse(
            data=[
                MockEmbedding(embedding=[0.1, 0.2]),
                MockEmbedding(embedding=[0.3, 0.4]),
            ]
        )

        original_func = Mock(return_value=response)
        wrapper = instrumentor._create_embeddings_wrapper(original_func)

        wrapper(Mock(), model="text-embedding-ada-002", input=["text1", "text2"])

        span = mock_tracer.start_span.return_value
        attribute_calls = {
            call[0][0]: call[0][1] for call in span.set_attribute.call_args_list
        }
        assert attribute_calls["embedding.input_count"] == 2
        assert attribute_calls["embedding.count"] == 2

    @pytest.mark.asyncio
    async def test_async_streaming(self, mock_tracer: Mock) -> None:
        """Test async streaming chat completion."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._tracer = mock_tracer

        async def async_chunks():
            yield MockStreamChunk(content="Hello")
            yield MockStreamChunk(content=" world")
            yield MockStreamChunk(finish_reason="stop")

        async def original_func(*args: Any, **kwargs: Any):
            return async_chunks()

        wrapper = instrumentor._create_chat_completions_wrapper(
            original_func, is_async=True
        )

        result = await wrapper(
            Mock(), model="gpt-4", messages=[], stream=True
        )

        assert isinstance(result, TracedAsyncChatCompletionStream)

        async with result as traced_stream:
            consumed = []
            async for chunk in traced_stream:
                consumed.append(chunk)

        assert len(consumed) == 3

    def test_stream_with_exception(self, mock_tracer: Mock) -> None:
        """Test stream handles exceptions during iteration."""
        span = Mock()

        def broken_stream():
            yield MockStreamChunk(content="test")
            raise RuntimeError("Stream broke!")

        traced = TracedChatCompletionStream(
            stream=broken_stream(),
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        with pytest.raises(RuntimeError, match="Stream broke"):
            with traced:
                for _ in traced:
                    pass

        span.set_status.assert_called_with(SpanStatus.ERROR, "Stream broke!")

    @pytest.mark.asyncio
    async def test_async_stream_with_exception(self, mock_tracer: Mock) -> None:
        """Test async stream handles exceptions during iteration."""
        span = Mock()

        async def broken_stream():
            yield MockStreamChunk(content="test")
            raise RuntimeError("Stream broke!")

        traced = TracedAsyncChatCompletionStream(
            stream=broken_stream(),
            span=span,
            tracer=mock_tracer,
            start_time=0,
        )

        with pytest.raises(RuntimeError, match="Stream broke"):
            async with traced:
                async for _ in traced:
                    pass

        span.set_status.assert_called_with(SpanStatus.ERROR, "Stream broke!")
