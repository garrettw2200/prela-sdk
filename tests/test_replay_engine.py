"""Tests for replay execution engine."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from prela.core.replay import ReplaySnapshot
from prela.core.span import Span, SpanStatus, SpanType
from prela.replay.engine import ReplayEngine
from prela.replay.loader import Trace


class TestReplayEngine:
    """Test ReplayEngine class."""

    def create_test_trace(self, with_replay_data: bool = True) -> Trace:
        """Create a test trace for replay testing.

        Args:
            with_replay_data: Whether to add replay snapshots to spans

        Returns:
            Trace object with test data
        """
        # Root span (agent)
        span1 = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test_agent",
            span_type=SpanType.AGENT,
            started_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            status=SpanStatus.SUCCESS,
            _ended=True,
        )

        # Child LLM span
        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            name="test_llm",
            span_type=SpanType.LLM,
            started_at=datetime(2024, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc),
            ended_at=datetime(2024, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc),
            status=SpanStatus.SUCCESS,
            _ended=True,
        )

        # Child tool span
        span3 = Span(
            trace_id="trace-1",
            span_id="span-3",
            parent_span_id="span-1",
            name="test_tool",
            span_type=SpanType.TOOL,
            started_at=datetime(2024, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc),
            ended_at=datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            status=SpanStatus.SUCCESS,
            _ended=True,
        )

        if with_replay_data:
            # Add replay snapshots (using object.__setattr__ for immutable spans)
            object.__setattr__(
                span1,
                "replay_snapshot",
                ReplaySnapshot(agent_memory=["message1"], agent_config={"name": "test"}),
            )

            object.__setattr__(
                span2,
                "replay_snapshot",
                ReplaySnapshot(
                    llm_request={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "temperature": 0.7,
                        "max_tokens": 100,
                    },
                    llm_response={
                        "text": "Hi there!",
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "finish_reason": "stop",
                    },
                    model_info={"model": "gpt-4-0613"},
                ),
            )

            object.__setattr__(
                span3,
                "replay_snapshot",
                ReplaySnapshot(
                    tool_name="calculator",
                    tool_input={"operation": "add", "a": 1, "b": 2},
                    tool_output="3",
                ),
            )

        return Trace("trace-1", [span1, span2, span3])

    def test_init_requires_replay_data(self):
        """ReplayEngine requires traces with replay data."""
        trace = self.create_test_trace(with_replay_data=False)

        with pytest.raises(ValueError, match="does not contain replay data"):
            ReplayEngine(trace)

    def test_init_warns_incomplete_data(self, caplog):
        """ReplayEngine warns about incomplete replay data."""
        trace = self.create_test_trace(with_replay_data=True)
        # Remove replay snapshot from one span
        trace.spans[1].replay_snapshot = None

        engine = ReplayEngine(trace)
        assert "incomplete replay data" in caplog.text

    def test_replay_exact_basic(self):
        """replay_exact executes trace using captured data."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_exact()

        assert result.trace_id == "trace-1"
        assert len(result.spans) == 3
        assert result.success is True
        assert len(result.errors) == 0

    def test_replay_exact_llm_span(self):
        """replay_exact extracts LLM response correctly."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_exact()

        # Find LLM span
        llm_span = next(s for s in result.spans if s.span_type == "llm")
        assert llm_span.output == "Hi there!"
        assert llm_span.tokens_used == 15  # 10 + 5
        assert llm_span.cost_usd > 0

    def test_replay_exact_tool_span(self):
        """replay_exact extracts tool output correctly."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_exact()

        # Find tool span
        tool_span = next(s for s in result.spans if s.span_type == "tool")
        assert tool_span.output == "3"
        assert tool_span.tokens_used == 0
        assert tool_span.cost_usd == 0.0

    def test_replay_exact_aggregates_metrics(self):
        """replay_exact aggregates duration, tokens, and cost."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_exact()

        assert result.total_duration_ms > 0
        assert result.total_tokens == 15  # From LLM span
        assert result.total_cost_usd > 0

    def test_replay_exact_extracts_final_output(self):
        """replay_exact extracts final output from root span."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_exact()

        # Final output comes from root span (agent)
        assert result.final_output == {"name": "test"}

    def test_replay_exact_missing_snapshot(self):
        """replay_exact handles missing replay snapshot gracefully."""
        trace = self.create_test_trace(with_replay_data=True)
        trace.spans[1].replay_snapshot = None  # Remove snapshot from one span

        engine = ReplayEngine(trace)
        result = engine.replay_exact()

        # Should complete but with error
        assert result.success is False
        assert len(result.errors) == 1
        assert "No replay data available" in result.errors[0]

    def test_replay_with_modifications_model(self):
        """replay_with_modifications tracks model modification."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_with_modifications(model="gpt-4o")

        assert result.modifications_applied["model"] == "gpt-4o"
        assert result.modified_span_count >= 1  # At least LLM span should be modified

    def test_replay_with_modifications_temperature(self):
        """replay_with_modifications tracks temperature modification."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_with_modifications(temperature=0.5)

        assert result.modifications_applied["temperature"] == 0.5

    def test_replay_with_modifications_system_prompt(self):
        """replay_with_modifications tracks system prompt modification."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_with_modifications(system_prompt="New prompt")

        assert result.modifications_applied["system_prompt"] == "New prompt"

    def test_replay_with_modifications_max_tokens(self):
        """replay_with_modifications tracks max_tokens modification."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_with_modifications(max_tokens=200)

        assert result.modifications_applied["max_tokens"] == 200

    def test_replay_with_modifications_mock_tool(self):
        """replay_with_modifications can mock tool responses."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        result = engine.replay_with_modifications(
            mock_tool_responses={"calculator": "42"}
        )

        # Find tool span
        tool_span = next(s for s in result.spans if s.span_type == "tool")
        assert tool_span.output == "42"
        assert tool_span.was_modified is True

    def test_replay_with_modifications_mock_retrieval(self):
        """replay_with_modifications can mock retrieval results."""
        # Create trace with retrieval span
        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="test_retrieval",
            span_type=SpanType.RETRIEVAL,
        )
        span.replay_snapshot = ReplaySnapshot(
            retrieval_query="test query",
            retrieved_documents=[{"text": "doc1"}],
        )
        trace = Trace("trace-1", [span])

        engine = ReplayEngine(trace)

        mock_docs = [{"text": "mocked doc"}]
        result = engine.replay_with_modifications(mock_retrieval_results=mock_docs)

        # Find retrieval span
        retrieval_span = result.spans[0]
        assert retrieval_span.output == mock_docs
        assert retrieval_span.was_modified is True

    def test_replay_with_modifications_unmodified_spans_use_cache(self):
        """Unmodified spans use captured data."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        # Modify only temperature, not tool responses
        result = engine.replay_with_modifications(temperature=0.5)

        # Tool span should not be modified (uses cached data)
        tool_span = next(s for s in result.spans if s.span_type == "tool")
        assert tool_span.was_modified is False
        assert tool_span.output == "3"  # Original output

    def test_span_needs_modification_llm(self):
        """_span_needs_modification detects LLM parameter changes."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        llm_span = next(s for s in trace.spans if s.span_type == SpanType.LLM)

        # Should need modification with model change
        assert (
            engine._span_needs_modification(
                llm_span,
                model="gpt-4o",
                temperature=None,
                system_prompt=None,
                max_tokens=None,
                mock_tool_responses=None,
                mock_retrieval_results=None,
                enable_tool_execution=False,
                enable_retrieval_execution=False,
                retrieval_query_override=None,
            )
            is True
        )

        # Should not need modification without changes
        assert (
            engine._span_needs_modification(
                llm_span,
                model=None,
                temperature=None,
                system_prompt=None,
                max_tokens=None,
                mock_tool_responses=None,
                mock_retrieval_results=None,
                enable_tool_execution=False,
                enable_retrieval_execution=False,
                retrieval_query_override=None,
            )
            is False
        )

    def test_span_needs_modification_tool(self):
        """_span_needs_modification detects tool mocks."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        tool_span = next(s for s in trace.spans if s.span_type == SpanType.TOOL)

        # Should need modification if tool is in mock dict
        assert (
            engine._span_needs_modification(
                tool_span,
                model=None,
                temperature=None,
                system_prompt=None,
                max_tokens=None,
                mock_tool_responses={"calculator": "42"},
                mock_retrieval_results=None,
                enable_tool_execution=False,
                enable_retrieval_execution=False,
                retrieval_query_override=None,
            )
            is True
        )

        # Should not need modification if tool is not in mock dict
        assert (
            engine._span_needs_modification(
                tool_span,
                model=None,
                temperature=None,
                system_prompt=None,
                max_tokens=None,
                mock_tool_responses={"other_tool": "42"},
                mock_retrieval_results=None,
                enable_tool_execution=False,
                enable_retrieval_execution=False,
                retrieval_query_override=None,
            )
            is False
        )

    def test_call_llm_api_requires_model(self):
        """_call_llm_api requires model parameter."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        with pytest.raises(ValueError, match="Model is required"):
            engine._call_llm_api({})

    def test_estimate_cost_gpt4(self):
        """_estimate_cost calculates GPT-4 pricing."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        cost = engine._estimate_cost("gpt-4-0613", 1_000_000)
        assert cost == 30.0  # $30 per 1M tokens

    def test_estimate_cost_unknown_model(self):
        """_estimate_cost uses default for unknown models."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        cost = engine._estimate_cost("unknown-model", 1_000_000)
        assert cost == 5.0  # Default $5 per 1M tokens

    def test_estimate_cost_none_model(self):
        """_estimate_cost returns 0 for None model."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        cost = engine._estimate_cost(None, 1_000_000)
        assert cost == 0.0

    def test_call_openai_api_streaming(self):
        """_call_openai_api supports streaming."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        mock_chunk2.usage = None

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock(delta=MagicMock(content="!"))]
        mock_chunk3.usage = MagicMock(prompt_tokens=10, completion_tokens=3)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(
            [mock_chunk1, mock_chunk2, mock_chunk3]
        )

        # Mock the openai module itself
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # Track streaming chunks
        streamed_chunks = []

        def callback(chunk: str) -> None:
            streamed_chunks.append(chunk)

        # Call with streaming using mock
        with patch.dict("sys.modules", {"openai": mock_openai}):
            request = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            response_text, tokens, cost = engine._call_openai_api(
                request, stream=True, stream_callback=callback
            )

        # Verify streaming
        assert response_text == "Hello world!"
        assert tokens == 13
        assert cost > 0
        assert streamed_chunks == ["Hello", " world", "!"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

    def test_call_anthropic_api_streaming(self):
        """_call_anthropic_api supports streaming."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        # Mock streaming response
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["Hello", " world", "!"])

        mock_final_message = MagicMock()
        mock_final_message.usage = MagicMock(input_tokens=10, output_tokens=3)
        mock_stream.get_final_message.return_value = mock_final_message

        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        # Mock the anthropic module itself
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Track streaming chunks
        streamed_chunks = []

        def callback(chunk: str) -> None:
            streamed_chunks.append(chunk)

        # Call with streaming using mock
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            request = {
                "model": "claude-sonnet-4",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            response_text, tokens, cost = engine._call_anthropic_api(
                request, stream=True, stream_callback=callback
            )

        # Verify streaming
        assert response_text == "Hello world!"
        assert tokens == 13
        assert cost > 0
        assert streamed_chunks == ["Hello", " world", "!"]

    def test_replay_with_modifications_streaming(self):
        """replay_with_modifications supports streaming."""
        trace = self.create_test_trace(with_replay_data=True)
        engine = ReplayEngine(trace)

        # Track streaming chunks
        streamed_chunks = []

        def callback(chunk: str) -> None:
            streamed_chunks.append(chunk)

        # Mock API call to test callback propagation
        def mock_call_llm(request, stream=False, stream_callback=None):
            if stream and stream_callback:
                # Simulate streaming
                for chunk in ["Test", " streaming", " output"]:
                    stream_callback(chunk)
            return "Test streaming output", 100, 0.01

        engine._call_llm_api = mock_call_llm

        # Replay with streaming
        result = engine.replay_with_modifications(
            model="gpt-4o", stream=True, stream_callback=callback
        )

        # Verify callback was called
        assert streamed_chunks == ["Test", " streaming", " output"]

    def test_retry_logic_retryable_error(self):
        """Test that retryable errors are retried with exponential backoff."""
        from prela.replay.engine import _is_retryable_error, with_retry

        # Test retryable error detection
        assert _is_retryable_error(Exception("rate limit exceeded"))
        assert _is_retryable_error(Exception("429 Too Many Requests"))
        assert _is_retryable_error(Exception("503 Service Unavailable"))
        assert _is_retryable_error(Exception("connection timeout"))
        assert _is_retryable_error(Exception("temporarily unavailable"))

        # Test non-retryable error
        assert not _is_retryable_error(Exception("invalid API key"))
        assert not _is_retryable_error(ValueError("bad request"))

        # Test retry decorator with mock
        call_count = 0
        max_retries = 3

        @with_retry(max_retries=max_retries, initial_delay=0.01, max_delay=0.05)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third attempt

    def test_retry_logic_non_retryable_error(self):
        """Test that non-retryable errors fail immediately without retries."""
        from prela.replay.engine import with_retry

        call_count = 0

        @with_retry(max_retries=3, initial_delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid API key")

        with pytest.raises(ValueError, match="invalid API key"):
            failing_function()

        # Should only be called once (no retries for non-retryable errors)
        assert call_count == 1

    def test_retry_logic_max_retries_exceeded(self):
        """Test that retryable errors eventually fail after max retries."""
        from prela.replay.engine import with_retry

        call_count = 0
        max_retries = 2

        @with_retry(max_retries=max_retries, initial_delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise Exception("503 Service Unavailable")

        with pytest.raises(Exception, match="503 Service Unavailable"):
            always_failing()

        # Should be called max_retries + 1 times (initial + retries)
        assert call_count == max_retries + 1

    def test_retry_count_tracking(self):
        """Test that retry counts are tracked and included in ReplayedSpan."""
        trace = self.create_test_trace()
        engine = ReplayEngine(trace, max_retries=2, retry_initial_delay=0.01)

        # Mock API call that fails once then succeeds
        call_count = 0

        def mock_api_call_with_retry(request, stream=False, stream_callback=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 rate limit")
            return "Success after retry", 50, 0.005

        engine._call_openai_api_impl = mock_api_call_with_retry

        # Replay with modification (triggers API call)
        result = engine.replay_with_modifications(model="gpt-4o")

        # Verify retry count is recorded
        llm_span = next(s for s in result.spans if s.span_type == "llm")
        assert llm_span.retry_count == 1  # One retry
        assert llm_span.output == "Success after retry"
        assert llm_span.error is None

    def test_replay_engine_retry_config(self):
        """Test that ReplayEngine accepts retry configuration."""
        trace = self.create_test_trace()

        # Test with custom retry config
        engine = ReplayEngine(
            trace,
            max_retries=5,
            retry_initial_delay=2.0,
            retry_max_delay=120.0,
            retry_exponential_base=3.0,
        )

        assert engine.max_retries == 5
        assert engine.retry_initial_delay == 2.0
        assert engine.retry_max_delay == 120.0
        assert engine.retry_exponential_base == 3.0

    def test_retry_exponential_backoff(self):
        """Test that retry delays follow exponential backoff pattern."""
        from prela.replay.engine import with_retry
        import time

        delays = []

        @with_retry(max_retries=3, initial_delay=0.05, max_delay=1.0, exponential_base=2.0)
        def measure_delays():
            if len(delays) < 3:
                start = time.time()
                raise Exception("retry me")
            return "success"

        # Record retry timing
        call_times = []

        @with_retry(max_retries=2, initial_delay=0.01, exponential_base=2.0)
        def track_timing():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("rate limit")
            return "done"

        track_timing()

        # Verify exponential pattern (roughly 0.01s, 0.02s delays)
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Delay2 should be roughly 2x delay1 (exponential_base=2.0)
        # Allow tolerance for timing variance
        assert 0.008 < delay1 < 0.015  # ~0.01s
        assert 0.015 < delay2 < 0.030  # ~0.02s
