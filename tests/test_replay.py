"""Tests for replay capture functionality."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from prela.core.replay import (
    REPLAY_SIZE_WARNING_THRESHOLD,
    ReplayCapture,
    ReplaySnapshot,
    estimate_replay_storage,
    serialize_replay_data,
)
from prela.core.span import Span, SpanType


class TestReplaySnapshot:
    """Test ReplaySnapshot data structure."""

    def test_empty_snapshot(self):
        """Empty snapshot serializes to dict with safe defaults."""
        snapshot = ReplaySnapshot()
        # has_side_effects defaults to True (safe default for tools)
        assert snapshot.to_dict() == {"has_side_effects": True}

    def test_llm_snapshot(self):
        """LLM snapshot captures request/response."""
        snapshot = ReplaySnapshot(
            llm_request={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            llm_response={"text": "Hello!", "finish_reason": "stop"},
            model_info={"model": "gpt-4-0613"},
        )

        data = snapshot.to_dict()
        assert data["llm_request"]["model"] == "gpt-4"
        assert data["llm_response"]["text"] == "Hello!"
        assert data["model_info"]["model"] == "gpt-4-0613"

    def test_tool_snapshot(self):
        """Tool snapshot captures input/output."""
        snapshot = ReplaySnapshot(
            tool_name="calculator",
            tool_description="Performs calculations",
            tool_input={"operation": "add", "a": 1, "b": 2},
            tool_output="3",
            has_side_effects=False,
        )

        data = snapshot.to_dict()
        assert data["tool_name"] == "calculator"
        assert data["tool_input"]["operation"] == "add"
        assert data["tool_output"] == "3"
        assert data["has_side_effects"] is False

    def test_retrieval_snapshot(self):
        """Retrieval snapshot captures documents and scores."""
        snapshot = ReplaySnapshot(
            retrieval_query="What is AI?",
            retrieved_documents=[
                {"text": "AI is artificial intelligence", "id": "doc1"},
                {"text": "AI is machine learning", "id": "doc2"},
            ],
            retrieval_scores=[0.95, 0.87],
            retrieval_metadata={"index": "knowledge_base", "k": 2},
        )

        data = snapshot.to_dict()
        assert data["retrieval_query"] == "What is AI?"
        assert len(data["retrieved_documents"]) == 2
        assert data["retrieval_scores"] == [0.95, 0.87]

    def test_agent_snapshot(self):
        """Agent snapshot captures context and configuration."""
        snapshot = ReplaySnapshot(
            system_prompt="You are a helpful assistant",
            available_tools=[
                {"name": "search", "description": "Search the web"},
                {"name": "calculate", "description": "Perform calculations"},
            ],
            agent_memory={"conversation_history": ["Hello", "Hi there!"]},
            agent_config={"temperature": 0.7, "max_iterations": 3},
        )

        data = snapshot.to_dict()
        assert data["system_prompt"] == "You are a helpful assistant"
        assert len(data["available_tools"]) == 2
        assert "conversation_history" in data["agent_memory"]

    def test_streaming_chunks(self):
        """Streaming chunks are captured."""
        snapshot = ReplaySnapshot(
            llm_streaming_chunks=[
                {"delta": {"content": "Hello"}},
                {"delta": {"content": " world"}},
                {"delta": {"content": "!"}},
            ]
        )

        data = snapshot.to_dict()
        assert len(data["llm_streaming_chunks"]) == 3
        assert data["llm_streaming_chunks"][0]["delta"]["content"] == "Hello"

    def test_datetime_serialization(self):
        """Datetime fields are serialized to ISO format."""
        now = datetime.now(timezone.utc)
        snapshot = ReplaySnapshot(request_timestamp=now)

        data = snapshot.to_dict()
        assert "request_timestamp" in data
        assert isinstance(data["request_timestamp"], str)
        assert data["request_timestamp"] == now.isoformat()

    def test_roundtrip_serialization(self):
        """Snapshot can be serialized and deserialized."""
        now = datetime.now(timezone.utc)
        original = ReplaySnapshot(
            llm_request={"model": "gpt-4"},
            tool_name="search",
            request_timestamp=now,
        )

        data = original.to_dict()
        restored = ReplaySnapshot.from_dict(data)

        assert restored.llm_request == original.llm_request
        assert restored.tool_name == original.tool_name
        assert restored.request_timestamp == original.request_timestamp

    def test_estimate_size_bytes(self):
        """Can estimate size of snapshot."""
        snapshot = ReplaySnapshot(
            llm_request={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            llm_response={"text": "Hi there!", "finish_reason": "stop"},
        )

        size = snapshot.estimate_size_bytes()
        assert size > 0
        assert size < 1000  # Should be small for this simple snapshot

    def test_size_warning_threshold(self, caplog):
        """Warning logged when snapshot exceeds 100 KB."""
        # Create a large snapshot (>100 KB)
        large_text = "x" * (REPLAY_SIZE_WARNING_THRESHOLD + 1000)
        snapshot = ReplaySnapshot(
            llm_response={"text": large_text}
        )

        size = snapshot.estimate_size_bytes()
        assert size > REPLAY_SIZE_WARNING_THRESHOLD

        # Check that warning was logged
        assert "exceeds recommended threshold" in caplog.text


class TestReplayCapture:
    """Test ReplayCapture builder pattern."""

    def test_llm_capture(self):
        """Can capture LLM request and response."""
        capture = ReplayCapture()

        capture.set_llm_request(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

        capture.set_llm_response(
            text="Hi there!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
        )

        snapshot = capture.build()
        assert snapshot.llm_request["model"] == "gpt-4"
        assert snapshot.llm_request["temperature"] == 0.7
        assert snapshot.llm_response["text"] == "Hi there!"
        assert snapshot.llm_response["prompt_tokens"] == 10

    def test_llm_request_kwargs(self):
        """Additional kwargs captured in request."""
        capture = ReplayCapture()

        capture.set_llm_request(
            model="gpt-4",
            messages=[],
            top_p=0.9,
            frequency_penalty=0.5,
        )

        snapshot = capture.build()
        assert snapshot.llm_request["top_p"] == 0.9
        assert snapshot.llm_request["frequency_penalty"] == 0.5

    def test_streaming_chunks(self):
        """Can capture streaming chunks."""
        capture = ReplayCapture()

        capture.add_streaming_chunk({"delta": {"content": "Hello"}})
        capture.add_streaming_chunk({"delta": {"content": " world"}})

        snapshot = capture.build()
        assert len(snapshot.llm_streaming_chunks) == 2
        assert snapshot.llm_streaming_chunks[0]["delta"]["content"] == "Hello"

    def test_model_info(self):
        """Can capture model metadata."""
        capture = ReplayCapture()

        capture.set_model_info(
            model="gpt-4-0613",
            created=1234567890,
            id="chatcmpl-123",
        )

        snapshot = capture.build()
        assert snapshot.model_info["model"] == "gpt-4-0613"
        assert snapshot.model_info["created"] == 1234567890

    def test_tool_call(self):
        """Can capture tool call details."""
        capture = ReplayCapture()

        capture.set_tool_call(
            name="calculator",
            description="Performs math",
            input_args={"operation": "add", "a": 1, "b": 2},
            output="3",
            has_side_effects=False,
        )

        snapshot = capture.build()
        assert snapshot.tool_name == "calculator"
        assert snapshot.tool_input["operation"] == "add"
        assert snapshot.tool_output == "3"
        assert snapshot.has_side_effects is False

    def test_tool_default_side_effects(self):
        """Tools default to has_side_effects=True."""
        capture = ReplayCapture()

        capture.set_tool_call(
            name="send_email",
            input_args={"to": "user@example.com"},
        )

        snapshot = capture.build()
        assert snapshot.has_side_effects is True  # Safe default

    def test_retrieval(self):
        """Can capture retrieval operation."""
        capture = ReplayCapture()

        capture.set_retrieval(
            query="What is AI?",
            documents=[{"text": "AI is...", "id": "doc1"}],
            scores=[0.95],
            metadata={"index": "kb"},
        )

        snapshot = capture.build()
        assert snapshot.retrieval_query == "What is AI?"
        assert len(snapshot.retrieved_documents) == 1
        assert snapshot.retrieval_scores == [0.95]

    def test_agent_context(self):
        """Can capture agent context."""
        capture = ReplayCapture()

        capture.set_agent_context(
            system_prompt="You are helpful",
            available_tools=[{"name": "search"}],
            memory={"history": []},
            config={"temperature": 0.7},
        )

        snapshot = capture.build()
        assert snapshot.system_prompt == "You are helpful"
        assert len(snapshot.available_tools) == 1
        assert "history" in snapshot.agent_memory
        assert snapshot.agent_config["temperature"] == 0.7

    def test_agent_context_partial(self):
        """Can set agent context fields individually."""
        capture = ReplayCapture()

        # Only set system prompt
        capture.set_agent_context(system_prompt="Be helpful")

        snapshot = capture.build()
        assert snapshot.system_prompt == "Be helpful"
        assert snapshot.available_tools is None
        assert snapshot.agent_memory is None

    def test_request_timestamp_auto(self):
        """Request timestamp automatically set on LLM request."""
        capture = ReplayCapture()

        before = datetime.now(timezone.utc)
        capture.set_llm_request(model="gpt-4", messages=[])
        after = datetime.now(timezone.utc)

        snapshot = capture.build()
        assert snapshot.request_timestamp is not None
        assert before <= snapshot.request_timestamp <= after


class TestEstimateReplayStorage:
    """Test storage estimation functions."""

    def test_estimate_span_with_replay(self):
        """Can estimate storage for span with replay data."""
        span = Span(
            name="test_span",
            span_type=SpanType.LLM,
        )
        span.set_attribute("llm.model", "gpt-4")
        span.end()

        # Attach replay snapshot
        replay_snapshot = ReplaySnapshot(
            llm_request={"model": "gpt-4"},
            llm_response={"text": "Hello!"},
        )
        object.__setattr__(span, "replay_snapshot", replay_snapshot)

        size = estimate_replay_storage(span)
        assert size > 0

        # Size with replay should be larger than without
        span_dict = span.to_dict()
        span_dict.pop("replay_snapshot", None)
        span_no_replay = Span.from_dict(span_dict)
        size_no_replay = estimate_replay_storage(span_no_replay)

        assert size > size_no_replay

    def test_estimate_with_external_snapshot(self):
        """Can estimate with snapshot passed separately."""
        span = Span(name="test", span_type=SpanType.LLM)
        span.end()

        snapshot = ReplaySnapshot(llm_request={"model": "gpt-4"})
        size = estimate_replay_storage(span, replay_snapshot=snapshot)

        assert size > 0


class TestSerializeReplayData:
    """Test serialization helper."""

    def test_basic_types(self):
        """Basic types pass through."""
        assert serialize_replay_data("hello") == "hello"
        assert serialize_replay_data(123) == 123
        assert serialize_replay_data(45.67) == 45.67
        assert serialize_replay_data(True) is True
        assert serialize_replay_data(None) is None

    def test_datetime(self):
        """Datetime converted to ISO format."""
        now = datetime.now(timezone.utc)
        result = serialize_replay_data(now)
        assert isinstance(result, str)
        assert result == now.isoformat()

    def test_list(self):
        """Lists recursively serialized."""
        data = [1, "two", datetime.now(timezone.utc)]
        result = serialize_replay_data(data)
        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == "two"
        assert isinstance(result[2], str)  # datetime converted

    def test_dict(self):
        """Dicts recursively serialized."""
        now = datetime.now(timezone.utc)
        data = {"a": 1, "b": now, "c": {"nested": True}}
        result = serialize_replay_data(data)
        assert isinstance(result, dict)
        assert result["a"] == 1
        assert isinstance(result["b"], str)  # datetime converted
        assert result["c"]["nested"] is True

    def test_tuple(self):
        """Tuples converted to lists."""
        data = (1, 2, 3)
        result = serialize_replay_data(data)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_custom_object(self):
        """Custom objects converted to strings."""
        class CustomObject:
            def __str__(self):
                return "custom"

        obj = CustomObject()
        result = serialize_replay_data(obj)
        assert result == "custom"


class TestSpanReplayIntegration:
    """Test replay snapshot integration with Span class."""

    def test_span_with_replay_snapshot(self):
        """Span can have replay snapshot attached."""
        replay_snapshot = ReplaySnapshot(
            llm_request={"model": "gpt-4"},
            llm_response={"text": "Hello!"},
        )

        span = Span(
            name="test_span",
            span_type=SpanType.LLM,
            replay_snapshot=replay_snapshot,
        )

        assert span.replay_snapshot is not None
        assert span.replay_snapshot.llm_request["model"] == "gpt-4"

    def test_span_to_dict_includes_replay(self):
        """Span.to_dict() includes replay snapshot."""
        replay_snapshot = ReplaySnapshot(
            llm_request={"model": "gpt-4"},
        )

        span = Span(
            name="test_span",
            span_type=SpanType.LLM,
            replay_snapshot=replay_snapshot,
        )
        span.end()

        data = span.to_dict()
        assert "replay_snapshot" in data
        assert data["replay_snapshot"]["llm_request"]["model"] == "gpt-4"

    def test_span_from_dict_with_replay(self):
        """Span.from_dict() deserializes replay snapshot."""
        replay_snapshot = ReplaySnapshot(
            llm_request={"model": "gpt-4"},
            tool_name="calculator",
        )

        original_span = Span(
            name="test_span",
            span_type=SpanType.LLM,
            replay_snapshot=replay_snapshot,
        )
        original_span.end()

        # Roundtrip
        data = original_span.to_dict()
        restored_span = Span.from_dict(data)

        assert restored_span.replay_snapshot is not None
        assert restored_span.replay_snapshot.llm_request["model"] == "gpt-4"
        assert restored_span.replay_snapshot.tool_name == "calculator"

    def test_span_without_replay(self):
        """Span without replay snapshot works normally."""
        span = Span(name="test_span", span_type=SpanType.LLM)
        span.end()

        data = span.to_dict()
        assert "replay_snapshot" not in data

        # Roundtrip works
        restored_span = Span.from_dict(data)
        assert restored_span.replay_snapshot is None
