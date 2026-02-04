"""Tests for tool and retrieval re-execution in replay comparison."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from prela.core.replay import ReplaySnapshot
from prela.core.span import Span, SpanStatus, SpanType
from prela.replay.engine import ReplayEngine
from prela.replay.loader import Trace


def create_tool_trace(tool_name: str, tool_input: dict, tool_output: dict | None = None) -> Trace:
    """Create a test trace with a tool span.

    Args:
        tool_name: Name of the tool
        tool_input: Input data for the tool
        tool_output: Output data from the tool (default: None)

    Returns:
        Trace object with tool span
    """
    span = Span(
        trace_id="trace-1",
        span_id="span-1",
        name=tool_name,
        span_type=SpanType.TOOL,
        started_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
        _ended=True,
    )

    object.__setattr__(
        span,
        "replay_snapshot",
        ReplaySnapshot(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
        ),
    )

    return Trace(
        trace_id="trace-1",
        spans=[span],
    )


def create_retrieval_trace(query: str, documents: list[dict] | None = None) -> Trace:
    """Create a test trace with a retrieval span.

    Args:
        query: Retrieval query
        documents: Retrieved documents (default: empty list)

    Returns:
        Trace object with retrieval span
    """
    span = Span(
        trace_id="trace-1",
        span_id="span-1",
        name="vector_search",
        span_type=SpanType.RETRIEVAL,
        started_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
        _ended=True,
    )

    object.__setattr__(
        span,
        "replay_snapshot",
        ReplaySnapshot(
            retrieval_query=query,
            retrieved_documents=documents or [],
        ),
    )

    return Trace(
        trace_id="trace-1",
        spans=[span],
    )


class TestToolReExecution:
    """Test tool re-execution during replay."""

    def test_tool_execution_with_allowlist(self):
        """Tool executes when in allowlist."""
        # Create mock tool
        def mock_calculator(input_data):
            return {"result": input_data["a"] + input_data["b"]}

        tool_registry = {"calculator": mock_calculator}

        # Create trace
        trace = create_tool_trace("calculator", {"a": 5, "b": 3}, {"result": 8})
        engine = ReplayEngine(trace)

        # Replay with tool execution enabled
        result = engine.replay_with_modifications(
            enable_tool_execution=True,
            tool_execution_allowlist=["calculator"],
            tool_registry=tool_registry,
        )

        # Verify tool was executed
        assert len(result.spans) == 1
        tool_span = result.spans[0]
        assert tool_span.span_type == "tool"
        assert tool_span.output == {"result": 8}
        assert tool_span.was_modified is True
        assert "tool_executed" in tool_span.modification_details

    def test_tool_execution_blocked_by_blocklist(self):
        """Tool execution fails when in blocklist."""
        # Create mock tool
        def mock_dangerous_tool(input_data):
            return {"deleted": True}

        tool_registry = {"dangerous_tool": mock_dangerous_tool}

        # Create trace
        trace = create_tool_trace("dangerous_tool", {"action": "delete_all"}, None)
        engine = ReplayEngine(trace)

        # Replay with tool execution enabled but tool blocked
        result = engine.replay_with_modifications(
            enable_tool_execution=True,
            tool_execution_blocklist=["dangerous_tool"],
            tool_registry=tool_registry,
        )

        # Verify tool was blocked (error captured)
        assert len(result.spans) == 1
        tool_span = result.spans[0]
        assert tool_span.span_type == "tool"
        assert tool_span.error is not None
        assert "blocked" in tool_span.error.lower()

    def test_tool_execution_not_in_allowlist(self):
        """Tool execution fails when not in allowlist."""
        # Create mock tool
        def mock_tool(input_data):
            return {"result": "ok"}

        tool_registry = {"tool_a": mock_tool, "tool_b": mock_tool}

        # Create trace
        trace = create_tool_trace("tool_b", {}, None)
        engine = ReplayEngine(trace)

        # Replay with tool execution enabled but only tool_a allowed
        result = engine.replay_with_modifications(
            enable_tool_execution=True,
            tool_execution_allowlist=["tool_a"],
            tool_registry=tool_registry,
        )

        # Verify tool was blocked (not in allowlist)
        assert len(result.spans) == 1
        tool_span = result.spans[0]
        assert tool_span.span_type == "tool"
        assert tool_span.error is not None
        assert "not in allowlist" in tool_span.error.lower()

    def test_tool_execution_priority_mock_over_execution(self):
        """Mock responses have priority over tool execution."""
        # Create mock tool
        def mock_tool(input_data):
            return {"result": "executed"}

        tool_registry = {"calculator": mock_tool}

        # Create trace
        trace = create_tool_trace("calculator", {"a": 5, "b": 3}, {"result": 8})
        engine = ReplayEngine(trace)

        # Replay with both mock and execution enabled
        result = engine.replay_with_modifications(
            mock_tool_responses={"calculator": {"result": "mocked"}},
            enable_tool_execution=True,
            tool_execution_allowlist=["calculator"],
            tool_registry=tool_registry,
        )

        # Verify mock was used (priority over execution)
        assert len(result.spans) == 1
        tool_span = result.spans[0]
        assert tool_span.output == {"result": "mocked"}
        assert "mocked_output" in tool_span.modification_details
        assert "tool_executed" not in tool_span.modification_details

    def test_tool_execution_uses_cached_when_disabled(self):
        """Uses cached output when tool execution disabled."""
        # Create trace
        trace = create_tool_trace("calculator", {"a": 5, "b": 3}, {"result": 8})
        engine = ReplayEngine(trace)

        # Replay with tool execution disabled (default)
        result = engine.replay_with_modifications()

        # Verify cached output was used
        assert len(result.spans) == 1
        tool_span = result.spans[0]
        assert tool_span.output == {"result": 8}
        assert tool_span.was_modified is False


class TestRetrievalReExecution:
    """Test retrieval re-execution during replay."""

    def test_retrieval_execution_chromadb(self):
        """Retrieval executes with ChromaDB client."""

        # Create mock ChromaDB client
        class MockChromaClient:
            def __init__(self):
                self.__class__.__name__ = "ChromaClient"

            def query(self, query_texts, n_results):
                return {
                    "documents": [["LangChain is a framework", "LlamaIndex is a framework"]],
                    "distances": [[0.1, 0.3]],
                }

        client = MockChromaClient()

        # Create trace
        trace = create_retrieval_trace("What is LangChain?", [])
        engine = ReplayEngine(trace)

        # Replay with retrieval execution enabled
        result = engine.replay_with_modifications(
            enable_retrieval_execution=True,
            retrieval_client=client,
        )

        # Verify retrieval was executed
        assert len(result.spans) == 1
        retrieval_span = result.spans[0]
        assert retrieval_span.span_type == "retrieval"
        assert len(retrieval_span.output) == 2
        assert retrieval_span.output[0]["text"] == "LangChain is a framework"
        assert retrieval_span.output[0]["score"] == 0.9  # 1.0 - 0.1
        assert retrieval_span.was_modified is True
        assert "retrieval_executed" in retrieval_span.modification_details

    def test_retrieval_execution_with_query_override(self):
        """Retrieval executes with overridden query."""

        # Create mock ChromaDB client
        class MockChromaClient:
            def __init__(self):
                self.__class__.__name__ = "ChromaClient"

            def query(self, query_texts, n_results):
                # Return different results based on query
                if "LlamaIndex" in query_texts[0]:
                    return {
                        "documents": [["LlamaIndex is a data framework"]],
                        "distances": [[0.05]],
                    }
                return {
                    "documents": [["LangChain is a framework"]],
                    "distances": [[0.1]],
                }

        client = MockChromaClient()

        # Create trace
        trace = create_retrieval_trace("What is LangChain?", [])
        engine = ReplayEngine(trace)

        # Replay with query override
        result = engine.replay_with_modifications(
            enable_retrieval_execution=True,
            retrieval_client=client,
            retrieval_query_override="What is LlamaIndex?",
        )

        # Verify overridden query was used
        assert len(result.spans) == 1
        retrieval_span = result.spans[0]
        assert retrieval_span.span_type == "retrieval"
        assert retrieval_span.input == "What is LlamaIndex?"
        assert "LlamaIndex" in retrieval_span.output[0]["text"]
        assert "retrieval_executed" in retrieval_span.modification_details
        assert "query_overridden" in retrieval_span.modification_details

    def test_retrieval_execution_priority_mock_over_execution(self):
        """Mock results have priority over retrieval execution."""

        # Create mock ChromaDB client
        class MockChromaClient:
            def __init__(self):
                self.__class__.__name__ = "ChromaClient"

            def query(self, query_texts, n_results):
                return {
                    "documents": [["Executed document"]],
                    "distances": [[0.1]],
                }

        client = MockChromaClient()

        # Create trace
        trace = create_retrieval_trace("What is LangChain?", [])
        engine = ReplayEngine(trace)

        # Replay with both mock and execution enabled
        mock_results = [{"text": "Mocked document", "score": 1.0}]
        result = engine.replay_with_modifications(
            mock_retrieval_results=mock_results,
            enable_retrieval_execution=True,
            retrieval_client=client,
        )

        # Verify mock was used (priority over execution)
        assert len(result.spans) == 1
        retrieval_span = result.spans[0]
        assert retrieval_span.output == mock_results
        assert "mocked_documents" in retrieval_span.modification_details
        assert "retrieval_executed" not in retrieval_span.modification_details

    def test_retrieval_execution_uses_cached_when_disabled(self):
        """Uses cached documents when retrieval execution disabled."""
        # Create trace
        cached_docs = [{"text": "Cached document", "score": 0.95}]
        trace = create_retrieval_trace("What is LangChain?", cached_docs)
        engine = ReplayEngine(trace)

        # Replay with retrieval execution disabled (default)
        result = engine.replay_with_modifications()

        # Verify cached documents were used
        assert len(result.spans) == 1
        retrieval_span = result.spans[0]
        assert retrieval_span.output == cached_docs
        assert retrieval_span.was_modified is False

    def test_retrieval_execution_failure_handling(self):
        """Handles retrieval execution failures gracefully."""

        # Create mock ChromaDB client that raises exception
        class MockChromaClient:
            def __init__(self):
                self.__class__.__name__ = "ChromaClient"

            def query(self, query_texts, n_results):
                raise Exception("Connection timeout")

        client = MockChromaClient()

        # Create trace
        trace = create_retrieval_trace("What is LangChain?", [])
        engine = ReplayEngine(trace)

        # Replay with retrieval execution enabled
        result = engine.replay_with_modifications(
            enable_retrieval_execution=True,
            retrieval_client=client,
        )

        # Verify error was captured
        assert len(result.spans) == 1
        retrieval_span = result.spans[0]
        assert retrieval_span.error is not None
        assert "Connection timeout" in retrieval_span.error


class TestClientDetection:
    """Test vector DB client detection."""

    def test_detect_chromadb_client(self):
        """Detects ChromaDB client correctly."""

        class MockChromaClient:
            def __init__(self):
                self.__class__.__name__ = "ChromaClient"
                self.__class__.__module__ = "chromadb.api"

        # Create trace
        trace = create_tool_trace("test_tool", {}, {})
        engine = ReplayEngine(trace)

        client = MockChromaClient()
        client_type = engine._detect_retrieval_client(client)

        assert client_type == "chromadb"

    def test_detect_unknown_client(self):
        """Returns 'unknown' for unsupported client."""

        class MockUnknownClient:
            def __init__(self):
                self.__class__.__name__ = "UnknownClient"
                self.__class__.__module__ = "unknown_db"

        # Create trace
        trace = create_tool_trace("test_tool", {}, {})
        engine = ReplayEngine(trace)

        client = MockUnknownClient()
        client_type = engine._detect_retrieval_client(client)

        assert client_type == "unknown"
