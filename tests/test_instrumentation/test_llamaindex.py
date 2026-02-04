"""Tests for LlamaIndex instrumentation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from prela.core.span import SpanStatus, SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter


# Mock LlamaIndex modules
@pytest.fixture(autouse=True)
def mock_llamaindex():
    """Mock llama_index modules to avoid requiring installation."""
    # Create mock CBEventType
    mock_event_type = MagicMock()
    mock_event_type.LLM = "LLM"
    mock_event_type.EMBEDDING = "EMBEDDING"
    mock_event_type.RETRIEVE = "RETRIEVE"
    mock_event_type.QUERY = "QUERY"
    mock_event_type.SYNTHESIZE = "SYNTHESIZE"
    mock_event_type.TREE = "TREE"
    mock_event_type.SUB_QUESTION = "SUB_QUESTION"
    mock_event_type.CHUNKING = "CHUNKING"
    mock_event_type.NODE_PARSING = "NODE_PARSING"
    mock_event_type.TEMPLATING = "TEMPLATING"

    # Create mock Settings and CallbackManager
    mock_callback_manager = MagicMock()
    mock_callback_manager.add_handler = MagicMock()
    mock_callback_manager.remove_handler = MagicMock()

    mock_settings = MagicMock()
    mock_settings.callback_manager = mock_callback_manager

    # Create mock modules
    mock_callbacks_schema = MagicMock()
    mock_callbacks_schema.CBEventType = mock_event_type

    mock_callbacks_module = MagicMock()

    mock_core_module = MagicMock()
    mock_core_module.Settings = mock_settings
    mock_core_module.callbacks = mock_callbacks_module
    mock_core_module.callbacks.CallbackManager = MagicMock(
        return_value=mock_callback_manager
    )
    mock_core_module.callbacks.schema = mock_callbacks_schema

    # Patch the imports at sys.modules level
    with patch.dict(
        "sys.modules",
        {
            "llama_index": MagicMock(),
            "llama_index.core": mock_core_module,
            "llama_index.core.callbacks": mock_callbacks_module,
            "llama_index.core.callbacks.schema": mock_callbacks_schema,
        },
    ):
        yield {
            "event_type": mock_event_type,
            "settings": mock_settings,
            "callback_manager": mock_callback_manager,
        }


class TestPrelaHandler:
    """Tests for PrelaHandler callback class."""

    def test_init(self):
        """Test handler initialization."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        assert handler._tracer is tracer
        assert handler._spans == {}
        assert handler._contexts == {}

    def test_event_start_llm(self):
        """Test LLM event start."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        event_id = str(uuid4())
        payload = {
            "serialized": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "messages": ["What is 2+2?"],
        }

        result_id = handler.on_event_start(
            event_type="LLM", payload=payload, event_id=event_id
        )

        assert result_id == event_id
        assert event_id in handler._spans
        assert event_id in handler._contexts

        span = handler._spans[event_id]
        assert "llamaindex.event_type" in span.attributes
        assert span.attributes["llamaindex.event_type"] == "LLM"
        assert span.attributes["llm.model"] == "gpt-4"
        assert span.attributes["llm.temperature"] == 0.7
        assert span.attributes["llm.max_tokens"] == 1024

    def test_event_start_embedding(self):
        """Test embedding event start."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        event_id = str(uuid4())
        payload = {
            "serialized": {"model_name": "text-embedding-ada-002"},
            "chunks": ["chunk1", "chunk2", "chunk3"],
        }

        handler.on_event_start(
            event_type="EMBEDDING", payload=payload, event_id=event_id
        )

        span = handler._spans[event_id]
        assert span.attributes["embedding.model"] == "text-embedding-ada-002"
        assert span.attributes["embedding.input_count"] == 3

    def test_event_start_retrieve(self):
        """Test retrieval event start."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        event_id = str(uuid4())
        payload = {
            "query_str": "What is machine learning?",
            "retriever_type": "VectorIndexRetriever",
            "similarity_top_k": 5,
        }

        handler.on_event_start(
            event_type="RETRIEVE", payload=payload, event_id=event_id
        )

        span = handler._spans[event_id]
        assert span.attributes["retrieval.query"] == "What is machine learning?"
        assert span.attributes["retrieval.type"] == "VectorIndexRetriever"
        assert span.attributes["retrieval.top_k"] == 5

    def test_event_start_query(self):
        """Test query event start."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        event_id = str(uuid4())
        payload = {"query_str": "Tell me about AI", "query_mode": "default"}

        handler.on_event_start(
            event_type="QUERY", payload=payload, event_id=event_id
        )

        span = handler._spans[event_id]
        assert span.attributes["query.input"] == "Tell me about AI"
        assert span.attributes["query.mode"] == "default"

    def test_event_end_llm(self):
        """Test LLM event end."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start event
        event_id = str(uuid4())
        handler.on_event_start(
            event_type="LLM",
            payload={"serialized": {"model": "gpt-4"}},
            event_id=event_id,
        )

        # Create mock response with usage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_raw = Mock()
        mock_raw.usage = mock_usage

        mock_response = Mock()
        mock_response.text = "The answer is 4"
        mock_response.raw = mock_raw

        # End event
        handler.on_event_end(
            event_type="LLM", payload={"response": mock_response}, event_id=event_id
        )

        # Span should be ended and cleaned up
        assert event_id not in handler._spans
        assert event_id not in handler._contexts

    def test_event_end_embedding(self):
        """Test embedding event end."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start event
        event_id = str(uuid4())
        handler.on_event_start(
            event_type="EMBEDDING", payload={}, event_id=event_id
        )

        # End event with embeddings
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        handler.on_event_end(
            event_type="EMBEDDING", payload={"chunks": embeddings}, event_id=event_id
        )

        # Span cleaned up
        assert event_id not in handler._spans

    def test_event_end_retrieve(self):
        """Test retrieval event end with nodes."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        exporter = ConsoleExporter()
        handler = PrelaHandler(tracer)

        # Start event
        event_id = str(uuid4())
        handler.on_event_start(
            event_type="RETRIEVE",
            payload={"query_str": "test query"},
            event_id=event_id,
        )

        # Create mock nodes with scores and text
        mock_node1 = Mock()
        mock_node1.node.text = "This is the first retrieved document"
        mock_node1.node.metadata = {"file_name": "doc1.txt", "page_label": "1"}
        mock_node1.score = 0.95

        mock_node2 = Mock()
        mock_node2.node.text = "This is the second retrieved document"
        mock_node2.node.metadata = {"file_name": "doc2.txt"}
        mock_node2.score = 0.87

        nodes = [mock_node1, mock_node2]

        # End event
        handler.on_event_end(
            event_type="RETRIEVE", payload={"nodes": nodes}, event_id=event_id
        )

        # Verify span was cleaned up
        assert event_id not in handler._spans

    def test_event_end_query(self):
        """Test query event end."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start event
        event_id = str(uuid4())
        handler.on_event_start(
            event_type="QUERY", payload={"query_str": "test"}, event_id=event_id
        )

        # Create mock response
        mock_response = Mock()
        mock_response.response = "This is the generated answer"
        mock_response.source_nodes = [Mock(), Mock(), Mock()]

        # End event
        handler.on_event_end(
            event_type="QUERY", payload={"response": mock_response}, event_id=event_id
        )

        assert event_id not in handler._spans

    def test_event_end_synthesize(self):
        """Test synthesis event end."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start event
        event_id = str(uuid4())
        handler.on_event_start(
            event_type="SYNTHESIZE",
            payload={"query_str": "test", "nodes": [Mock(), Mock()]},
            event_id=event_id,
        )

        # Create mock response
        mock_response = Mock()
        mock_response.response = "Synthesized response from nodes"

        # End event
        handler.on_event_end(
            event_type="SYNTHESIZE",
            payload={"response": mock_response},
            event_id=event_id,
        )

        assert event_id not in handler._spans

    def test_event_missing_span(self):
        """Test ending event with no matching start."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Try to end non-existent event (should not crash)
        handler.on_event_end(
            event_type="LLM", payload={"response": Mock()}, event_id="nonexistent"
        )

        # Should not raise any errors

    def test_truncation_long_text(self):
        """Test that long text is truncated."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        event_id = str(uuid4())
        long_text = "a" * 1000

        payload = {"query_str": long_text}

        handler.on_event_start(
            event_type="RETRIEVE", payload=payload, event_id=event_id
        )

        span = handler._spans[event_id]
        # Should be truncated to 500 chars
        assert len(span.attributes["retrieval.query"]) == 500

    def test_span_type_mapping(self):
        """Test that event types map to correct span types."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        test_cases = [
            ("LLM", SpanType.LLM),
            ("EMBEDDING", SpanType.EMBEDDING),
            ("RETRIEVE", SpanType.RETRIEVAL),
            ("QUERY", SpanType.AGENT),
            ("SYNTHESIZE", SpanType.AGENT),
            ("CHUNKING", SpanType.CUSTOM),
        ]

        for event_type, expected_span_type in test_cases:
            event_id = str(uuid4())
            handler.on_event_start(
                event_type=event_type, payload={}, event_id=event_id
            )

            span = handler._spans[event_id]
            assert span.span_type == expected_span_type

            # Clean up
            handler.on_event_end(event_type=event_type, payload={}, event_id=event_id)

    def test_error_in_event_start(self):
        """Test that errors in event_start don't crash."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Pass malformed payload that might cause errors
        event_id = str(uuid4())
        payload = {"serialized": "not a dict"}  # Wrong type

        # Should not raise
        handler.on_event_start(event_type="LLM", payload=payload, event_id=event_id)

    def test_error_in_event_end(self):
        """Test that errors in event_end don't crash."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start event
        event_id = str(uuid4())
        handler.on_event_start(event_type="LLM", payload={}, event_id=event_id)

        # Pass malformed response
        handler.on_event_end(
            event_type="LLM", payload={"response": "not an object"}, event_id=event_id
        )

        # Should still clean up
        assert event_id not in handler._spans

    def test_trace_methods(self):
        """Test start_trace and end_trace methods."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # These are no-ops but should not crash
        handler.start_trace(trace_id="test-trace")
        handler.end_trace(trace_id="test-trace", trace_map={})


class TestLlamaIndexInstrumentor:
    """Tests for LlamaIndexInstrumentor."""

    def test_init(self):
        """Test instrumentor initialization."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        instrumentor = LlamaIndexInstrumentor()

        assert instrumentor._handler is None
        assert not instrumentor._instrumented
        assert not instrumentor.is_instrumented

    def test_instrument_success(self, mock_llamaindex):
        """Test successful instrumentation."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        tracer = Tracer(service_name="test")
        instrumentor = LlamaIndexInstrumentor()

        instrumentor.instrument(tracer)

        assert instrumentor.is_instrumented
        assert instrumentor._handler is not None
        assert mock_llamaindex["settings"].callback_manager.add_handler.called

    def test_instrument_idempotent(self, mock_llamaindex):
        """Test that instrumentation is idempotent."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        tracer = Tracer(service_name="test")
        instrumentor = LlamaIndexInstrumentor()

        # Instrument twice
        instrumentor.instrument(tracer)
        instrumentor.instrument(tracer)

        # Should only add handler once
        assert mock_llamaindex["settings"].callback_manager.add_handler.call_count == 1

    def test_uninstrument(self, mock_llamaindex):
        """Test uninstrumentation."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        tracer = Tracer(service_name="test")
        instrumentor = LlamaIndexInstrumentor()

        # Instrument first
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        # Then uninstrument
        instrumentor.uninstrument()

        assert not instrumentor.is_instrumented
        assert instrumentor._handler is None
        assert mock_llamaindex["settings"].callback_manager.remove_handler.called

    def test_uninstrument_when_not_instrumented(self):
        """Test uninstrument when not instrumented."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        instrumentor = LlamaIndexInstrumentor()

        # Should not crash
        instrumentor.uninstrument()

        assert not instrumentor.is_instrumented

    def test_get_handler(self, mock_llamaindex):
        """Test getting handler instance."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        tracer = Tracer(service_name="test")
        instrumentor = LlamaIndexInstrumentor()

        # Before instrumentation
        assert instrumentor.get_handler() is None

        # After instrumentation
        instrumentor.instrument(tracer)
        handler = instrumentor.get_handler()
        assert handler is not None
        assert handler is instrumentor._handler

    def test_instrument_missing_package(self, mock_llamaindex):
        """Test instrumentation with missing package."""
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor

        tracer = Tracer(service_name="test")
        instrumentor = LlamaIndexInstrumentor()

        # Simulate missing package by making import fail
        # Remove the mocked module temporarily
        import sys

        saved_modules = {}
        for key in ["llama_index.core", "llama_index.core.callbacks"]:
            if key in sys.modules:
                saved_modules[key] = sys.modules[key]
                del sys.modules[key]

        try:
            with pytest.raises(RuntimeError, match="llama-index-core is not installed"):
                instrumentor.instrument(tracer)
        finally:
            # Restore modules
            sys.modules.update(saved_modules)

    def test_concurrent_events(self, mock_llamaindex):
        """Test handling concurrent events."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start multiple events concurrently
        event_id1 = str(uuid4())
        event_id2 = str(uuid4())
        event_id3 = str(uuid4())

        handler.on_event_start(
            event_type="LLM",
            payload={"serialized": {"model": "gpt-4"}},
            event_id=event_id1,
        )
        handler.on_event_start(
            event_type="RETRIEVE",
            payload={"query_str": "query"},
            event_id=event_id2,
        )
        handler.on_event_start(
            event_type="EMBEDDING",
            payload={"chunks": ["a", "b"]},
            event_id=event_id3,
        )

        # All should be tracked
        assert len(handler._spans) == 3
        assert event_id1 in handler._spans
        assert event_id2 in handler._spans
        assert event_id3 in handler._spans

        # End them in different order
        handler.on_event_end(event_type="RETRIEVE", payload={}, event_id=event_id2)
        assert len(handler._spans) == 2

        handler.on_event_end(event_type="LLM", payload={}, event_id=event_id1)
        assert len(handler._spans) == 1

        handler.on_event_end(event_type="EMBEDDING", payload={}, event_id=event_id3)
        assert len(handler._spans) == 0

    def test_nested_events(self, mock_llamaindex):
        """Test handling nested events with parent_id."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = Tracer(service_name="test")
        handler = PrelaHandler(tracer)

        # Start parent event
        parent_id = str(uuid4())
        handler.on_event_start(
            event_type="QUERY",
            payload={"query_str": "parent query"},
            event_id=parent_id,
        )

        # Start child event
        child_id = str(uuid4())
        handler.on_event_start(
            event_type="LLM",
            payload={"serialized": {"model": "gpt-4"}},
            event_id=child_id,
            parent_id=parent_id,
        )

        # Both should be tracked
        assert len(handler._spans) == 2

        # End child first
        handler.on_event_end(event_type="LLM", payload={}, event_id=child_id)
        assert len(handler._spans) == 1

        # Then parent
        handler.on_event_end(event_type="QUERY", payload={}, event_id=parent_id)
        assert len(handler._spans) == 0
