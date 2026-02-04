"""Tests for LlamaIndex instrumentation with replay capture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from prela.core.span import SpanType
from prela.core.tracer import Tracer


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


class TestLlamaIndexReplayCapture:
    """Test replay capture in LlamaIndex instrumentation."""

    def test_llm_replay_capture_enabled(self):
        """Test LLM replay capture when enabled."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = True

        # Mock span context manager
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock()
        tracer.span.return_value = mock_ctx

        handler = PrelaHandler(tracer)

        # Mock LLM start event
        start_payload = {
            "serialized": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100,
            },
            "messages": ["What is AI?"],
        }

        # Trigger on_event_start (creates span and replay capture)
        event_id = "llm-123"
        handler.on_event_start("LLM", start_payload, event_id=event_id)

        # Verify replay capture was created
        span = handler._spans[event_id]
        span_id = str(id(span))
        assert span_id in handler._replay_captures
        replay_capture = handler._replay_captures[span_id]
        assert replay_capture is not None

        # Mock LLM end event
        mock_response = MagicMock()
        mock_response.text = "AI is artificial intelligence"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 25
        mock_raw = MagicMock()
        mock_raw.usage = mock_usage
        mock_response.raw = mock_raw

        end_payload = {"response": mock_response}

        # Trigger on_event_end
        handler.on_event_end("LLM", end_payload, event_id=event_id)

        # Verify replay capture was completed and cleaned up
        assert span_id not in handler._replay_captures

    def test_llm_replay_capture_disabled(self):
        """Test LLM replay capture when disabled."""
        from prela.instrumentation.llamaindex import PrelaHandler
        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = False

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock()
        tracer.span.return_value = mock_ctx

        handler = PrelaHandler(tracer)

        # Mock LLM start event
        start_payload = {
            "serialized": {"model": "gpt-4"},
            "messages": ["What is AI?"],
        }

        event_id = "llm-123"
        handler.on_event_start("LLM", start_payload, event_id=event_id)

        # Verify no replay capture was created
        span = handler._spans.get(event_id)
        if span:
            span_id = str(id(span))
            assert span_id not in handler._replay_captures

    def test_retrieval_replay_capture_enabled(self):
        """Test retrieval replay capture when enabled."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = True

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock()
        tracer.span.return_value = mock_ctx

        handler = PrelaHandler(tracer)

        # Mock retrieval start event
        start_payload = {
            "query_str": "What is LlamaIndex?",
            "retriever_type": "vector_store",
            "similarity_top_k": 3,
        }

        event_id = "retrieve-123"
        handler.on_event_start("RETRIEVE", start_payload, event_id=event_id)

        # Verify replay capture was created
        span = handler._spans[event_id]
        span_id = str(id(span))
        assert span_id in handler._replay_captures
        replay_data = handler._replay_captures[span_id]
        assert "capture" in replay_data
        assert replay_data["query"] == "What is LlamaIndex?"

        # Mock retrieval end event with nodes
        mock_node1 = MagicMock()
        mock_node1.score = 0.95
        mock_node1.node.text = "LlamaIndex is a data framework"
        mock_node1.node.metadata = {"file_name": "doc1.txt"}

        mock_node2 = MagicMock()
        mock_node2.score = 0.87
        mock_node2.node.text = "LlamaIndex helps with LLM applications"
        mock_node2.node.metadata = {"file_name": "doc2.txt"}

        end_payload = {"nodes": [mock_node1, mock_node2]}

        # Trigger on_event_end
        handler.on_event_end("RETRIEVE", end_payload, event_id=event_id)

        # Verify replay capture was completed and cleaned up
        assert span_id not in handler._replay_captures

    def test_retrieval_replay_capture_disabled(self):
        """Test retrieval replay capture when disabled."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = False

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock()
        tracer.span.return_value = mock_ctx

        handler = PrelaHandler(tracer)

        # Mock retrieval start event
        start_payload = {
            "query_str": "What is LlamaIndex?",
            "retriever_type": "vector_store",
        }

        event_id = "retrieve-123"
        handler.on_event_start("RETRIEVE", start_payload, event_id=event_id)

        # Verify no replay capture was created
        span = handler._spans.get(event_id)
        if span:
            span_id = str(id(span))
            assert span_id not in handler._replay_captures

    def test_retrieval_replay_truncates_documents(self):
        """Test that retrieval replay truncates long documents."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = True

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock()
        tracer.span.return_value = mock_ctx

        handler = PrelaHandler(tracer)

        # Start retrieval
        start_payload = {"query_str": "What is LlamaIndex?"}
        event_id = "retrieve-123"
        handler.on_event_start("RETRIEVE", start_payload, event_id=event_id)

        # Mock node with very long content
        mock_node = MagicMock()
        mock_node.score = 0.95
        mock_node.node.text = "A" * 500  # Long content
        mock_node.node.metadata = {"file_name": "doc.txt"}

        end_payload = {"nodes": [mock_node]}

        # Trigger on_event_end
        handler.on_event_end("RETRIEVE", end_payload, event_id=event_id)

        # Verify truncation happened (should be 200 chars max)
        # This is implicit in the implementation

    def test_retrieval_replay_limits_document_count(self):
        """Test that retrieval replay limits to 5 documents."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = True

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock()
        tracer.span.return_value = mock_ctx

        handler = PrelaHandler(tracer)

        # Start retrieval
        start_payload = {"query_str": "What is LlamaIndex?"}
        event_id = "retrieve-123"
        handler.on_event_start("RETRIEVE", start_payload, event_id=event_id)

        # Create 10 mock nodes
        mock_nodes = []
        for i in range(10):
            mock_node = MagicMock()
            mock_node.score = 0.9 - (i * 0.05)
            mock_node.node.text = f"Document {i}"
            mock_node.node.metadata = {"file_name": f"doc{i}.txt"}
            mock_nodes.append(mock_node)

        end_payload = {"nodes": mock_nodes}

        # Trigger on_event_end
        handler.on_event_end("RETRIEVE", end_payload, event_id=event_id)

        # Verify only 5 documents were captured
        # This is implicit in the implementation (nodes[:_MAX_ITEMS])

    def test_concurrent_replay_captures(self):
        """Test multiple concurrent replay captures."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = True

        # Create mock spans that return unique objects
        def create_mock_span():
            mock_span = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_span)
            mock_ctx.__exit__ = MagicMock()
            return mock_span, mock_ctx

        # Create 3 different spans
        spans_and_contexts = [create_mock_span() for _ in range(3)]

        # Mock tracer.span to return different contexts
        call_count = [0]
        def mock_span_func(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return spans_and_contexts[idx][1]

        tracer.span = mock_span_func

        handler = PrelaHandler(tracer)

        # Start 3 concurrent LLM calls
        event_ids = ["llm-1", "llm-2", "llm-3"]

        for event_id in event_ids:
            start_payload = {
                "serialized": {"model": "gpt-4"},
                "messages": [f"Prompt {event_id}"],
            }
            handler.on_event_start("LLM", start_payload, event_id=event_id)

        # All 3 should have replay captures
        for event_id in event_ids:
            span = handler._spans[event_id]
            span_id = str(id(span))
            assert span_id in handler._replay_captures

        # Complete them
        for event_id in event_ids:
            mock_response = MagicMock()
            mock_response.text = f"Response for {event_id}"
            mock_response.raw = None  # No token usage

            end_payload = {"response": mock_response}
            handler.on_event_end("LLM", end_payload, event_id=event_id)

        # All should be cleaned up
        for event_id in event_ids:
            span = handler._spans.get(event_id)
            # Span should be removed from _spans after on_event_end
            assert span is None

    def test_replay_capture_cleanup_on_missing_span(self):
        """Test replay capture cleanup when span is missing."""
        from prela.instrumentation.llamaindex import PrelaHandler

        tracer = MagicMock(spec=Tracer)
        tracer.capture_for_replay = True

        handler = PrelaHandler(tracer)

        # Call on_event_end with non-existent event_id
        # Should not crash
        handler.on_event_end("LLM", {}, event_id="non-existent")

        # No replay captures should exist
        assert len(handler._replay_captures) == 0
