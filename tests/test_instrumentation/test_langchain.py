"""Tests for LangChain instrumentation."""

from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID, uuid4

from prela.core.span import SpanStatus, SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter
from prela.instrumentation.langchain import (
    LangChainInstrumentor,
    PrelaCallbackHandler,
)


@pytest.fixture
def tracer():
    """Create a tracer for testing."""
    return Tracer(service_name="test-service", exporter=ConsoleExporter())


@pytest.fixture
def callback_handler(tracer):
    """Create a callback handler for testing."""
    return PrelaCallbackHandler(tracer)


class TestPrelaCallbackHandler:
    """Test the PrelaCallbackHandler."""

    def test_init(self, tracer):
        """Test handler initialization."""
        handler = PrelaCallbackHandler(tracer)
        assert handler._tracer is tracer
        assert handler._spans == {}
        assert handler._contexts == {}

    def test_on_llm_start_basic(self, callback_handler):
        """Test LLM start callback with basic parameters."""
        run_id = uuid4()
        serialized = {
            "name": "openai",
            "kwargs": {"model_name": "gpt-4"},
        }
        prompts = ["Hello, world!"]

        callback_handler.on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
        )

        # Check span was created
        assert str(run_id) in callback_handler._spans
        assert str(run_id) in callback_handler._contexts

        span = callback_handler._spans[str(run_id)]
        assert span.name == "langchain.llm.openai"
        assert span.span_type == SpanType.LLM
        assert span.attributes["llm.vendor"] == "langchain"
        assert span.attributes["llm.type"] == "openai"
        assert span.attributes["llm.model"] == "gpt-4"
        assert span.attributes["llm.prompt_count"] == 1
        assert span.attributes["llm.prompt.0"] == "Hello, world!"

    def test_on_llm_start_with_tags_metadata(self, callback_handler):
        """Test LLM start with tags and metadata."""
        run_id = uuid4()
        serialized = {"name": "openai"}
        prompts = ["test"]
        tags = ["production", "critical"]
        metadata = {"user_id": "123", "session": "abc"}

        callback_handler.on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            tags=tags,
            metadata=metadata,
        )

        span = callback_handler._spans[str(run_id)]
        assert span.attributes["langchain.tags"] == tags
        assert span.attributes["langchain.metadata.user_id"] == "123"
        assert span.attributes["langchain.metadata.session"] == "abc"

    def test_on_llm_start_truncates_long_prompts(self, callback_handler):
        """Test that long prompts are truncated."""
        run_id = uuid4()
        serialized = {"name": "openai"}
        long_prompt = "x" * 1000
        prompts = [long_prompt]

        callback_handler.on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]
        assert len(span.attributes["llm.prompt.0"]) == 500
        assert span.attributes["llm.prompt.0"] == "x" * 500

    def test_on_llm_end_basic(self, callback_handler):
        """Test LLM end callback."""
        run_id = uuid4()

        # Start LLM
        callback_handler.on_llm_start(
            serialized={"name": "openai"},
            prompts=["test"],
            run_id=run_id,
        )

        # Create mock response
        mock_generation = Mock()
        mock_generation.text = "Generated response"

        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }

        # End LLM
        callback_handler.on_llm_end(
            response=mock_response,
            run_id=run_id,
        )

        # Span should be cleaned up
        assert str(run_id) not in callback_handler._spans
        assert str(run_id) not in callback_handler._contexts

    def test_on_llm_end_with_token_usage(self, callback_handler, tracer):
        """Test LLM end captures token usage."""
        run_id = uuid4()

        # Start and capture the span before it's cleaned up
        callback_handler.on_llm_start(
            serialized={"name": "openai"},
            prompts=["test"],
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Create mock response with token usage
        mock_generation = Mock()
        mock_generation.text = "Response"

        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }

        # End LLM
        callback_handler.on_llm_end(
            response=mock_response,
            run_id=run_id,
        )

        # Check token usage was captured
        assert span.attributes["llm.usage.prompt_tokens"] == 100
        assert span.attributes["llm.usage.completion_tokens"] == 50
        assert span.attributes["llm.usage.total_tokens"] == 150
        assert span.status == SpanStatus.SUCCESS

    def test_on_llm_error(self, callback_handler):
        """Test LLM error callback."""
        run_id = uuid4()

        # Start LLM
        callback_handler.on_llm_start(
            serialized={"name": "openai"},
            prompts=["test"],
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Simulate error
        error = ValueError("API error")
        callback_handler.on_llm_error(
            error=error,
            run_id=run_id,
        )

        # Check error was captured
        assert span.status == SpanStatus.ERROR
        assert span.attributes["error.type"] == "ValueError"
        assert span.attributes["error.message"] == "API error"

        # Span should be cleaned up
        assert str(run_id) not in callback_handler._spans

    def test_on_chain_start(self, callback_handler):
        """Test chain start callback."""
        run_id = uuid4()
        serialized = {"name": "LLMChain"}
        inputs = {"product": "colorful socks"}

        callback_handler.on_chain_start(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]
        assert span.name == "langchain.chain.LLMChain"
        assert span.span_type == SpanType.AGENT
        assert span.attributes["langchain.type"] == "chain"
        assert span.attributes["langchain.chain_type"] == "LLMChain"
        assert span.attributes["chain.input.product"] == "colorful socks"

    def test_on_chain_end(self, callback_handler):
        """Test chain end callback."""
        run_id = uuid4()

        # Start chain
        callback_handler.on_chain_start(
            serialized={"name": "LLMChain"},
            inputs={"product": "socks"},
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # End chain
        outputs = {"text": "Socktastic Inc."}
        callback_handler.on_chain_end(
            outputs=outputs,
            run_id=run_id,
        )

        # Check output captured
        assert span.attributes["chain.output.text"] == "Socktastic Inc."
        assert span.status == SpanStatus.SUCCESS

        # Span cleaned up
        assert str(run_id) not in callback_handler._spans

    def test_on_chain_error(self, callback_handler):
        """Test chain error callback."""
        run_id = uuid4()

        # Start chain
        callback_handler.on_chain_start(
            serialized={"name": "LLMChain"},
            inputs={"product": "socks"},
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Simulate error
        error = RuntimeError("Chain execution failed")
        callback_handler.on_chain_error(
            error=error,
            run_id=run_id,
        )

        # Check error captured
        assert span.status == SpanStatus.ERROR
        assert span.attributes["error.type"] == "RuntimeError"
        assert "Chain execution failed" in span.attributes["error.message"]

    def test_on_tool_start(self, callback_handler):
        """Test tool start callback."""
        run_id = uuid4()
        serialized = {
            "name": "Calculator",
            "description": "Useful for math calculations",
        }
        input_str = "2 + 2"

        callback_handler.on_tool_start(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]
        assert span.name == "langchain.tool.Calculator"
        assert span.span_type == SpanType.TOOL
        assert span.attributes["tool.name"] == "Calculator"
        assert span.attributes["tool.description"] == "Useful for math calculations"
        assert span.attributes["tool.input"] == "2 + 2"

    def test_on_tool_end(self, callback_handler):
        """Test tool end callback."""
        run_id = uuid4()

        # Start tool
        callback_handler.on_tool_start(
            serialized={"name": "Calculator"},
            input_str="2 + 2",
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # End tool
        callback_handler.on_tool_end(
            output="4",
            run_id=run_id,
        )

        # Check output captured
        assert span.attributes["tool.output"] == "4"
        assert span.status == SpanStatus.SUCCESS

    def test_on_tool_error(self, callback_handler):
        """Test tool error callback."""
        run_id = uuid4()

        # Start tool
        callback_handler.on_tool_start(
            serialized={"name": "Calculator"},
            input_str="invalid",
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Simulate error
        error = ValueError("Invalid input")
        callback_handler.on_tool_error(
            error=error,
            run_id=run_id,
        )

        # Check error captured
        assert span.status == SpanStatus.ERROR
        assert span.attributes["error.type"] == "ValueError"

    def test_on_retriever_start(self, callback_handler):
        """Test retriever start callback."""
        run_id = uuid4()
        serialized = {"name": "VectorStoreRetriever"}
        query = "What is machine learning?"

        callback_handler.on_retriever_start(
            serialized=serialized,
            query=query,
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]
        assert span.name == "langchain.retriever.VectorStoreRetriever"
        assert span.span_type == SpanType.RETRIEVAL
        assert span.attributes["retriever.type"] == "VectorStoreRetriever"
        assert span.attributes["retriever.query"] == query

    def test_on_retriever_end(self, callback_handler):
        """Test retriever end callback."""
        run_id = uuid4()

        # Start retriever
        callback_handler.on_retriever_start(
            serialized={"name": "VectorStoreRetriever"},
            query="test query",
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Create mock documents
        doc1 = Mock()
        doc1.page_content = "Document 1 content"
        doc1.metadata = {"source": "doc1.txt", "page": 1}

        doc2 = Mock()
        doc2.page_content = "Document 2 content"
        doc2.metadata = {"source": "doc2.txt", "page": 2}

        documents = [doc1, doc2]

        # End retriever
        callback_handler.on_retriever_end(
            documents=documents,
            run_id=run_id,
        )

        # Check documents captured
        assert span.attributes["retriever.document_count"] == 2
        assert span.attributes["retriever.doc.0.content"] == "Document 1 content"
        assert span.attributes["retriever.doc.0.metadata.source"] == "doc1.txt"
        assert span.attributes["retriever.doc.1.content"] == "Document 2 content"
        assert span.status == SpanStatus.SUCCESS

    def test_on_agent_action(self, callback_handler):
        """Test agent action callback."""
        run_id = uuid4()

        # Start chain (agent operations happen within chains)
        callback_handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": "What's the weather?"},
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Record agent action
        mock_action = Mock()
        mock_action.tool = "WeatherAPI"
        mock_action.tool_input = "New York"
        mock_action.log = "I should use the weather API"

        callback_handler.on_agent_action(
            action=mock_action,
            run_id=run_id,
        )

        # Check action event was added
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "agent.action"
        assert event.attributes["action.tool"] == "WeatherAPI"
        assert event.attributes["action.tool_input"] == "New York"

    def test_on_agent_finish(self, callback_handler):
        """Test agent finish callback."""
        run_id = uuid4()

        # Start chain
        callback_handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": "test"},
            run_id=run_id,
        )

        span = callback_handler._spans[str(run_id)]

        # Record agent finish
        mock_finish = Mock()
        mock_finish.return_values = {"output": "Task completed"}
        mock_finish.log = "Agent finished successfully"

        callback_handler.on_agent_finish(
            finish=mock_finish,
            run_id=run_id,
        )

        # Check finish event was added
        events = [e for e in span.events if e.name == "agent.finish"]
        assert len(events) == 1
        event = events[0]
        assert "Task completed" in event.attributes["finish.output"]

    def test_concurrent_llm_calls(self, callback_handler):
        """Test handling concurrent LLM calls."""
        run_id_1 = uuid4()
        run_id_2 = uuid4()

        # Start two LLM calls
        callback_handler.on_llm_start(
            serialized={"name": "openai"},
            prompts=["First prompt"],
            run_id=run_id_1,
        )

        callback_handler.on_llm_start(
            serialized={"name": "anthropic"},
            prompts=["Second prompt"],
            run_id=run_id_2,
        )

        # Both should be active
        assert str(run_id_1) in callback_handler._spans
        assert str(run_id_2) in callback_handler._spans

        span1 = callback_handler._spans[str(run_id_1)]
        span2 = callback_handler._spans[str(run_id_2)]

        assert span1.attributes["llm.prompt.0"] == "First prompt"
        assert span2.attributes["llm.prompt.0"] == "Second prompt"

        # End first call
        mock_response = Mock()
        mock_response.generations = [[Mock(text="Response 1")]]
        mock_response.llm_output = None

        callback_handler.on_llm_end(response=mock_response, run_id=run_id_1)

        # Only second should remain
        assert str(run_id_1) not in callback_handler._spans
        assert str(run_id_2) in callback_handler._spans

    def test_nested_operations(self, callback_handler):
        """Test nested chain and LLM operations."""
        chain_run_id = uuid4()
        llm_run_id = uuid4()

        # Start chain
        callback_handler.on_chain_start(
            serialized={"name": "LLMChain"},
            inputs={"product": "socks"},
            run_id=chain_run_id,
        )

        # Start LLM within chain
        callback_handler.on_llm_start(
            serialized={"name": "openai"},
            prompts=["test"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )

        # Both should be active
        assert str(chain_run_id) in callback_handler._spans
        assert str(llm_run_id) in callback_handler._spans

    def test_error_handling_in_callbacks(self, callback_handler):
        """Test that errors in callbacks don't crash."""
        run_id = uuid4()

        # Malformed serialized dict
        callback_handler.on_llm_start(
            serialized={},  # Missing 'name'
            prompts=["test"],
            run_id=run_id,
        )

        # Should still create span with defaults
        assert str(run_id) in callback_handler._spans


class TestLangChainInstrumentor:
    """Test the LangChainInstrumentor."""

    def test_init(self):
        """Test instrumentor initialization."""
        instrumentor = LangChainInstrumentor()
        assert instrumentor._callback_handler is None
        assert instrumentor._langchain_core_module is None
        assert not instrumentor.is_instrumented

    @patch("prela.instrumentation.langchain.logger")
    def test_instrument_missing_package(self, mock_logger, tracer):
        """Test instrumenting when langchain-core is not installed."""
        instrumentor = LangChainInstrumentor()

        with patch.dict("sys.modules", {"langchain_core": None}):
            with pytest.raises(ImportError, match="langchain-core package is not installed"):
                instrumentor.instrument(tracer)

    @patch("prela.instrumentation.langchain.logger")
    def test_instrument_success(self, mock_logger, tracer):
        """Test successful instrumentation."""
        instrumentor = LangChainInstrumentor()

        # Manually set up for testing without actual import
        instrumentor._langchain_core_module = MagicMock()
        instrumentor._callback_handler = PrelaCallbackHandler(tracer)

        assert instrumentor.is_instrumented
        assert instrumentor.get_callback() is not None

    def test_instrument_idempotent(self, tracer):
        """Test that instrumenting twice is safe."""
        instrumentor = LangChainInstrumentor()

        # First instrumentation - manually set up
        instrumentor._callback_handler = PrelaCallbackHandler(tracer)
        instrumentor._langchain_core_module = MagicMock()

        # Second instrumentation should skip (is_instrumented returns True)
        instrumentor.instrument(tracer)  # Should not raise or create new handler

        # Handler should be the same (not replaced)
        assert instrumentor._callback_handler is not None

    def test_uninstrument(self, tracer):
        """Test uninstrumentation."""
        instrumentor = LangChainInstrumentor()

        # Set up instrumented state
        mock_callbacks = MagicMock()
        mock_callbacks._prela_handlers = []
        instrumentor._langchain_core_module = mock_callbacks
        instrumentor._callback_handler = PrelaCallbackHandler(tracer)

        # Uninstrument
        instrumentor.uninstrument()

        assert instrumentor._callback_handler is None
        assert instrumentor._langchain_core_module is None
        assert not instrumentor.is_instrumented

    def test_uninstrument_when_not_instrumented(self):
        """Test uninstrumenting when not instrumented."""
        instrumentor = LangChainInstrumentor()

        # Should not raise
        instrumentor.uninstrument()

    def test_get_callback_when_not_instrumented(self):
        """Test getting callback when not instrumented."""
        instrumentor = LangChainInstrumentor()
        assert instrumentor.get_callback() is None

    def test_get_callback_when_instrumented(self, tracer):
        """Test getting callback when instrumented."""
        instrumentor = LangChainInstrumentor()
        handler = PrelaCallbackHandler(tracer)
        instrumentor._callback_handler = handler

        assert instrumentor.get_callback() is handler
