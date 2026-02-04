"""
Tests for n8n Code node tracing helpers.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from prela.core.span import SpanStatus, SpanType
from prela.core.tracer import Tracer
from prela.instrumentation.n8n.code_node import (
    PrelaN8nContext,
    prela_n8n_traced,
    trace_n8n_code,
)


@pytest.fixture
def mock_tracer():
    """Create a mock tracer with exporter."""
    tracer = Mock(spec=Tracer)
    tracer.exporter = Mock()
    tracer.exporter.export = Mock()
    return tracer


@pytest.fixture
def n8n_contexts():
    """Sample n8n context objects."""
    return {
        "workflow": {
            "id": "wf_123",
            "name": "Test Workflow",
            "active": True,
        },
        "execution": {
            "id": "exec_456",
            "mode": "manual",
            "startedAt": "2025-01-27T10:00:00Z",
        },
        "node": {
            "name": "Code Node",
            "type": "n8n-nodes-base.code",
            "parameters": {},
        },
        "items": [
            {"json": {"input": "test data"}},
        ],
    }


class TestPrelaN8nContext:
    """Tests for PrelaN8nContext class."""

    def test_initialization(self, mock_tracer):
        """Test basic initialization."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        assert ctx.workflow_id == "wf_123"
        assert ctx.workflow_name == "Test Workflow"
        assert ctx.execution_id == "exec_456"
        assert ctx.node_name == "Code Node"
        assert ctx.trace_id == "n8n-exec_456"
        assert ctx.tracer == mock_tracer

    def test_context_manager_creates_spans(self, mock_tracer):
        """Test context manager creates workflow and node spans."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with ctx:
            # Spans should be created
            assert ctx.workflow_span is not None
            assert ctx.node_span is not None

            # Check workflow span
            assert ctx.workflow_span.name == "n8n.workflow.Test Workflow"
            assert ctx.workflow_span.span_type == SpanType.AGENT
            assert ctx.workflow_span.parent_span_id is None
            assert ctx.workflow_span.attributes["n8n.workflow_id"] == "wf_123"
            assert ctx.workflow_span.attributes["n8n.workflow_name"] == "Test Workflow"

            # Check node span
            assert ctx.node_span.name == "n8n.node.Code Node"
            assert ctx.node_span.span_type == SpanType.CUSTOM
            assert ctx.node_span.parent_span_id == ctx.workflow_span.span_id
            assert ctx.node_span.attributes["n8n.node_name"] == "Code Node"

    def test_context_manager_ends_spans(self, mock_tracer):
        """Test context manager ends spans on exit."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with ctx:
            pass

        # Spans should be ended
        assert ctx.workflow_span.ended_at is not None
        assert ctx.node_span.ended_at is not None
        assert ctx.workflow_span.status == SpanStatus.SUCCESS
        assert ctx.node_span.status == SpanStatus.SUCCESS

    def test_context_manager_exports_spans(self, mock_tracer):
        """Test context manager exports spans to tracer."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with ctx:
            pass

        # Should export both spans
        assert mock_tracer.exporter.export.call_count == 2
        exported_spans = [
            call[0][0][0] for call in mock_tracer.exporter.export.call_args_list
        ]
        assert ctx.workflow_span in exported_spans
        assert ctx.node_span in exported_spans

    def test_context_manager_handles_exception(self, mock_tracer):
        """Test context manager handles exceptions properly."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test error")

        # Spans should be marked as error
        assert ctx.node_span.status == SpanStatus.ERROR
        assert ctx.workflow_span.status == SpanStatus.ERROR

        # Should have exception event on node span
        events = [e.name for e in ctx.node_span.events]
        assert "exception" in events

    def test_log_llm_call(self, mock_tracer):
        """Test logging LLM call creates child span."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with ctx:
            ctx.log_llm_call(
                model="gpt-4",
                prompt="Hello world",
                response="Hi there!",
                tokens={"prompt": 10, "completion": 5, "total": 15},
                provider="openai",
                temperature=0.7,
            )

        # Should export workflow, node, and LLM spans (3 total)
        assert mock_tracer.exporter.export.call_count == 3

        # Get LLM span (last export)
        llm_span = mock_tracer.exporter.export.call_args_list[0][0][0][0]

        assert llm_span.name == "llm.gpt-4"
        assert llm_span.span_type == SpanType.LLM
        assert llm_span.parent_span_id == ctx.node_span.span_id
        assert llm_span.attributes["llm.model"] == "gpt-4"
        assert llm_span.attributes["llm.prompt"] == "Hello world"
        assert llm_span.attributes["llm.response"] == "Hi there!"
        assert llm_span.attributes["llm.provider"] == "openai"
        assert llm_span.attributes["llm.temperature"] == 0.7
        assert llm_span.attributes["llm.prompt_tokens"] == 10
        assert llm_span.attributes["llm.completion_tokens"] == 5
        assert llm_span.attributes["llm.total_tokens"] == 15

    def test_log_llm_call_truncates_long_content(self, mock_tracer):
        """Test LLM logging truncates long prompts and responses."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        long_text = "x" * 1000

        with ctx:
            ctx.log_llm_call(
                model="gpt-4",
                prompt=long_text,
                response=long_text,
            )

        # Get LLM span
        llm_span = mock_tracer.exporter.export.call_args_list[0][0][0][0]

        assert len(llm_span.attributes["llm.prompt"]) == 500
        assert len(llm_span.attributes["llm.response"]) == 500

    def test_log_tool_call(self, mock_tracer):
        """Test logging tool call creates child span."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with ctx:
            ctx.log_tool_call(
                tool_name="calculator",
                input={"operation": "add", "x": 5, "y": 3},
                output={"result": 8},
            )

        # Should export workflow, node, and tool spans (3 total)
        assert mock_tracer.exporter.export.call_count == 3

        # Get tool span (first export)
        tool_span = mock_tracer.exporter.export.call_args_list[0][0][0][0]

        assert tool_span.name == "tool.calculator"
        assert tool_span.span_type == SpanType.TOOL
        assert tool_span.parent_span_id == ctx.node_span.span_id
        assert tool_span.attributes["tool.name"] == "calculator"
        assert "operation" in tool_span.attributes["tool.input"]
        assert "result" in tool_span.attributes["tool.output"]

    def test_log_tool_call_with_error(self, mock_tracer):
        """Test logging tool call with error marks span as failed."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        with ctx:
            ctx.log_tool_call(
                tool_name="calculator",
                input={"operation": "divide", "x": 5, "y": 0},
                output=None,
                error="Division by zero",
            )

        # Get tool span
        tool_span = mock_tracer.exporter.export.call_args_list[0][0][0][0]

        assert tool_span.status == SpanStatus.ERROR
        assert tool_span.attributes["tool.error"] == "Division by zero"

    def test_log_retrieval(self, mock_tracer):
        """Test logging retrieval creates child span."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        documents = [
            {"text": "Document 1", "score": 0.95},
            {"text": "Document 2", "score": 0.87},
            {"text": "Document 3", "score": 0.72},
        ]

        with ctx:
            ctx.log_retrieval(
                query="What is AI?",
                documents=documents,
                retriever_type="vector",
                similarity_top_k=5,
            )

        # Should export workflow, node, and retrieval spans (3 total)
        assert mock_tracer.exporter.export.call_count == 3

        # Get retrieval span (first export)
        retrieval_span = mock_tracer.exporter.export.call_args_list[0][0][0][0]

        assert retrieval_span.name == "retrieval"
        assert retrieval_span.span_type == SpanType.RETRIEVAL
        assert retrieval_span.parent_span_id == ctx.node_span.span_id
        assert retrieval_span.attributes["retrieval.query"] == "What is AI?"
        assert retrieval_span.attributes["retrieval.document_count"] == 3
        assert retrieval_span.attributes["retrieval.type"] == "vector"
        assert retrieval_span.attributes["retrieval.similarity_top_k"] == 5
        assert retrieval_span.attributes["retrieval.document.0.score"] == 0.95
        assert retrieval_span.attributes["retrieval.document.1.score"] == 0.87

    def test_log_retrieval_limits_documents(self, mock_tracer):
        """Test retrieval logging limits document count to 5."""
        ctx = PrelaN8nContext(
            workflow_id="wf_123",
            workflow_name="Test Workflow",
            execution_id="exec_456",
            node_name="Code Node",
            tracer=mock_tracer,
        )

        documents = [{"text": f"Doc {i}", "score": 0.9 - i * 0.1} for i in range(10)]

        with ctx:
            ctx.log_retrieval(
                query="Test query",
                documents=documents,
            )

        # Get retrieval span
        retrieval_span = mock_tracer.exporter.export.call_args_list[0][0][0][0]

        # Should only have first 5 documents
        assert "retrieval.document.0.text" in retrieval_span.attributes
        assert "retrieval.document.4.text" in retrieval_span.attributes
        assert "retrieval.document.5.text" not in retrieval_span.attributes

    def test_without_tracer_initializes_default(self, mock_tracer):
        """Test context initializes default tracer if none provided."""
        # When no tracer is provided, PrelaN8nContext will call get_tracer()
        # and if None, it will call prela.init() to create a default tracer
        with patch(
            "prela.instrumentation.n8n.code_node.get_tracer", return_value=None
        ), patch("prela.init") as mock_init:
            # Set up mock_init to return our mock tracer
            mock_init.return_value = mock_tracer

            ctx = PrelaN8nContext(
                workflow_id="wf_123",
                workflow_name="Test Workflow",
                execution_id="exec_456",
                node_name="Code Node",
            )

            with ctx:
                pass

            # Should initialize tracer
            mock_init.assert_called_once()


class TestTraceN8nCode:
    """Tests for trace_n8n_code convenience function."""

    def test_trace_n8n_code_creates_context(self, n8n_contexts, mock_tracer):
        """Test trace_n8n_code creates PrelaN8nContext."""
        ctx = trace_n8n_code(
            items=n8n_contexts["items"],
            workflow_context=n8n_contexts["workflow"],
            execution_context=n8n_contexts["execution"],
            node_context=n8n_contexts["node"],
            tracer=mock_tracer,
        )

        assert isinstance(ctx, PrelaN8nContext)
        assert ctx.workflow_id == "wf_123"
        assert ctx.workflow_name == "Test Workflow"
        assert ctx.execution_id == "exec_456"
        assert ctx.node_name == "Code Node"

    def test_trace_n8n_code_as_context_manager(self, n8n_contexts, mock_tracer):
        """Test trace_n8n_code can be used as context manager."""
        with trace_n8n_code(
            items=n8n_contexts["items"],
            workflow_context=n8n_contexts["workflow"],
            execution_context=n8n_contexts["execution"],
            node_context=n8n_contexts["node"],
            tracer=mock_tracer,
        ) as ctx:
            assert ctx.workflow_span is not None
            assert ctx.node_span is not None

    def test_trace_n8n_code_handles_missing_context_fields(self, mock_tracer):
        """Test trace_n8n_code handles missing context fields gracefully."""
        ctx = trace_n8n_code(
            items=[],
            workflow_context={},  # Missing id, name
            execution_context={},  # Missing id
            node_context={},  # Missing name, type
            tracer=mock_tracer,
        )

        assert ctx.workflow_id == "unknown"
        assert ctx.workflow_name == "Unknown Workflow"
        assert ctx.execution_id == "unknown"
        assert ctx.node_name == "Unknown Node"
        assert ctx.node_type == "n8n-nodes-base.code"


class TestPrelaN8nTracedDecorator:
    """Tests for prela_n8n_traced decorator."""

    def test_decorator_without_parentheses(self, n8n_contexts, mock_tracer):
        """Test @prela_n8n_traced decorator usage."""

        @prela_n8n_traced
        def my_function(items, workflow, execution, node):
            return [{"json": {"result": "success"}}]

        # Mock get_tracer to return our mock
        with patch(
            "prela.instrumentation.n8n.code_node.get_tracer", return_value=mock_tracer
        ):
            result = my_function(
                n8n_contexts["items"],
                n8n_contexts["workflow"],
                n8n_contexts["execution"],
                n8n_contexts["node"],
            )

        assert result == [{"json": {"result": "success"}}]
        # Should have exported spans
        assert mock_tracer.exporter.export.call_count > 0

    def test_decorator_with_parentheses(self, n8n_contexts, mock_tracer):
        """Test @prela_n8n_traced() decorator usage with parentheses."""

        @prela_n8n_traced()
        def my_function(items, workflow, execution, node):
            return [{"json": {"result": "success"}}]

        with patch(
            "prela.instrumentation.n8n.code_node.get_tracer", return_value=mock_tracer
        ):
            result = my_function(
                n8n_contexts["items"],
                n8n_contexts["workflow"],
                n8n_contexts["execution"],
                n8n_contexts["node"],
            )

        assert result == [{"json": {"result": "success"}}]
        assert mock_tracer.exporter.export.call_count > 0

    def test_decorator_with_custom_tracer(self, n8n_contexts, mock_tracer):
        """Test decorator with custom tracer."""

        @prela_n8n_traced(tracer=mock_tracer)
        def my_function(items, workflow, execution, node):
            return [{"json": {"result": "success"}}]

        result = my_function(
            n8n_contexts["items"],
            n8n_contexts["workflow"],
            n8n_contexts["execution"],
            n8n_contexts["node"],
        )

        assert result == [{"json": {"result": "success"}}]
        assert mock_tracer.exporter.export.call_count > 0

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @prela_n8n_traced
        def my_function(items, workflow, execution, node):
            """My function docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My function docstring."

    def test_decorator_forwards_extra_arguments(self, n8n_contexts, mock_tracer):
        """Test decorator forwards extra arguments to wrapped function."""

        @prela_n8n_traced(tracer=mock_tracer)
        def my_function(items, workflow, execution, node, extra_arg, kwarg=None):
            return {"extra": extra_arg, "kwarg": kwarg}

        result = my_function(
            n8n_contexts["items"],
            n8n_contexts["workflow"],
            n8n_contexts["execution"],
            n8n_contexts["node"],
            "extra_value",
            kwarg="kwarg_value",
        )

        assert result == {"extra": "extra_value", "kwarg": "kwarg_value"}

    def test_decorator_handles_exception(self, n8n_contexts, mock_tracer):
        """Test decorator handles exceptions and still exports spans."""

        @prela_n8n_traced(tracer=mock_tracer)
        def my_function(items, workflow, execution, node):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            my_function(
                n8n_contexts["items"],
                n8n_contexts["workflow"],
                n8n_contexts["execution"],
                n8n_contexts["node"],
            )

        # Should still export spans (with error status)
        assert mock_tracer.exporter.export.call_count > 0
