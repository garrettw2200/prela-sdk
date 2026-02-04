"""Tests for LangGraph instrumentation."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from prela.core.span import SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter
from prela.instrumentation.multi_agent.langgraph import LangGraphInstrumentor


# Mock LangGraph classes
class MockStateGraph:
    """Mock LangGraph StateGraph."""

    def __init__(self, state_schema=None):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []

    def add_node(self, name: str, action: callable):
        """Add a node to the graph."""
        self.nodes[name] = action
        return self

    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes."""
        self.edges.append((from_node, to_node))
        return self

    def compile(self, *args, **kwargs):
        """Compile the graph."""
        compiled = MockCompiledGraph(self.nodes)
        return compiled


class MockCompiledGraph:
    """Mock compiled LangGraph graph."""

    def __init__(self, nodes: dict):
        self.nodes = nodes

    def invoke(self, state: dict, *args, **kwargs) -> dict:
        """Execute the graph."""
        if isinstance(state, dict):
            result = state.copy()
        else:
            result = {"input": state}
        # Execute each node in sequence
        for node_name, action in self.nodes.items():
            result = action(result)
        return result

    def stream(self, state: dict, *args, **kwargs):
        """Stream execution steps."""
        result = state.copy()
        for node_name, action in self.nodes.items():
            result = action(result)
            yield {node_name: result}


class MockTool:
    """Mock tool for react agent."""

    def __init__(self, name: str):
        self.name = name


def mock_create_react_agent(model, tools, *args, **kwargs):
    """Mock create_react_agent function."""
    # Create a simple compiled graph
    graph = MockStateGraph()
    graph.add_node("agent", lambda state: {**state, "output": "Agent response"})
    return graph.compile()


# Create mock langgraph modules
mock_langgraph_graph = MagicMock()
mock_langgraph_graph.StateGraph = MockStateGraph

mock_langgraph_prebuilt = MagicMock()
mock_langgraph_prebuilt.create_react_agent = mock_create_react_agent


@pytest.fixture
def mock_langgraph_modules():
    """Fixture to mock langgraph modules."""
    # Create a proper module structure
    sys.modules["langgraph"] = MagicMock()
    sys.modules["langgraph.graph"] = mock_langgraph_graph
    sys.modules["langgraph.prebuilt"] = mock_langgraph_prebuilt
    yield
    # Cleanup
    for module in ["langgraph.prebuilt", "langgraph.graph", "langgraph"]:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture
def instrumentor():
    """Create LangGraph instrumentor."""
    return LangGraphInstrumentor()


@pytest.fixture
def tracer():
    """Create tracer with console exporter."""
    return Tracer(service_name="test-langgraph", exporter=ConsoleExporter())


class TestLangGraphInstrumentor:
    """Test LangGraph instrumentor."""

    def test_init(self, instrumentor):
        """Test instrumentor initialization."""
        assert instrumentor.FRAMEWORK == "langgraph"
        assert not instrumentor.is_instrumented
        assert instrumentor._original_methods == {}
        assert instrumentor._graph_metadata == {}

    def test_instrument_without_langgraph(self, instrumentor, tracer):
        """Test instrument when langgraph not installed."""
        # Should not raise error
        instrumentor.instrument(tracer)
        assert not instrumentor.is_instrumented

    def test_instrument_with_langgraph(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test instrument with langgraph installed."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented
        assert instrumentor._tracer is tracer

        # Verify that StateGraph.compile was patched
        assert (MockStateGraph, "compile") in instrumentor._original_methods
        assert (MockStateGraph, "add_node") in instrumentor._original_methods

    def test_instrument_idempotent(self, instrumentor, tracer, mock_langgraph_modules):
        """Test that instrument can be called multiple times."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        # Call again - should not raise
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

    def test_uninstrument(self, instrumentor, tracer, mock_langgraph_modules):
        """Test uninstrument restores original methods."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        # Uninstrument
        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented
        assert instrumentor._tracer is None
        assert len(instrumentor._original_methods) == 0
        assert len(instrumentor._graph_metadata) == 0

    def test_state_graph_compile_basic(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test StateGraph.compile creates metadata."""
        instrumentor.instrument(tracer)

        # Create and compile graph
        graph = MockStateGraph()
        graph.add_node("node1", lambda state: {**state, "result": "test"})
        compiled = graph.compile()

        # Verify metadata was stored
        metadata = instrumentor._graph_metadata.get(id(compiled))
        assert metadata is not None
        assert "graph_id" in metadata
        assert "nodes" in metadata
        assert "node1" in metadata["nodes"]

    def test_graph_invoke_creates_span(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test graph.invoke creates span."""
        instrumentor.instrument(tracer)

        # Create graph
        graph = MockStateGraph()
        graph.add_node("node1", lambda state: {**state, "result": "test"})
        compiled = graph.compile()

        # Execute graph
        result = compiled.invoke({"input": "test"})

        assert result["input"] == "test"
        assert result["result"] == "test"

    def test_graph_invoke_without_tracer(
        self, instrumentor, mock_langgraph_modules
    ):
        """Test graph.invoke works without tracer."""
        instrumentor.instrument(None)  # No tracer provided

        # Create graph
        graph = MockStateGraph()
        graph.add_node("node1", lambda state: {**state, "result": "test"})
        compiled = graph.compile()

        # Execute graph - should work without tracing
        result = compiled.invoke({"input": "test"})
        assert result["result"] == "test"

    def test_graph_invoke_with_exception(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test graph.invoke handles exceptions."""
        instrumentor.instrument(tracer)

        # Create graph with failing node
        def failing_node(state):
            raise ValueError("Node failed")

        graph = MockStateGraph()
        graph.add_node("failing", failing_node)
        compiled = graph.compile()

        # Execute graph - should raise
        with pytest.raises(ValueError, match="Node failed"):
            compiled.invoke({"input": "test"})

    def test_graph_stream_creates_span(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test graph.stream creates span with events."""
        instrumentor.instrument(tracer)

        # Create graph
        graph = MockStateGraph()
        graph.add_node("node1", lambda state: {**state, "step": 1})
        graph.add_node("node2", lambda state: {**state, "step": 2})
        compiled = graph.compile()

        # Stream execution
        steps = list(compiled.stream({"input": "test"}))

        assert len(steps) == 2
        assert steps[0]["node1"]["step"] == 1
        assert steps[1]["node2"]["step"] == 2

    def test_graph_stream_without_tracer(
        self, instrumentor, mock_langgraph_modules
    ):
        """Test graph.stream works without tracer."""
        instrumentor.instrument(None)

        # Create graph
        graph = MockStateGraph()
        graph.add_node("node1", lambda state: {**state, "result": "test"})
        compiled = graph.compile()

        # Stream execution
        steps = list(compiled.stream({"input": "test"}))
        assert len(steps) == 1

    def test_node_action_wrapping(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test node actions are wrapped for tracing."""
        instrumentor.instrument(tracer)

        # Create graph with multiple nodes
        graph = MockStateGraph()
        graph.add_node("analyze", lambda state: {**state, "analyzed": True})
        graph.add_node("process", lambda state: {**state, "processed": True})
        compiled = graph.compile()

        # Execute
        result = compiled.invoke({"input": "data"})

        assert result["analyzed"] is True
        assert result["processed"] is True

    def test_node_tracks_changed_keys(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test node tracking of changed state keys."""
        instrumentor.instrument(tracer)

        # Create graph
        graph = MockStateGraph()
        graph.add_node(
            "transform",
            lambda state: {**state, "new_key": "new_value", "modified": "changed"},
        )
        compiled = graph.compile()

        # Execute
        result = compiled.invoke({"input": "test", "modified": "original"})

        assert result["new_key"] == "new_value"
        assert result["modified"] == "changed"

    def test_node_action_with_exception(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test node action handles exceptions."""
        instrumentor.instrument(tracer)

        def failing_action(state):
            raise RuntimeError("Node processing failed")

        graph = MockStateGraph()
        graph.add_node("failing", failing_action)
        compiled = graph.compile()

        with pytest.raises(RuntimeError, match="Node processing failed"):
            compiled.invoke({"input": "test"})

    def test_create_react_agent_patching(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test create_react_agent function is patched."""
        instrumentor.instrument(tracer)

        # Verify patching was registered
        # The wrapper stores the original function
        has_react_wrapper = any(
            "create_react_agent" in str(key) for key in instrumentor._original_methods.keys()
        )
        assert has_react_wrapper, "create_react_agent wrapper not registered"

        # Create react agent
        from langgraph.prebuilt import create_react_agent

        tools = [MockTool("search"), MockTool("calculate")]
        agent = create_react_agent(model="gpt-4", tools=tools)

        # The agent should be a compiled graph
        assert agent is not None
        assert hasattr(agent, "invoke")

        # Basic metadata should exist (from StateGraph.compile)
        metadata = instrumentor._graph_metadata.get(id(agent))
        assert metadata is not None, "No metadata found for agent"
        assert "graph_id" in metadata

    def test_create_react_agent_execution(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test react agent can execute."""
        instrumentor.instrument(tracer)

        from langgraph.prebuilt import create_react_agent

        tools = [MockTool("search")]
        agent = create_react_agent(model="gpt-4", tools=tools)

        # Execute agent
        result = agent.invoke({"input": "What is the weather?"})

        assert "output" in result
        assert result["output"] == "Agent response"

    def test_multiple_graphs(self, instrumentor, tracer, mock_langgraph_modules):
        """Test multiple graphs can be instrumented."""
        instrumentor.instrument(tracer)

        # Create first graph
        graph1 = MockStateGraph()
        graph1.add_node("node1", lambda state: {**state, "graph": 1})
        compiled1 = graph1.compile()

        # Create second graph
        graph2 = MockStateGraph()
        graph2.add_node("node2", lambda state: {**state, "graph": 2})
        compiled2 = graph2.compile()

        # Execute both
        result1 = compiled1.invoke({"input": "test"})
        result2 = compiled2.invoke({"input": "test"})

        assert result1["graph"] == 1
        assert result2["graph"] == 2

        # Verify separate metadata
        metadata1 = instrumentor._graph_metadata.get(id(compiled1))
        metadata2 = instrumentor._graph_metadata.get(id(compiled2))

        assert metadata1 is not None
        assert metadata2 is not None
        assert metadata1["graph_id"] != metadata2["graph_id"]

    def test_nested_node_execution(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test nested node execution (node calls another function)."""
        instrumentor.instrument(tracer)

        def helper_function(data: str) -> str:
            return data.upper()

        def node_action(state):
            processed = helper_function(state["input"])
            return {**state, "output": processed}

        graph = MockStateGraph()
        graph.add_node("process", node_action)
        compiled = graph.compile()

        result = compiled.invoke({"input": "hello"})
        assert result["output"] == "HELLO"

    def test_graph_with_empty_state(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test graph execution with empty state."""
        instrumentor.instrument(tracer)

        graph = MockStateGraph()
        graph.add_node("init", lambda state: {"initialized": True})
        compiled = graph.compile()

        result = compiled.invoke({})
        assert result["initialized"] is True

    def test_graph_with_non_dict_state(
        self, instrumentor, tracer, mock_langgraph_modules
    ):
        """Test graph with non-dict state (shouldn't crash)."""
        instrumentor.instrument(tracer)

        # Node that handles dict state (mock invoke wraps string in dict)
        def node_action(state):
            # MockCompiledGraph.invoke wraps non-dict in {"input": value}
            return {**state, "processed": True}

        graph = MockStateGraph()
        graph.add_node("convert", node_action)
        compiled = graph.compile()

        # This should not crash even with string state
        result = compiled.invoke("test_string")
        # Mock wraps string as {"input": "test_string"}
        assert result["input"] == "test_string"
        assert result["processed"] is True
