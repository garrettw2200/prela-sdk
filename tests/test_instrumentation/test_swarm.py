"""Tests for Swarm instrumentation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock

import pytest

from prela.core.span import SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter
from prela.instrumentation.multi_agent.models import generate_agent_id
from prela.instrumentation.multi_agent.swarm import SwarmInstrumentor


# Mock Swarm classes
class MockAgent:
    """Mock Swarm Agent."""

    def __init__(self, name: str = "agent", **kwargs):
        self.name = name
        self.kwargs = kwargs


class MockResponse:
    """Mock Swarm Response."""

    def __init__(self, messages=None, agent=None, context_variables=None):
        self.messages = messages or []
        self.agent = agent
        self.context_variables = context_variables or {}


class MockSwarm:
    """Mock Swarm orchestrator."""

    def run(self, agent, messages, **kwargs):
        """Run swarm execution."""
        # Simulate handoff to another agent
        final_agent = MockAgent(name="final_agent")
        return MockResponse(
            messages=messages + [{"role": "assistant", "content": "Response"}],
            agent=final_agent,
            context_variables=kwargs.get("context_variables", {}),
        )


# Create mock swarm module
mock_swarm = MagicMock()
mock_swarm.Agent = MockAgent
mock_swarm.Swarm = MockSwarm


@pytest.fixture
def mock_swarm_module():
    """Fixture to mock swarm module."""
    sys.modules["swarm"] = mock_swarm
    yield mock_swarm
    if "swarm" in sys.modules:
        del sys.modules["swarm"]


@pytest.fixture
def instrumentor():
    """Create Swarm instrumentor."""
    return SwarmInstrumentor()


@pytest.fixture
def tracer():
    """Create tracer with console exporter."""
    return Tracer(service_name="test-swarm", exporter=ConsoleExporter())


class TestSwarmInstrumentor:
    """Test Swarm instrumentor."""

    def test_init(self, instrumentor):
        """Test instrumentor initialization."""
        assert instrumentor.FRAMEWORK == "swarm"
        assert not instrumentor.is_instrumented
        assert instrumentor._active_swarms == {}

    def test_instrument_without_swarm(self, instrumentor):
        """Test instrument when swarm not installed."""
        # Should not raise error
        instrumentor.instrument()
        assert not instrumentor.is_instrumented

    def test_instrument_with_swarm(self, instrumentor, tracer, mock_swarm_module):
        """Test instrument with swarm installed."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented
        assert instrumentor._tracer is tracer

        # Verify that methods were patched
        assert hasattr(MockSwarm, "_prela_original_run")
        assert hasattr(MockAgent, "_prela_original___init__")

    def test_instrument_idempotent(self, instrumentor, tracer, mock_swarm_module):
        """Test that instrument can be called multiple times."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        # Call again - should not raise
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

    def test_uninstrument(self, instrumentor, tracer, mock_swarm_module):
        """Test uninstrument restores original methods."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented
        assert instrumentor._tracer is None

        # Check that original methods were restored
        assert not hasattr(MockSwarm, "_prela_original_run")
        assert not hasattr(MockAgent, "_prela_original___init__")

    def test_swarm_run_basic(self, instrumentor, tracer, mock_swarm_module):
        """Test swarm run creates execution span."""
        instrumentor.instrument(tracer)

        # Create swarm and agent AFTER instrumentation
        swarm = MockSwarm()
        agent = MockAgent("assistant")

        messages = [{"role": "user", "content": "Hello"}]

        # The wrapped method should call the original
        result = swarm.run(agent, messages)

        # Result should be from the original method
        assert isinstance(result, MockResponse)
        assert result.agent.name == "final_agent"
        assert len(result.messages) == 2

    def test_swarm_run_with_context(self, instrumentor, tracer, mock_swarm_module):
        """Test swarm run with context variables."""
        instrumentor.instrument(tracer)

        swarm = MockSwarm()
        agent = MockAgent("assistant")
        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123", "session": "abc"}

        result = swarm.run(agent, messages, context_variables=context)

        assert isinstance(result, MockResponse)
        assert result.context_variables == context

    def test_swarm_run_with_exception(self, instrumentor, tracer, mock_swarm_module):
        """Test swarm run handles exceptions."""
        instrumentor.instrument(tracer)

        swarm = MockSwarm()
        agent = MockAgent("assistant")

        # Make run raise exception
        swarm.run = Mock(side_effect=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            swarm.run(agent, [{"role": "user", "content": "Hello"}])

    def test_agent_init_basic(self, instrumentor, tracer, mock_swarm_module):
        """Test agent init adds agent ID."""
        instrumentor.instrument(tracer)

        # Create agent AFTER instrumentation
        agent = MockAgent("assistant")

        assert hasattr(agent, "_prela_agent_id")
        assert agent._prela_agent_id == generate_agent_id("swarm", "assistant")

    def test_agent_id_generation(self, instrumentor):
        """Test agent ID generation is deterministic."""
        agent_id_1 = generate_agent_id("swarm", "Assistant")
        agent_id_2 = generate_agent_id("swarm", "Assistant")

        assert agent_id_1 == agent_id_2
        assert len(agent_id_1) == 12

    def test_different_agents_different_ids(self, instrumentor):
        """Test different agents get different IDs."""
        alice_id = generate_agent_id("swarm", "Alice")
        bob_id = generate_agent_id("swarm", "Bob")

        assert alice_id != bob_id

    def test_execution_tracking(self, instrumentor, tracer, mock_swarm_module):
        """Test execution tracking lifecycle."""
        instrumentor.instrument(tracer)

        swarm = MockSwarm()
        agent = MockAgent("assistant")

        # Before execution
        assert len(instrumentor._active_swarms) == 0

        # During execution
        result = swarm.run(agent, [{"role": "user", "content": "Hello"}])

        # After execution - should be cleaned up
        assert len(instrumentor._active_swarms) == 0
        assert isinstance(result, MockResponse)


class TestSwarmIntegration:
    """Integration tests for Swarm instrumentation."""

    def test_multi_agent_execution(self, instrumentor, tracer, mock_swarm_module):
        """Test complete multi-agent execution."""
        instrumentor.instrument(tracer)

        # Create agents
        assistant = MockAgent("assistant")
        researcher = MockAgent("researcher")

        # Create swarm
        swarm = MockSwarm()

        # Run execution
        messages = [{"role": "user", "content": "Research AI"}]
        result = swarm.run(assistant, messages)

        assert isinstance(result, MockResponse)
        assert result.agent.name == "final_agent"

    def test_uninstrumented_execution(self, mock_swarm_module):
        """Test that swarm works normally when not instrumented."""
        swarm = MockSwarm()
        agent = MockAgent("assistant")

        # Should work normally without instrumentation
        result = swarm.run(agent, [{"role": "user", "content": "Hello"}])
        assert isinstance(result, MockResponse)

    def test_execution_tracking_isolation(
        self, instrumentor, tracer, mock_swarm_module
    ):
        """Test that executions are tracked and cleaned up."""
        instrumentor.instrument(tracer)

        swarm = MockSwarm()
        agent = MockAgent("assistant")

        # Before execution
        assert len(instrumentor._active_swarms) == 0

        # During execution
        result = swarm.run(agent, [{"role": "user", "content": "Hello"}])

        # After execution - should be cleaned up
        assert len(instrumentor._active_swarms) == 0

    def test_response_attributes(self, instrumentor, tracer, mock_swarm_module):
        """Test that response attributes are captured."""
        instrumentor.instrument(tracer)

        swarm = MockSwarm()
        agent = MockAgent("assistant")

        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123"}

        result = swarm.run(agent, messages, context_variables=context)

        # Check response structure
        assert hasattr(result, "agent")
        assert hasattr(result, "messages")
        assert hasattr(result, "context_variables")
        assert result.agent.name == "final_agent"
        assert len(result.messages) == 2
        assert result.context_variables == context
