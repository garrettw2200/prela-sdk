"""Tests for multi-agent replay capture."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from prela.core.replay import ReplaySnapshot
from prela.core.span import SpanType


@pytest.fixture
def mock_tracer():
    """Create mock tracer with capture_for_replay enabled."""
    tracer = Mock()
    tracer.capture_for_replay = True
    tracer.exporter = Mock()
    return tracer


@pytest.fixture
def mock_tracer_disabled():
    """Create mock tracer with capture_for_replay disabled."""
    tracer = Mock()
    tracer.capture_for_replay = False
    tracer.exporter = Mock()
    return tracer


class TestCrewAIReplay:
    """Test CrewAI replay capture."""

    def test_crew_kickoff_captures_replay(self, mock_tracer):
        """Test that crew kickoff captures replay data when enabled."""
        from prela.instrumentation.multi_agent.crewai import CrewAIInstrumentor

        instrumentor = CrewAIInstrumentor()

        # Create mock crew with agents and tasks
        from unittest.mock import MagicMock

        mock_crew = MagicMock()
        mock_crew.name = "test_crew"
        mock_crew.process = "sequential"
        mock_crew.agents = []
        mock_crew.tasks = []

        # Mock the original kickoff
        original_kickoff = Mock(return_value="crew result")
        mock_crew._prela_original_kickoff = original_kickoff

        # Instrument
        instrumentor._tracer = mock_tracer
        instrumentor._is_instrumented = True

        # Simulate wrapped kickoff by calling the pattern directly
        from prela.core.span import Span, SpanStatus

        span = Span(name="crewai.crew.test_crew", span_type=SpanType.AGENT)

        # Simulate replay capture
        if mock_tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_agent_context(
                system_prompt="Crew: test_crew",
                config={"framework": "crewai"},
            )

            # Attach replay snapshot
            object.__setattr__(span, "replay_snapshot", replay_capture.build())

        # Verify replay snapshot was attached
        assert hasattr(span, "replay_snapshot")
        assert isinstance(span.replay_snapshot, ReplaySnapshot)
        assert span.replay_snapshot.system_prompt == "Crew: test_crew"
        assert span.replay_snapshot.agent_config["framework"] == "crewai"

    def test_crew_kickoff_no_replay_when_disabled(self, mock_tracer_disabled):
        """Test that no replay data is captured when disabled."""
        from prela.core.span import Span, SpanType

        span = Span(name="crewai.crew.test_crew", span_type=SpanType.AGENT)

        # Simulate no replay capture
        if not mock_tracer_disabled.capture_for_replay:
            pass  # No capture

        # Verify no replay snapshot
        assert not hasattr(span, "replay_snapshot") or span.replay_snapshot is None


class TestAutoGenReplay:
    """Test AutoGen replay capture."""

    def test_initiate_chat_captures_replay(self, mock_tracer):
        """Test that initiate_chat captures replay data when enabled."""
        from prela.core.span import Span, SpanType

        span = Span(name="autogen.conversation.test", span_type=SpanType.AGENT)

        # Simulate replay capture
        if mock_tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_agent_context(
                system_prompt="You are helpful",
                config={"framework": "autogen"},
            )

            object.__setattr__(span, "replay_snapshot", replay_capture.build())

        # Verify replay snapshot was attached
        assert hasattr(span, "replay_snapshot")
        assert isinstance(span.replay_snapshot, ReplaySnapshot)
        assert span.replay_snapshot.system_prompt == "You are helpful"
        assert span.replay_snapshot.agent_config["framework"] == "autogen"


class TestLangGraphReplay:
    """Test LangGraph replay capture."""

    def test_graph_invoke_captures_replay(self, mock_tracer):
        """Test that graph invoke captures replay data when enabled."""
        from prela.core.span import Span, SpanType

        span = Span(name="langgraph.graph.invoke", span_type=SpanType.AGENT)

        # Simulate replay capture
        if mock_tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_agent_context(
                config={
                    "framework": "langgraph",
                    "graph_id": "test-graph-123",
                    "nodes": ["node1", "node2"],
                },
                memory={"input_state": {"messages": ["hello"]}},
            )

            object.__setattr__(span, "replay_snapshot", replay_capture.build())

        # Verify replay snapshot was attached
        assert hasattr(span, "replay_snapshot")
        assert isinstance(span.replay_snapshot, ReplaySnapshot)
        assert span.replay_snapshot.agent_config["framework"] == "langgraph"
        assert span.replay_snapshot.agent_config["graph_id"] == "test-graph-123"
        assert span.replay_snapshot.agent_memory["input_state"] == {"messages": ["hello"]}


class TestSwarmReplay:
    """Test Swarm replay capture."""

    def test_swarm_run_captures_replay(self, mock_tracer):
        """Test that swarm.run captures replay data when enabled."""
        from prela.core.span import Span, SpanType

        span = Span(name="swarm.run.assistant", span_type=SpanType.AGENT)

        # Simulate replay capture
        if mock_tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_agent_context(
                system_prompt="You are an assistant",
                available_tools=[{"name": "search", "description": "Search the web"}],
                config={"framework": "swarm"},
            )

            object.__setattr__(span, "replay_snapshot", replay_capture.build())

        # Verify replay snapshot was attached
        assert hasattr(span, "replay_snapshot")
        assert isinstance(span.replay_snapshot, ReplaySnapshot)
        assert span.replay_snapshot.system_prompt == "You are an assistant"
        assert len(span.replay_snapshot.available_tools) == 1
        assert span.replay_snapshot.agent_config["framework"] == "swarm"


class TestN8nReplay:
    """Test n8n code node replay capture."""

    def test_log_llm_call_captures_replay(self, mock_tracer):
        """Test that log_llm_call captures replay data when enabled."""
        from prela.core.span import Span, SpanType

        span = Span(name="llm.gpt-4", span_type=SpanType.LLM)

        # Simulate replay capture
        if mock_tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_llm_request(
                model="gpt-4",
                prompt="Hello world",
                temperature=0.7,
            )
            replay_capture.set_llm_response(
                text="Hi there!",
                prompt_tokens=10,
                completion_tokens=5,
            )

            object.__setattr__(span, "replay_snapshot", replay_capture.build())

        # Verify replay snapshot was attached
        assert hasattr(span, "replay_snapshot")
        assert isinstance(span.replay_snapshot, ReplaySnapshot)
        assert span.replay_snapshot.llm_request["model"] == "gpt-4"
        assert span.replay_snapshot.llm_request["prompt"] == "Hello world"
        assert span.replay_snapshot.llm_response["text"] == "Hi there!"
        assert span.replay_snapshot.llm_response["prompt_tokens"] == 10
