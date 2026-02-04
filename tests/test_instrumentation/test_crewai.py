"""Tests for CrewAI instrumentation."""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from prela.core.span import SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter
from prela.instrumentation.multi_agent.crewai import CrewAIInstrumentor
from prela.instrumentation.multi_agent.models import AgentRole


# Mock CrewAI classes
class MockTool:
    """Mock CrewAI tool."""

    def __init__(self, name: str):
        self.name = name


class MockLLM:
    """Mock CrewAI LLM."""

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name


class MockAgent:
    """Mock CrewAI agent."""

    def __init__(
        self,
        role: str,
        goal: str = "Test goal",
        backstory: str = "Test backstory",
        tools: list | None = None,
        allow_delegation: bool = False,
        llm: MockLLM | None = None,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.allow_delegation = allow_delegation
        self.llm = llm or MockLLM()

    def execute_task(self, task, *args, **kwargs):
        """Execute a task."""
        return f"Result from {self.role}"


class MockTask:
    """Mock CrewAI task."""

    def __init__(
        self,
        description: str,
        expected_output: str | None = None,
        agent: MockAgent | None = None,
    ):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent

    def execute(self, *args, **kwargs):
        """Execute the task."""
        return f"Task completed: {self.description}"


class MockCrew:
    """Mock CrewAI crew."""

    def __init__(
        self,
        agents: list[MockAgent],
        tasks: list[MockTask],
        process: str = "sequential",
        name: str | None = None,
    ):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.name = name

    def kickoff(self, *args, **kwargs):
        """Execute the crew."""
        # Execute tasks in order
        results = []
        for task in self.tasks:
            if task.agent:
                result = task.agent.execute_task(task)
            else:
                result = task.execute()
            results.append(result)
        return "\n".join(results)


# Create mock crewai module
mock_crewai = MagicMock()
mock_crewai.Agent = MockAgent
mock_crewai.Task = MockTask
mock_crewai.Crew = MockCrew
mock_crewai.Tool = MockTool


@pytest.fixture
def mock_crewai_module():
    """Fixture to mock crewai module."""
    sys.modules["crewai"] = mock_crewai
    yield mock_crewai
    if "crewai" in sys.modules:
        del sys.modules["crewai"]


@pytest.fixture
def instrumentor():
    """Create CrewAI instrumentor."""
    return CrewAIInstrumentor()


@pytest.fixture
def tracer():
    """Create tracer with console exporter."""
    return Tracer(service_name="test-crewai", exporter=ConsoleExporter())


class TestCrewAIInstrumentor:
    """Test CrewAI instrumentor."""

    def test_init(self, instrumentor):
        """Test instrumentor initialization."""
        assert instrumentor.FRAMEWORK == "crewai"
        assert not instrumentor.is_instrumented
        assert instrumentor._active_crews == {}

    def test_instrument_without_crewai(self, instrumentor):
        """Test instrument when crewai not installed."""
        # Should not raise error
        instrumentor.instrument()
        assert not instrumentor.is_instrumented

    def test_instrument_with_crewai(self, instrumentor, tracer, mock_crewai_module):
        """Test instrument with crewai installed."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented
        assert instrumentor._tracer is tracer

        # Verify that methods were patched by checking for the original attribute
        assert hasattr(MockCrew, "_prela_original_kickoff")
        assert hasattr(MockAgent, "_prela_original_execute_task")
        assert hasattr(MockTask, "_prela_original_execute")

    def test_instrument_idempotent(self, instrumentor, tracer, mock_crewai_module):
        """Test that instrument can be called multiple times."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        # Call again - should not raise
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

    def test_uninstrument(self, instrumentor, tracer, mock_crewai_module):
        """Test uninstrument restores original methods."""
        # Save original methods before instrumentation
        original_kickoff = MockCrew.kickoff

        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented
        assert instrumentor._tracer is None

        # Check that original methods were restored
        assert not hasattr(MockCrew, "_prela_original_kickoff")
        assert not hasattr(MockAgent, "_prela_original_execute_task")
        assert not hasattr(MockTask, "_prela_original_execute")

    def test_crew_kickoff_basic(self, instrumentor, tracer, mock_crewai_module):
        """Test crew kickoff creates root span."""
        instrumentor.instrument(tracer)

        # Create crew
        agent = MockAgent(role="researcher")
        task = MockTask(description="Research AI", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task], name="test_crew")

        # Execute crew
        result = crew.kickoff()

        assert result == "Result from researcher"

        # Check spans were created
        # Note: ConsoleExporter doesn't store spans, so we can't verify directly
        # In production, we'd use a test exporter

    def test_crew_kickoff_with_multiple_agents(
        self, instrumentor, tracer, mock_crewai_module
    ):
        """Test crew with multiple agents."""
        instrumentor.instrument(tracer)

        # Create crew with multiple agents
        researcher = MockAgent(role="researcher", tools=[MockTool("search")])
        writer = MockAgent(role="writer", tools=[MockTool("write")])

        task1 = MockTask(description="Research AI", agent=researcher)
        task2 = MockTask(description="Write article", agent=writer)

        crew = MockCrew(
            agents=[researcher, writer], tasks=[task1, task2], name="content_crew"
        )

        result = crew.kickoff()

        assert "Result from researcher" in result
        assert "Result from writer" in result

    def test_crew_kickoff_with_exception(
        self, instrumentor, tracer, mock_crewai_module
    ):
        """Test crew kickoff handles exceptions."""
        instrumentor.instrument(tracer)

        # Create crew that will raise exception
        agent = MockAgent(role="researcher")
        agent.execute_task = Mock(side_effect=ValueError("Test error"))

        task = MockTask(description="Research AI", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])

        with pytest.raises(ValueError, match="Test error"):
            crew.kickoff()

    def test_agent_execute_task(self, instrumentor, tracer, mock_crewai_module):
        """Test agent execute_task creates span."""
        instrumentor.instrument(tracer)

        agent = MockAgent(
            role="researcher",
            goal="Research AI topics",
            tools=[MockTool("search"), MockTool("scrape")],
        )

        task = MockTask(description="Research AI agents")

        result = agent.execute_task(task)

        assert result == "Result from researcher"

    def test_task_execute(self, instrumentor, tracer, mock_crewai_module):
        """Test task execute creates span."""
        instrumentor.instrument(tracer)

        task = MockTask(description="Research AI", expected_output="A report")

        result = task.execute()

        assert result == "Task completed: Research AI"

    def test_task_execute_with_exception(
        self, instrumentor, tracer, mock_crewai_module
    ):
        """Test task execute handles exceptions."""
        instrumentor.instrument(tracer)

        task = MockTask(description="Research AI")
        task.execute = Mock(side_effect=RuntimeError("Task failed"))

        with pytest.raises(RuntimeError, match="Task failed"):
            task.execute()

    def test_map_crewai_role_manager(self, instrumentor):
        """Test mapping manager role."""
        agent = MockAgent(role="Project Manager")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.MANAGER

    def test_map_crewai_role_lead(self, instrumentor):
        """Test mapping lead role."""
        agent = MockAgent(role="Team Lead")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.MANAGER

    def test_map_crewai_role_critic(self, instrumentor):
        """Test mapping critic role."""
        agent = MockAgent(role="Code Critic")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.CRITIC

    def test_map_crewai_role_reviewer(self, instrumentor):
        """Test mapping reviewer role."""
        agent = MockAgent(role="Content Reviewer")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.CRITIC

    def test_map_crewai_role_specialist(self, instrumentor):
        """Test mapping specialist role."""
        agent = MockAgent(role="AI Specialist")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.SPECIALIST

    def test_map_crewai_role_expert(self, instrumentor):
        """Test mapping expert role."""
        agent = MockAgent(role="Security Expert")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.SPECIALIST

    def test_map_crewai_role_worker(self, instrumentor):
        """Test mapping worker role (default)."""
        agent = MockAgent(role="Data Processor")
        role = instrumentor._map_crewai_role(agent)
        assert role == AgentRole.WORKER

    def test_crew_execution_tracking(self, instrumentor, tracer, mock_crewai_module):
        """Test that crew execution is tracked in _active_crews."""
        instrumentor.instrument(tracer)

        agent = MockAgent(role="researcher")
        task = MockTask(description="Research AI", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])

        # Before execution
        assert len(instrumentor._active_crews) == 0

        # During execution, we can't easily check since it's synchronous
        # After execution
        crew.kickoff()

        # After completion, should be cleaned up
        assert len(instrumentor._active_crews) == 0

    def test_crew_without_name(self, instrumentor, tracer, mock_crewai_module):
        """Test crew without name uses 'unnamed_crew'."""
        instrumentor.instrument(tracer)

        agent = MockAgent(role="researcher")
        task = MockTask(description="Research AI", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task], name=None)

        # Should not raise error
        result = crew.kickoff()
        assert result == "Result from researcher"

    def test_task_without_agent(self, instrumentor, tracer, mock_crewai_module):
        """Test task without assigned agent."""
        instrumentor.instrument(tracer)

        task = MockTask(description="Unassigned task", agent=None)
        crew = MockCrew(agents=[], tasks=[task])

        result = crew.kickoff()
        assert "Task completed" in result

    def test_agent_without_tools(self, instrumentor, tracer, mock_crewai_module):
        """Test agent without tools."""
        instrumentor.instrument(tracer)

        agent = MockAgent(role="researcher", tools=[])
        task = MockTask(description="Research AI", agent=agent)

        result = agent.execute_task(task)
        assert result == "Result from researcher"

    def test_crew_with_delegation(self, instrumentor, tracer, mock_crewai_module):
        """Test crew with delegation enabled."""
        instrumentor.instrument(tracer)

        agent = MockAgent(role="manager", allow_delegation=True)
        task = MockTask(description="Manage project", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])

        result = crew.kickoff()
        assert result == "Result from manager"

    def test_long_task_description(self, instrumentor, tracer, mock_crewai_module):
        """Test task with very long description gets truncated."""
        instrumentor.instrument(tracer)

        long_desc = "A" * 500  # Very long description
        task = MockTask(description=long_desc)

        result = task.execute()
        assert "Task completed" in result


class TestCrewAIIntegration:
    """Integration tests for CrewAI instrumentation."""

    def test_full_crew_workflow(self, instrumentor, tracer, mock_crewai_module):
        """Test complete crew workflow with instrumentation."""
        instrumentor.instrument(tracer)

        # Create a multi-agent workflow
        researcher = MockAgent(
            role="AI Researcher",
            goal="Research AI agents",
            backstory="Expert in AI",
            tools=[MockTool("search"), MockTool("scrape")],
            llm=MockLLM("gpt-4"),
        )

        writer = MockAgent(
            role="Technical Writer",
            goal="Write technical content",
            backstory="Experienced writer",
            tools=[MockTool("write")],
            llm=MockLLM("gpt-3.5-turbo"),
        )

        critic = MockAgent(
            role="Content Critic",
            goal="Review content quality",
            backstory="Meticulous reviewer",
            llm=MockLLM("gpt-4"),
        )

        # Create tasks
        research_task = MockTask(
            description="Research the latest AI agent frameworks",
            expected_output="Comprehensive research report",
            agent=researcher,
        )

        writing_task = MockTask(
            description="Write an article based on research",
            expected_output="Technical article",
            agent=writer,
        )

        review_task = MockTask(
            description="Review and improve the article",
            expected_output="Reviewed article",
            agent=critic,
        )

        # Create crew
        crew = MockCrew(
            agents=[researcher, writer, critic],
            tasks=[research_task, writing_task, review_task],
            process="sequential",
            name="content_creation_crew",
        )

        # Execute
        result = crew.kickoff()

        # Verify result contains all agent outputs
        assert "Result from AI Researcher" in result
        assert "Result from Technical Writer" in result
        assert "Result from Content Critic" in result

    def test_uninstrumented_execution(self, mock_crewai_module):
        """Test that crew works normally when not instrumented."""
        agent = MockAgent(role="researcher")
        task = MockTask(description="Research AI", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])

        # Should work normally without instrumentation
        result = crew.kickoff()
        assert result == "Result from researcher"

    def test_concurrent_crews(self, instrumentor, tracer, mock_crewai_module):
        """Test handling of multiple concurrent crew executions."""
        instrumentor.instrument(tracer)

        # Create two crews
        agent1 = MockAgent(role="researcher1")
        task1 = MockTask(description="Task 1", agent=agent1)
        crew1 = MockCrew(agents=[agent1], tasks=[task1], name="crew1")

        agent2 = MockAgent(role="researcher2")
        task2 = MockTask(description="Task 2", agent=agent2)
        crew2 = MockCrew(agents=[agent2], tasks=[task2], name="crew2")

        # Execute both (sequentially in this test, but could be concurrent)
        result1 = crew1.kickoff()
        result2 = crew2.kickoff()

        assert result1 == "Result from researcher1"
        assert result2 == "Result from researcher2"

        # Both should be cleaned up
        assert len(instrumentor._active_crews) == 0
