"""CrewAI instrumentation for Prela.

This module provides automatic instrumentation for CrewAI (>=0.30.0),
capturing crew executions, agent actions, and task completions.
"""

from __future__ import annotations

import functools
import uuid
from datetime import datetime
from typing import Any, Optional

from prela.core.span import SpanType
from prela.core.tracer import Tracer, get_tracer
from prela.instrumentation.base import Instrumentor
from prela.instrumentation.multi_agent.models import (
    AgentDefinition,
    AgentMessage,
    AgentRole,
    CrewExecution,
    MessageType,
    TaskAssignment,
    generate_agent_id,
)
from prela.license import require_tier


class CrewAIInstrumentor(Instrumentor):
    """Instrumentor for CrewAI multi-agent framework."""

    FRAMEWORK = "crewai"

    @property
    def is_instrumented(self) -> bool:
        """Check if CrewAI is currently instrumented."""
        return self._is_instrumented

    def __init__(self):
        super().__init__()
        self._active_crews: dict[str, CrewExecution] = {}
        self._is_instrumented = False
        self._tracer: Optional[Tracer] = None

    @require_tier("CrewAI instrumentation", "lunch-money")
    def instrument(self, tracer: Optional[Tracer] = None) -> None:
        """Patch CrewAI classes for tracing.

        Args:
            tracer: Optional tracer to use. If None, uses global tracer.
        """
        if self.is_instrumented:
            return

        try:
            import crewai
        except ImportError:
            return  # CrewAI not installed

        self._tracer = tracer or get_tracer()

        # Patch Crew.kickoff (main entry point)
        self._patch_crew_kickoff(crewai.Crew)

        # Patch Agent execution
        self._patch_agent_execute(crewai.Agent)

        # Patch Task execution
        self._patch_task_execute(crewai.Task)

        self._is_instrumented = True

    def uninstrument(self) -> None:
        """Restore original methods."""
        if not self.is_instrumented:
            return

        try:
            import crewai
        except ImportError:
            return

        # Restore all patched methods
        for module_name, obj_name, method_name in [
            ("crewai", "Crew", "kickoff"),
            ("crewai", "Agent", "execute_task"),
            ("crewai", "Task", "execute"),
        ]:
            try:
                module = __import__(module_name, fromlist=[obj_name])
                cls = getattr(module, obj_name, None)
                if cls and hasattr(cls, f"_prela_original_{method_name}"):
                    original = getattr(cls, f"_prela_original_{method_name}")
                    setattr(cls, method_name, original)
                    delattr(cls, f"_prela_original_{method_name}")
            except (ImportError, AttributeError):
                pass

        self._is_instrumented = False
        self._tracer = None

    def _patch_crew_kickoff(self, crew_cls) -> None:
        """Patch Crew.kickoff to create root span for crew execution."""
        if hasattr(crew_cls, "_prela_original_kickoff"):
            return

        original = crew_cls.kickoff
        crew_cls._prela_original_kickoff = original

        instrumentor = self

        @functools.wraps(original)
        def wrapped_kickoff(crew_self, *args, **kwargs):
            tracer = instrumentor._tracer
            if not tracer:
                return original(crew_self, *args, **kwargs)

            execution_id = str(uuid.uuid4())

            # Build agent definitions from crew
            agents = []
            for agent in crew_self.agents:
                agent_def = AgentDefinition(
                    agent_id=generate_agent_id(
                        instrumentor.FRAMEWORK, agent.role
                    ),
                    name=agent.role,
                    role=instrumentor._map_crewai_role(agent),
                    framework=instrumentor.FRAMEWORK,
                    model=(
                        getattr(agent.llm, "model_name", None)
                        if hasattr(agent, "llm")
                        else None
                    ),
                    system_prompt=getattr(agent, "backstory", None),
                    tools=[t.name for t in getattr(agent, "tools", [])],
                    metadata={
                        "goal": getattr(agent, "goal", None),
                        "allow_delegation": getattr(agent, "allow_delegation", False),
                    },
                )
                agents.append(agent_def)

            # Build task definitions
            tasks = []
            for task in crew_self.tasks:
                task_def = TaskAssignment(
                    task_id=str(uuid.uuid4()),
                    assigner_id="crew_manager",
                    assignee_id=(
                        generate_agent_id(instrumentor.FRAMEWORK, task.agent.role)
                        if task.agent
                        else "unassigned"
                    ),
                    description=task.description,
                    expected_output=getattr(task, "expected_output", None),
                )
                tasks.append(task_def)

            # Create crew execution record
            crew_exec = CrewExecution(
                execution_id=execution_id,
                framework=instrumentor.FRAMEWORK,
                agents=agents,
                tasks=tasks,
                started_at=datetime.utcnow(),
            )
            instrumentor._active_crews[execution_id] = crew_exec

            # Create root span
            crew_name = getattr(crew_self, "name", None) or "unnamed_crew"

            crew_attributes = {
                "crew.execution_id": execution_id,
                "crew.framework": instrumentor.FRAMEWORK,
                "crew.num_agents": len(agents),
                "crew.num_tasks": len(tasks),
                "crew.process": getattr(crew_self, "process", "sequential"),
                "crew.agent_names": [a.name for a in agents],
            }

            # NEW: Replay capture if enabled
            replay_capture = None
            if tracer.capture_for_replay:
                from prela.core.replay import ReplayCapture

                replay_capture = ReplayCapture()
                # Capture agent context
                replay_capture.set_agent_context(
                    system_prompt=f"Crew: {crew_name}",
                    available_tools=[
                        {"name": t, "agent": a.name}
                        for a in agents
                        for t in a.tools
                    ],
                    memory={
                        "agents": [
                            {
                                "name": a.name,
                                "role": a.role.value if hasattr(a.role, "value") else str(a.role),
                                "model": a.model,
                            }
                            for a in agents
                        ],
                        "tasks": [
                            {
                                "description": t.description[:200],
                                "assignee": t.assignee_id,
                            }
                            for t in tasks
                        ],
                    },
                    config={
                        "framework": instrumentor.FRAMEWORK,
                        "execution_id": execution_id,
                        "num_agents": len(agents),
                        "num_tasks": len(tasks),
                        "process": getattr(crew_self, "process", "sequential"),
                    },
                )

            with tracer.span(
                name=f"crewai.crew.{crew_name}",
                span_type=SpanType.AGENT,
                attributes=crew_attributes,
            ) as span:
                try:
                    result = original(crew_self, *args, **kwargs)
                    crew_exec.status = "completed"
                    crew_exec.completed_at = datetime.utcnow()
                    span.set_attribute(
                        "crew.result_length", len(str(result)) if result else 0
                    )
                    span.set_attribute("crew.total_llm_calls", crew_exec.total_llm_calls)
                    span.set_attribute("crew.total_tokens", crew_exec.total_tokens)
                    span.set_attribute("crew.total_cost_usd", crew_exec.total_cost_usd)

                    # NEW: Attach replay snapshot
                    if replay_capture:
                        try:
                            object.__setattr__(span, "replay_snapshot", replay_capture.build())
                        except Exception as e:
                            logger.debug(f"Failed to capture replay data: {e}")

                    return result
                except Exception as e:
                    crew_exec.status = "failed"
                    crew_exec.completed_at = datetime.utcnow()
                    span.add_event(
                        "exception",
                        attributes={
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                        },
                    )
                    raise
                finally:
                    if execution_id in instrumentor._active_crews:
                        del instrumentor._active_crews[execution_id]

        crew_cls.kickoff = wrapped_kickoff

    def _patch_agent_execute(self, agent_cls) -> None:
        """Patch Agent execution to create agent spans."""
        method_name = "execute_task"
        if not hasattr(agent_cls, method_name):
            return

        if hasattr(agent_cls, f"_prela_original_{method_name}"):
            return

        original = getattr(agent_cls, method_name)
        setattr(agent_cls, f"_prela_original_{method_name}", original)

        instrumentor = self

        @functools.wraps(original)
        def wrapped_execute(agent_self, task, *args, **kwargs):
            tracer = instrumentor._tracer
            if not tracer:
                return original(agent_self, task, *args, **kwargs)

            agent_id = generate_agent_id(instrumentor.FRAMEWORK, agent_self.role)

            task_desc = (
                task.description[:200] if hasattr(task, "description") else None
            )

            agent_attributes = {
                "agent.id": agent_id,
                "agent.name": agent_self.role,
                "agent.framework": instrumentor.FRAMEWORK,
                "agent.goal": getattr(agent_self, "goal", None),
                "agent.tools": [t.name for t in getattr(agent_self, "tools", [])],
                "task.description": task_desc,
            }

            with tracer.span(
                name=f"crewai.agent.{agent_self.role}",
                span_type=SpanType.AGENT,
                attributes=agent_attributes,
            ) as span:
                try:
                    result = original(agent_self, task, *args, **kwargs)
                    span.set_attribute(
                        "agent.output_length", len(str(result)) if result else 0
                    )
                    return result
                except Exception as e:
                    span.add_event(
                        "exception",
                        attributes={
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                        },
                    )
                    raise

        setattr(agent_cls, method_name, wrapped_execute)

    def _patch_task_execute(self, task_cls) -> None:
        """Patch Task execution to create task spans."""
        # Try multiple possible method names
        method_name = None
        for candidate in ["execute", "execute_sync", "_execute", "run"]:
            if hasattr(task_cls, candidate):
                method_name = candidate
                break

        if not method_name:
            return

        if hasattr(task_cls, f"_prela_original_{method_name}"):
            return

        original = getattr(task_cls, method_name)
        setattr(task_cls, f"_prela_original_{method_name}", original)

        instrumentor = self

        @functools.wraps(original)
        def wrapped_execute(task_self, *args, **kwargs):
            tracer = instrumentor._tracer
            if not tracer:
                return original(task_self, *args, **kwargs)

            task_id = str(uuid.uuid4())
            agent_name = (
                task_self.agent.role if hasattr(task_self, "agent") and task_self.agent else "unassigned"
            )

            # Truncate description for span name
            desc_preview = task_self.description[:50] if hasattr(task_self, "description") else "task"

            task_attributes = {
                "task.id": task_id,
                "task.description": getattr(task_self, "description", None),
                "task.expected_output": getattr(task_self, "expected_output", None),
                "task.agent": agent_name,
            }

            with tracer.span(
                name=f"crewai.task.{desc_preview}",
                span_type=SpanType.AGENT,
                attributes=task_attributes,
            ) as span:
                try:
                    result = original(task_self, *args, **kwargs)
                    # Truncate output to 1000 chars
                    span.set_attribute(
                        "task.output", str(result)[:1000] if result else None
                    )
                    span.set_attribute("task.status", "completed")
                    return result
                except Exception as e:
                    span.set_attribute("task.status", "failed")
                    span.add_event(
                        "exception",
                        attributes={
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                        },
                    )
                    raise

        setattr(task_cls, method_name, wrapped_execute)

    def _map_crewai_role(self, agent) -> AgentRole:
        """Map CrewAI agent to standard role.

        Args:
            agent: CrewAI agent object

        Returns:
            Standardized AgentRole enum value
        """
        role_lower = agent.role.lower()
        if "manager" in role_lower or "lead" in role_lower:
            return AgentRole.MANAGER
        elif "critic" in role_lower or "review" in role_lower:
            return AgentRole.CRITIC
        elif "specialist" in role_lower or "expert" in role_lower:
            return AgentRole.SPECIALIST
        return AgentRole.WORKER
