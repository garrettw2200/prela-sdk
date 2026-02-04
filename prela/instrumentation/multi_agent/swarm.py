"""Swarm instrumentation for Prela.

This module provides automatic instrumentation for OpenAI Swarm multi-agent framework,
capturing agent orchestration, handoffs, and execution flow.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Optional

from prela.core.span import SpanType
from prela.core.tracer import Tracer, get_tracer
from prela.instrumentation.base import Instrumentor
from prela.instrumentation.multi_agent.models import generate_agent_id
from prela.license import require_tier


class SwarmInstrumentor(Instrumentor):
    """Instrumentor for OpenAI Swarm multi-agent framework."""

    FRAMEWORK = "swarm"

    @property
    def is_instrumented(self) -> bool:
        """Check if Swarm is currently instrumented."""
        return self._is_instrumented

    def __init__(self):
        super().__init__()
        self._active_swarms: dict[str, dict] = {}
        self._is_instrumented = False
        self._tracer: Optional[Tracer] = None

    @require_tier("Swarm instrumentation", "lunch-money")
    def instrument(self, tracer: Optional[Tracer] = None) -> None:
        """Patch Swarm classes for tracing.

        Args:
            tracer: Optional tracer to use. If None, uses global tracer.
        """
        if self.is_instrumented:
            return

        try:
            from swarm import Agent, Swarm
        except ImportError:
            return  # Swarm not installed

        self._tracer = tracer or get_tracer()

        # Patch Swarm.run (main execution entry point)
        if hasattr(Swarm, "run"):
            self._patch_swarm_run(Swarm)

        # Patch Agent.__init__ to track agents
        if hasattr(Agent, "__init__"):
            self._patch_agent_init(Agent)

        self._is_instrumented = True

    def uninstrument(self) -> None:
        """Restore original methods."""
        if not self.is_instrumented:
            return

        try:
            from swarm import Agent, Swarm
        except ImportError:
            return

        # Restore all patched methods
        for cls, method_name in [
            (Swarm, "run"),
            (Agent, "__init__"),
        ]:
            if hasattr(cls, f"_prela_original_{method_name}"):
                original = getattr(cls, f"_prela_original_{method_name}")
                setattr(cls, method_name, original)
                delattr(cls, f"_prela_original_{method_name}")

        self._is_instrumented = False
        self._tracer = None

    def _patch_swarm_run(self, swarm_cls) -> None:
        """Patch Swarm.run to create execution spans."""
        if hasattr(swarm_cls, "_prela_original_run"):
            return

        original_run = swarm_cls.run
        swarm_cls._prela_original_run = original_run

        instrumentor = self

        @functools.wraps(original_run)
        def wrapped_run(swarm_self, agent, messages, *args, **kwargs):
            tracer = instrumentor._tracer
            if not tracer:
                return original_run(swarm_self, agent, messages, *args, **kwargs)

            execution_id = str(uuid.uuid4())
            instrumentor._active_swarms[execution_id] = {
                "agents_seen": set(),
                "handoffs": [],
                "total_tokens": 0,
            }

            # Get initial agent name
            initial_agent_name = agent.name if hasattr(agent, "name") else "unnamed"

            # Extract context variables
            context_vars = kwargs.get("context_variables", {})
            context_keys = list(context_vars.keys()) if context_vars else []

            swarm_attributes = {
                "swarm.execution_id": execution_id,
                "swarm.framework": instrumentor.FRAMEWORK,
                "swarm.initial_agent": initial_agent_name,
                "swarm.num_messages": len(messages) if messages else 0,
                "swarm.context_variables": context_keys,
            }

            # NEW: Replay capture if enabled
            replay_capture = None
            if tracer.capture_for_replay:
                from prela.core.replay import ReplayCapture

                replay_capture = ReplayCapture()
                # Capture agent context
                replay_capture.set_agent_context(
                    system_prompt=getattr(agent, "instructions", None),
                    available_tools=[
                        {"name": func.__name__, "description": func.__doc__ or ""}
                        for func in getattr(agent, "functions", [])
                    ],
                    config={
                        "framework": instrumentor.FRAMEWORK,
                        "execution_id": execution_id,
                        "agent_name": initial_agent_name,
                    },
                    memory={"context_variables": context_keys},
                )

            with tracer.span(
                name=f"swarm.run.{initial_agent_name}",
                span_type=SpanType.AGENT,
                attributes=swarm_attributes,
            ) as span:
                try:
                    result = original_run(swarm_self, agent, messages, *args, **kwargs)

                    # Add execution statistics
                    swarm_state = instrumentor._active_swarms.get(execution_id, {})
                    agents_used = list(swarm_state.get("agents_seen", set()))
                    if agents_used:
                        span.set_attribute("swarm.agents_used", agents_used)

                    handoffs = swarm_state.get("handoffs", [])
                    span.set_attribute("swarm.num_handoffs", len(handoffs))

                    # Capture final agent
                    if hasattr(result, "agent"):
                        final_agent = (
                            result.agent.name
                            if hasattr(result.agent, "name")
                            else "unnamed"
                        )
                        span.set_attribute("swarm.final_agent", final_agent)

                    # Capture response messages
                    if hasattr(result, "messages"):
                        span.set_attribute(
                            "swarm.response_messages", len(result.messages)
                        )

                    # Capture context updates
                    if hasattr(result, "context_variables"):
                        updated_keys = list(result.context_variables.keys())
                        span.set_attribute("swarm.updated_context", updated_keys)

                    # NEW: Attach replay snapshot
                    if replay_capture:
                        try:
                            object.__setattr__(span, "replay_snapshot", replay_capture.build())
                        except Exception as e:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.debug(f"Failed to capture replay data: {e}")

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
                finally:
                    if execution_id in instrumentor._active_swarms:
                        del instrumentor._active_swarms[execution_id]

        swarm_cls.run = wrapped_run

    def _patch_agent_init(self, agent_cls) -> None:
        """Patch Agent.__init__ to track agent creation."""
        if hasattr(agent_cls, "_prela_original___init__"):
            return

        original_init = agent_cls.__init__
        agent_cls._prela_original___init__ = original_init

        instrumentor = self

        @functools.wraps(original_init)
        def wrapped_init(agent_self, *args, **kwargs):
            result = original_init(agent_self, *args, **kwargs)

            # Generate agent ID for tracking
            agent_name = (
                agent_self.name if hasattr(agent_self, "name") else str(id(agent_self))
            )
            agent_self._prela_agent_id = generate_agent_id(
                instrumentor.FRAMEWORK, agent_name
            )

            return result

        agent_cls.__init__ = wrapped_init
