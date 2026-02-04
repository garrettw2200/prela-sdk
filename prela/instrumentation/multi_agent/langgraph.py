"""LangGraph instrumentation for Prela.

This module provides automatic instrumentation for LangGraph (>=0.0.20),
capturing stateful agent workflows with node and edge tracking.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Callable, Optional

from prela.core.span import SpanStatus, SpanType
from prela.core.tracer import Tracer, get_tracer
from prela.instrumentation.base import Instrumentor
from prela.instrumentation.multi_agent.models import generate_agent_id
from prela.license import require_tier


class LangGraphInstrumentor(Instrumentor):
    """Instrumentor for LangGraph stateful agent workflows."""

    FRAMEWORK = "langgraph"

    @property
    def is_instrumented(self) -> bool:
        """Check if LangGraph is currently instrumented."""
        return self._is_instrumented

    def __init__(self):
        super().__init__()
        self._original_methods: dict[tuple, Any] = {}
        self._graph_metadata: dict[int, dict] = {}
        self._is_instrumented = False
        self._tracer: Optional[Tracer] = None

    @require_tier("LangGraph instrumentation", "lunch-money")
    def instrument(self, tracer: Optional[Tracer] = None) -> None:
        """Patch LangGraph classes for tracing.

        Args:
            tracer: Optional tracer to use. If None, uses global tracer.
        """
        if self.is_instrumented:
            return

        try:
            from langgraph.graph import StateGraph
        except ImportError:
            return  # LangGraph not installed

        self._tracer = tracer or get_tracer()

        self._patch_state_graph(StateGraph)

        try:
            from langgraph.prebuilt import create_react_agent

            self._patch_create_react_agent(create_react_agent)
        except ImportError:
            pass

        self._is_instrumented = True

    def uninstrument(self) -> None:
        """Restore original methods."""
        if not self.is_instrumented:
            return

        for (cls, method_name), original in list(self._original_methods.items()):
            if callable(cls):
                # Function-level patching (e.g., create_react_agent)
                try:
                    import langgraph.prebuilt as prebuilt

                    setattr(prebuilt, method_name, original)
                except ImportError:
                    pass
            else:
                # Class method patching
                setattr(cls, method_name, original)

        self._original_methods.clear()
        self._graph_metadata.clear()
        self._is_instrumented = False
        self._tracer = None

    def _patch_state_graph(self, graph_cls) -> None:
        """Patch StateGraph for node and edge tracking."""
        if (graph_cls, "compile") in self._original_methods:
            return

        original_compile = graph_cls.compile
        self._original_methods[(graph_cls, "compile")] = original_compile

        instrumentor = self

        @functools.wraps(original_compile)
        def wrapped_compile(graph_self, *args, **kwargs):
            compiled = original_compile(graph_self, *args, **kwargs)

            graph_id = str(uuid.uuid4())
            instrumentor._graph_metadata[id(compiled)] = {
                "graph_id": graph_id,
                "nodes": (
                    list(graph_self.nodes.keys())
                    if hasattr(graph_self, "nodes")
                    else []
                ),
            }

            instrumentor._patch_compiled_graph(compiled, graph_id)
            return compiled

        graph_cls.compile = wrapped_compile

        # Patch add_node
        if (graph_cls, "add_node") not in self._original_methods:
            original_add_node = graph_cls.add_node
            self._original_methods[(graph_cls, "add_node")] = original_add_node

            @functools.wraps(original_add_node)
            def wrapped_add_node(
                graph_self, node_name: str, action: Callable, *args, **kwargs
            ):
                wrapped_action = instrumentor._wrap_node_action(node_name, action)
                return original_add_node(
                    graph_self, node_name, wrapped_action, *args, **kwargs
                )

            graph_cls.add_node = wrapped_add_node

    def _patch_compiled_graph(self, compiled_graph, graph_id: str) -> None:
        """Patch a compiled graph's invoke and stream methods."""
        # Patch invoke
        if hasattr(compiled_graph, "invoke"):
            original_invoke = compiled_graph.invoke
            instrumentor = self

            @functools.wraps(original_invoke)
            def wrapped_invoke(state, *args, **kwargs):
                tracer = instrumentor._tracer
                if not tracer:
                    return original_invoke(state, *args, **kwargs)

                metadata = instrumentor._graph_metadata.get(id(compiled_graph), {})

                # NEW: Replay capture if enabled
                replay_capture = None
                if tracer.capture_for_replay:
                    from prela.core.replay import ReplayCapture

                    replay_capture = ReplayCapture()
                    # Capture agent context
                    replay_capture.set_agent_context(
                        config={
                            "framework": instrumentor.FRAMEWORK,
                            "graph_id": graph_id,
                            "nodes": metadata.get("nodes", []),
                        },
                        memory={"input_state": state if isinstance(state, dict) else str(state)[:500]},
                    )

                with tracer.span(
                    name="langgraph.graph.invoke",
                    span_type=SpanType.AGENT,
                    attributes={
                        "graph.id": graph_id,
                        "graph.framework": instrumentor.FRAMEWORK,
                        "graph.nodes": metadata.get("nodes", []),
                        "graph.input_keys": (
                            list(state.keys()) if isinstance(state, dict) else None
                        ),
                    },
                ) as span:
                    try:
                        result = original_invoke(state, *args, **kwargs)
                        if isinstance(result, dict):
                            span.set_attribute("graph.output_keys", list(result.keys()))
                        span.set_status(SpanStatus.SUCCESS)

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
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise

            compiled_graph.invoke = wrapped_invoke

        # Patch stream
        if hasattr(compiled_graph, "stream"):
            original_stream = compiled_graph.stream
            instrumentor = self

            @functools.wraps(original_stream)
            def wrapped_stream(state, *args, **kwargs):
                tracer = instrumentor._tracer
                if not tracer:
                    yield from original_stream(state, *args, **kwargs)
                    return

                with tracer.span(
                    name="langgraph.graph.stream",
                    span_type=SpanType.AGENT,
                    attributes={
                        "graph.id": graph_id,
                        "graph.framework": instrumentor.FRAMEWORK,
                        "graph.streaming": True,
                    },
                ) as span:
                    step_count = 0
                    try:
                        for step in original_stream(state, *args, **kwargs):
                            step_count += 1
                            span.add_event(
                                "graph.step", {"step.number": step_count}
                            )
                            yield step
                        span.set_attribute("graph.total_steps", step_count)
                        span.set_status(SpanStatus.SUCCESS)
                    except Exception as e:
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise

            compiled_graph.stream = wrapped_stream

    def _wrap_node_action(self, node_name: str, action: Callable) -> Callable:
        """Wrap a node action to add tracing."""
        instrumentor = self

        @functools.wraps(action)
        def wrapped_action(state, *args, **kwargs):
            tracer = instrumentor._tracer
            if not tracer:
                return action(state, *args, **kwargs)

            with tracer.span(
                name=f"langgraph.node.{node_name}",
                span_type=SpanType.CUSTOM,
                attributes={
                    "node.name": node_name,
                    "node.framework": instrumentor.FRAMEWORK,
                },
            ) as span:
                try:
                    result = action(state, *args, **kwargs)
                    if isinstance(state, dict) and isinstance(result, dict):
                        changed = [
                            k
                            for k in result
                            if k not in state or state.get(k) != result.get(k)
                        ]
                        span.set_attribute("node.changed_keys", changed)
                    span.set_status(SpanStatus.SUCCESS)
                    return result
                except Exception as e:
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return wrapped_action

    def _patch_create_react_agent(self, create_func: Callable) -> None:
        """Patch the prebuilt create_react_agent function."""
        import langgraph.prebuilt as prebuilt

        if (create_func, "create_react_agent") in self._original_methods:
            return

        self._original_methods[(create_func, "create_react_agent")] = create_func

        instrumentor = self

        @functools.wraps(create_func)
        def wrapped_create_react_agent(model, tools, *args, **kwargs):
            agent = create_func(model, tools, *args, **kwargs)
            agent_id = str(uuid.uuid4())
            # Update existing metadata or create new entry
            existing_metadata = instrumentor._graph_metadata.get(id(agent), {})
            existing_metadata.update({
                "agent_id": agent_id,
                "agent_type": "react",
                "tools": [t.name if hasattr(t, "name") else str(t) for t in tools],
            })
            instrumentor._graph_metadata[id(agent)] = existing_metadata
            # Don't call _patch_compiled_graph again as it was already called by compile()
            return agent

        prebuilt.create_react_agent = wrapped_create_react_agent
