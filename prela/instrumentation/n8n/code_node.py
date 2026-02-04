"""
Helper utilities for tracing Python code executed within n8n Code nodes.

This module provides context managers and decorators that enable users to
instrument custom Python code running inside n8n Code nodes, creating
properly nested spans that integrate with the n8n workflow execution.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

from prela.core.clock import now
from prela.core.span import Span, SpanStatus, SpanType
from prela.core.tracer import Tracer, get_tracer

logger = logging.getLogger(__name__)


class PrelaN8nContext:
    """
    Context manager for tracing custom Python code within n8n Code nodes.

    This class creates a workflow-level span and node-level span that properly
    integrate with Prela's tracing infrastructure. It provides helper methods
    for logging LLM calls, tool calls, and retrieval operations.

    Example:
        ```python
        # Inside n8n Code node
        from prela.instrumentation.n8n import PrelaN8nContext

        ctx = PrelaN8nContext(
            workflow_id=$workflow.id,
            workflow_name=$workflow.name,
            execution_id=$execution.id,
            node_name=$node.name
        )

        with ctx:
            # Your custom code
            response = call_my_llm(prompt)
            ctx.log_llm_call(
                model="gpt-4",
                prompt=prompt,
                response=response,
                tokens={"prompt": 100, "completion": 50}
            )

            return [{"json": {"result": response}}]
        ```
    """

    def __init__(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_id: str,
        node_name: str,
        node_type: str = "n8n-nodes-base.code",
        tracer: Optional[Tracer] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """
        Initialize n8n Code node tracing context.

        Args:
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable workflow name
            execution_id: Unique execution identifier
            node_name: Name of the Code node
            node_type: Node type identifier (default: n8n-nodes-base.code)
            tracer: Prela tracer instance (defaults to global tracer)
            api_key: Optional API key for remote export
            endpoint: Optional endpoint URL for remote export
        """
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.execution_id = execution_id
        self.node_name = node_name
        self.node_type = node_type
        self.api_key = api_key
        self.endpoint = endpoint

        # Get or initialize tracer
        if tracer is None:
            self.tracer = get_tracer()
            if self.tracer is None:
                # Initialize default tracer if none exists
                import prela

                self.tracer = prela.init(
                    service_name="n8n-code-node",
                    exporter="console" if not endpoint else "http",
                    http_endpoint=endpoint,
                    api_key=api_key,
                )
        else:
            self.tracer = tracer

        # Generate trace_id from execution_id
        self.trace_id = f"n8n-{execution_id}"

        # Spans
        self.workflow_span: Optional[Span] = None
        self.node_span: Optional[Span] = None

    def __enter__(self) -> "PrelaN8nContext":
        """Start tracing context."""
        try:
            # Create workflow-level span
            self.workflow_span = Span(
                trace_id=self.trace_id,
                parent_span_id=None,
                name=f"n8n.workflow.{self.workflow_name}",
                span_type=SpanType.AGENT,
                started_at=now(),
                attributes={
                    "n8n.workflow_id": self.workflow_id,
                    "n8n.workflow_name": self.workflow_name,
                    "n8n.execution_id": self.execution_id,
                    "service.name": "n8n",
                },
            )

            # Create node-level span
            self.node_span = Span(
                trace_id=self.trace_id,
                parent_span_id=self.workflow_span.span_id,
                name=f"n8n.node.{self.node_name}",
                span_type=SpanType.CUSTOM,
                started_at=now(),
                attributes={
                    "n8n.node_name": self.node_name,
                    "n8n.node_type": self.node_type,
                    "service.name": "n8n",
                },
            )

            logger.debug(
                f"Started n8n Code node tracing: {self.workflow_name}/{self.node_name}"
            )

        except Exception as e:
            logger.error(f"Failed to start n8n tracing context: {e}", exc_info=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracing context and export spans."""
        try:
            # Handle exceptions
            if exc_type is not None:
                if self.node_span:
                    self.node_span.set_status(SpanStatus.ERROR)
                    self.node_span.add_event(
                        name="exception",
                        attributes={
                            "exception.type": exc_type.__name__,
                            "exception.message": str(exc_val),
                        },
                    )
                if self.workflow_span:
                    self.workflow_span.set_status(SpanStatus.ERROR)

            # End spans
            if self.node_span:
                self.node_span.end()
            if self.workflow_span:
                self.workflow_span.end()

            # Export spans if tracer has exporter
            if self.tracer and self.tracer.exporter:
                spans = [s for s in [self.workflow_span, self.node_span] if s]
                for span in spans:
                    self.tracer.exporter.export([span])

            logger.debug(f"Completed n8n Code node tracing: {self.node_name}")

        except Exception as e:
            logger.error(f"Failed to end n8n tracing context: {e}", exc_info=True)

        # Don't suppress exceptions
        return False

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens: Optional[dict] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Log an LLM call within the Code node.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            prompt: Input prompt text
            response: Model response text
            tokens: Token usage dict with "prompt", "completion", "total" keys
            provider: AI provider (openai, anthropic, etc.)
            temperature: Temperature parameter used
            **kwargs: Additional attributes to attach to the span
        """
        if not self.node_span:
            logger.warning("Cannot log LLM call: node span not initialized")
            return

        try:
            # NEW: Replay capture if enabled
            replay_capture = None
            if self.tracer and getattr(self.tracer, "capture_for_replay", False):
                from prela.core.replay import ReplayCapture

                replay_capture = ReplayCapture()
                replay_capture.set_llm_request(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                )
                replay_capture.set_llm_response(
                    text=response,
                    prompt_tokens=tokens.get("prompt") if tokens else None,
                    completion_tokens=tokens.get("completion") if tokens else None,
                )
                if provider:
                    replay_capture.set_model_info(provider=provider)

            # Create child LLM span
            llm_span = Span(
                trace_id=self.trace_id,
                parent_span_id=self.node_span.span_id,
                name=f"llm.{model}",
                span_type=SpanType.LLM,
                started_at=now(),
                attributes={
                    "llm.model": model,
                    "llm.prompt": prompt[:500],  # Truncate
                    "llm.response": response[:500],  # Truncate
                    **kwargs,
                },
            )

            # Add provider if specified
            if provider:
                llm_span.attributes["llm.provider"] = provider

            # Add temperature if specified
            if temperature is not None:
                llm_span.attributes["llm.temperature"] = temperature

            # Add token usage if available
            if tokens:
                if "prompt" in tokens:
                    llm_span.attributes["llm.prompt_tokens"] = tokens["prompt"]
                if "completion" in tokens:
                    llm_span.attributes["llm.completion_tokens"] = tokens["completion"]
                if "total" in tokens:
                    llm_span.attributes["llm.total_tokens"] = tokens["total"]

            # NEW: Attach replay snapshot
            if replay_capture:
                try:
                    object.__setattr__(llm_span, "replay_snapshot", replay_capture.build())
                except Exception as e:
                    logger.debug(f"Failed to capture replay data: {e}")

            # End span immediately (synchronous call)
            llm_span.end()

            # Export if tracer has exporter
            if self.tracer and self.tracer.exporter:
                self.tracer.exporter.export([llm_span])

            logger.debug(f"Logged LLM call: {model}")

        except Exception as e:
            logger.error(f"Failed to log LLM call: {e}", exc_info=True)

    def log_tool_call(
        self,
        tool_name: str,
        input: Any,
        output: Any,
        error: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log a tool call within the Code node.

        Args:
            tool_name: Name of the tool/function called
            input: Input parameters to the tool
            output: Output result from the tool
            error: Optional error message if tool failed
            **kwargs: Additional attributes to attach to the span
        """
        if not self.node_span:
            logger.warning("Cannot log tool call: node span not initialized")
            return

        try:
            # Create child tool span
            tool_span = Span(
                trace_id=self.trace_id,
                parent_span_id=self.node_span.span_id,
                name=f"tool.{tool_name}",
                span_type=SpanType.TOOL,
                started_at=now(),
                attributes={
                    "tool.name": tool_name,
                    "tool.input": str(input)[:500],  # Truncate
                    "tool.output": str(output)[:500],  # Truncate
                    **kwargs,
                },
            )

            # Add error if present
            if error:
                tool_span.set_status(SpanStatus.ERROR)
                tool_span.attributes["tool.error"] = str(error)[:500]

            # End span immediately
            tool_span.end()

            # Export if tracer has exporter
            if self.tracer and self.tracer.exporter:
                self.tracer.exporter.export([tool_span])

            logger.debug(f"Logged tool call: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to log tool call: {e}", exc_info=True)

    def log_retrieval(
        self,
        query: str,
        documents: list[dict],
        retriever_type: Optional[str] = None,
        similarity_top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Log a retrieval/search operation within the Code node.

        Args:
            query: Search query text
            documents: Retrieved documents (list of dicts with text/score/metadata)
            retriever_type: Type of retriever used (vector, keyword, hybrid)
            similarity_top_k: Number of documents requested
            **kwargs: Additional attributes to attach to the span
        """
        if not self.node_span:
            logger.warning("Cannot log retrieval: node span not initialized")
            return

        try:
            # Create child retrieval span
            retrieval_span = Span(
                trace_id=self.trace_id,
                parent_span_id=self.node_span.span_id,
                name="retrieval",
                span_type=SpanType.RETRIEVAL,
                started_at=now(),
                attributes={
                    "retrieval.query": query[:200],  # Truncate
                    "retrieval.document_count": len(documents),
                    **kwargs,
                },
            )

            # Add retriever type if specified
            if retriever_type:
                retrieval_span.attributes["retrieval.type"] = retriever_type

            # Add similarity_top_k if specified
            if similarity_top_k is not None:
                retrieval_span.attributes["retrieval.similarity_top_k"] = (
                    similarity_top_k
                )

            # Add document details (limit to 5 docs)
            for i, doc in enumerate(documents[:5]):
                if "score" in doc:
                    retrieval_span.attributes[f"retrieval.document.{i}.score"] = doc[
                        "score"
                    ]
                if "text" in doc:
                    retrieval_span.attributes[f"retrieval.document.{i}.text"] = str(
                        doc["text"]
                    )[:200]

            # End span immediately
            retrieval_span.end()

            # Export if tracer has exporter
            if self.tracer and self.tracer.exporter:
                self.tracer.exporter.export([retrieval_span])

            logger.debug(f"Logged retrieval: {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}", exc_info=True)


def trace_n8n_code(
    items: list,
    workflow_context: dict,
    execution_context: dict,
    node_context: dict,
    tracer: Optional[Tracer] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> PrelaN8nContext:
    """
    Convenience function for tracing n8n Code node execution.

    This function extracts the necessary identifiers from n8n's built-in
    context variables ($workflow, $execution, $node) and creates a
    PrelaN8nContext for easy tracing.

    Args:
        items: n8n input items (from previous node)
        workflow_context: $workflow context from n8n
        execution_context: $execution context from n8n
        node_context: $node context from n8n
        tracer: Optional Prela tracer instance
        api_key: Optional API key for remote export
        endpoint: Optional endpoint URL for remote export

    Returns:
        PrelaN8nContext ready to use as context manager

    Example:
        ```python
        # Inside n8n Code node
        from prela.instrumentation.n8n import trace_n8n_code

        with trace_n8n_code(items, $workflow, $execution, $node) as ctx:
            # Your code here
            result = my_function(items[0]["json"])

            # Log LLM call
            ctx.log_llm_call(
                model="gpt-4",
                prompt="Hello",
                response=result,
                tokens={"prompt": 10, "completion": 20}
            )

            return [{"json": {"result": result}}]
        ```
    """
    return PrelaN8nContext(
        workflow_id=workflow_context.get("id", "unknown"),
        workflow_name=workflow_context.get("name", "Unknown Workflow"),
        execution_id=execution_context.get("id", "unknown"),
        node_name=node_context.get("name", "Unknown Node"),
        node_type=node_context.get("type", "n8n-nodes-base.code"),
        tracer=tracer,
        api_key=api_key,
        endpoint=endpoint,
    )


def prela_n8n_traced(
    func: Optional[Callable] = None,
    *,
    tracer: Optional[Tracer] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Callable:
    """
    Decorator for automatically tracing n8n Code node functions.

    This decorator expects the decorated function to accept n8n context
    variables as arguments and automatically wraps the execution in a
    PrelaN8nContext.

    Args:
        func: Function to decorate (optional, for @prela_n8n_traced usage)
        tracer: Optional Prela tracer instance
        api_key: Optional API key for remote export
        endpoint: Optional endpoint URL for remote export

    Returns:
        Decorated function

    Example:
        ```python
        # Inside n8n Code node
        from prela.instrumentation.n8n import prela_n8n_traced

        @prela_n8n_traced
        def process_items(items, workflow, execution, node):
            # Automatically traced!
            result = call_api(items[0]["json"])
            return [{"json": {"result": result}}]

        # Call the function with n8n contexts
        return process_items(items, $workflow, $execution, $node)
        ```
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(items, workflow, execution, node, *args, **kwargs):
            ctx = trace_n8n_code(
                items=items,
                workflow_context=workflow,
                execution_context=execution,
                node_context=node,
                tracer=tracer,
                api_key=api_key,
                endpoint=endpoint,
            )

            with ctx:
                result = f(items, workflow, execution, node, *args, **kwargs)
                return result

        return wrapper

    # Support both @prela_n8n_traced and @prela_n8n_traced()
    if func is None:
        return decorator
    else:
        return decorator(func)
