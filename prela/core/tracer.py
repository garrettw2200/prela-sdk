"""Tracer for creating and managing spans."""

from __future__ import annotations

import asyncio
import functools
import inspect
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from prela.core.clock import now
from prela.core.context import (
    TraceContext,
    get_current_context,
    reset_context,
    set_context,
)
from prela.core.sampler import AlwaysOnSampler, BaseSampler
from prela.core.span import Span, SpanStatus, SpanType
from prela.exporters.base import BaseExporter

# Type variable for preserving function signatures
F = TypeVar("F", bound=Callable[..., Any])

# Global tracer instance
_global_tracer: Tracer | None = None


class Tracer:
    """
    Main tracer for creating and managing spans.

    The Tracer is responsible for:
    - Creating spans with proper trace/span IDs
    - Managing trace context and span hierarchies
    - Applying sampling decisions
    - Exporting completed spans

    Example:
        ```python
        from prela.core.tracer import Tracer
        from prela.exporters.console import ConsoleExporter

        tracer = Tracer(
            service_name="my-agent",
            exporter=ConsoleExporter()
        )

        # Create spans using context manager
        with tracer.span("operation") as span:
            span.set_attribute("key", "value")
            # Nested spans inherit trace context
            with tracer.span("sub-operation") as child:
                child.set_attribute("nested", True)
        ```
    """

    def __init__(
        self,
        service_name: str = "default",
        exporter: BaseExporter | None = None,
        sampler: BaseSampler | None = None,
        capture_for_replay: bool = False,
    ):
        """
        Initialize a tracer.

        Args:
            service_name: Name of the service (added to all spans as service.name)
            exporter: Exporter for sending spans to backend (None = no export)
            sampler: Sampler for controlling trace volume (default: AlwaysOnSampler)
            capture_for_replay: If True, capture full replay data (default: False)
        """
        self.service_name = service_name
        self.exporter = exporter
        self.sampler = sampler or AlwaysOnSampler()
        self.capture_for_replay = capture_for_replay

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """
        Create a new span as a context manager.

        The span is automatically:
        - Started when entering the context
        - Ended when exiting the context
        - Exported if it's a root span and sampling decision is True
        - Linked to parent span if one exists in current context

        Exceptions are automatically captured and recorded on the span.

        Args:
            name: Name of the span (e.g., "process_request", "llm_call")
            span_type: Type of operation (LLM, TOOL, AGENT, etc.)
            attributes: Initial attributes to set on the span

        Yields:
            Span: The created span (can be used to add attributes/events)

        Example:
            ```python
            with tracer.span("database_query", SpanType.CUSTOM) as span:
                span.set_attribute("query", "SELECT * FROM users")
                result = execute_query()
                span.set_attribute("row_count", len(result))
            ```
        """
        # Get or create trace context
        ctx = get_current_context()
        token = None
        if ctx is None:
            # Start new trace
            trace_id = str(uuid.uuid4())
            sampled = self.sampler.should_sample(trace_id)
            ctx = TraceContext(trace_id=trace_id, sampled=sampled)
            token = set_context(ctx)
        else:
            # Continue existing trace
            trace_id = ctx.trace_id
            sampled = ctx.sampled

        # Create span
        parent_span = ctx.current_span()
        parent_span_id = parent_span.span_id if parent_span else None

        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            started_at=now(),
            attributes=attributes or {},
        )

        # Add service name
        span.set_attribute("service.name", self.service_name)

        # Push to context
        ctx.push_span(span)

        try:
            yield span
        except Exception as e:
            # Capture exception
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            # End span
            span.end()

            # Pop from context
            ctx.pop_span()

            # Add to completed spans collection
            ctx.add_completed_span(span)

            # Export if sampled and this was a root span
            if sampled and parent_span is None and self.exporter:
                # Export ALL spans in the trace, not just the root
                self.exporter.export(ctx.all_spans)

            # Reset context if we created it
            if token is not None:
                reset_context(token)

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Create a new span without using a context manager.

        Unlike the span() context manager, this method returns a Span
        object that must be manually ended by calling span.end().
        This is useful for instrumentations where the span lifetime
        cannot be expressed as a context manager.

        The span will automatically:
        - Pop itself from the context stack when ended
        - Export itself if it's a root span and sampling is enabled
        - Reset the context if it created a new trace

        Args:
            name: Name of the span (e.g., "process_request", "llm_call")
            span_type: Type of operation (LLM, TOOL, AGENT, etc.)
            attributes: Initial attributes to set on the span

        Returns:
            Span: The created span (must call .end() when done)

        Example:
            ```python
            span = tracer.start_span("llm_call", SpanType.LLM)
            span.set_attribute("model", "gpt-4")
            try:
                # Do work
                result = call_llm()
                span.set_status(SpanStatus.SUCCESS)
            except Exception as e:
                span.set_status(SpanStatus.ERROR, str(e))
            finally:
                span.end()  # Automatically handles cleanup
            ```
        """
        # Get or create trace context
        ctx = get_current_context()
        created_context = False
        context_token = None

        if ctx is None:
            # Start new trace
            trace_id = str(uuid.uuid4())
            sampled = self.sampler.should_sample(trace_id)
            ctx = TraceContext(trace_id=trace_id, sampled=sampled)
            context_token = set_context(ctx)
            created_context = True
        else:
            # Continue existing trace
            trace_id = ctx.trace_id
            sampled = ctx.sampled

        # Create span
        parent_span = ctx.current_span()
        parent_span_id = parent_span.span_id if parent_span else None

        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            started_at=now(),
            attributes=attributes or {},
        )

        # Add service name
        span.set_attribute("service.name", self.service_name)

        # Store cleanup metadata on the span (using private attributes)
        # These will be used by the span's end() method or by explicit cleanup
        object.__setattr__(span, "_tracer", self)
        object.__setattr__(span, "_context_token", context_token if created_context else None)
        object.__setattr__(span, "_sampled", sampled)

        # Push to context
        ctx.push_span(span)

        return span

    def get_current_span(self) -> Span | None:
        """
        Get the currently active span from context.

        Returns:
            Span | None: The active span, or None if no span is active
        """
        ctx = get_current_context()
        return ctx.current_span() if ctx else None

    def set_global(self) -> None:
        """
        Set this tracer as the global default.

        After calling this, get_tracer() will return this tracer instance.
        This is useful for auto-instrumentation where instrumentors need
        access to a tracer without explicit passing.

        Example:
            ```python
            tracer = Tracer(service_name="my-app")
            tracer.set_global()

            # Later, from anywhere in the code
            from prela.core.tracer import get_tracer
            tracer = get_tracer()
            ```
        """
        global _global_tracer
        _global_tracer = self

    def shutdown(self) -> None:
        """
        Shutdown the tracer and flush exporter.

        This ensures all pending spans are exported before the process exits.
        Should be called before application shutdown.

        Example:
            ```python
            import atexit

            tracer = Tracer(service_name="my-app")
            atexit.register(tracer.shutdown)
            ```
        """
        if self.exporter:
            self.exporter.shutdown()


def get_tracer() -> Tracer | None:
    """
    Get the global tracer instance.

    Returns:
        Tracer | None: The global tracer, or None if no global tracer is set

    Example:
        ```python
        from prela.core.tracer import get_tracer

        tracer = get_tracer()
        if tracer:
            with tracer.span("operation") as span:
                span.set_attribute("key", "value")
        ```
    """
    return _global_tracer


def set_global_tracer(tracer: Tracer) -> None:
    """
    Set the global tracer instance.

    This is an alternative to calling tracer.set_global().

    Args:
        tracer: The tracer to set as global

    Example:
        ```python
        from prela.core.tracer import Tracer, set_global_tracer

        tracer = Tracer(service_name="my-app")
        set_global_tracer(tracer)
        ```
    """
    global _global_tracer
    _global_tracer = tracer


def trace(
    name: str | None = None,
    span_type: SpanType = SpanType.CUSTOM,
    attributes: dict[str, Any] | None = None,
    tracer: Tracer | None = None,
) -> Callable[[F], F]:
    """
    Decorator for automatically tracing function execution.

    This decorator wraps a function (sync or async) and creates a span
    for each invocation. The span is automatically:
    - Created when the function is called
    - Ended when the function returns
    - Exported if it's a root span (uses global tracer)
    - Captures exceptions and marks span as ERROR

    The decorator works with both synchronous and asynchronous functions.

    Args:
        name: Name of the span (default: function name)
        span_type: Type of operation (default: CUSTOM)
        attributes: Initial attributes to set on the span
        tracer: Tracer instance to use (default: global tracer from init())

    Returns:
        Decorated function with automatic tracing

    Raises:
        RuntimeError: If no tracer is provided and no global tracer is set

    Example with sync function:
        ```python
        import prela

        prela.init(service_name="my-app")

        @prela.trace("process_data")
        def process_data(items):
            # Function is automatically traced
            result = [item * 2 for item in items]
            return result

        # Each call creates a span
        result = process_data([1, 2, 3])
        ```

    Example with async function:
        ```python
        import prela
        import asyncio

        prela.init(service_name="my-app")

        @prela.trace("fetch_data", span_type=prela.SpanType.RETRIEVAL)
        async def fetch_data(url):
            # Async function is automatically traced
            await asyncio.sleep(0.1)
            return {"data": "example"}

        # Each call creates a span
        result = await fetch_data("https://api.example.com")
        ```

    Example with custom attributes:
        ```python
        @prela.trace(
            "database_query",
            span_type=prela.SpanType.CUSTOM,
            attributes={"db": "postgres", "table": "users"}
        )
        def query_users(limit=10):
            # Span has initial attributes set
            return fetch_users(limit)
        ```

    Example with manual attribute setting:
        ```python
        @prela.trace("calculate")
        def calculate(x, y):
            # Access current span to add more attributes
            span = prela.get_current_span()
            if span:
                span.set_attribute("x", x)
                span.set_attribute("y", y)
            result = x + y
            if span:
                span.set_attribute("result", result)
            return result
        ```

    Example using function-specific tracer:
        ```python
        my_tracer = Tracer(service_name="custom")

        @prela.trace("operation", tracer=my_tracer)
        def my_operation():
            pass
        ```
    """

    def decorator(func: F) -> F:
        # Determine span name
        span_name = name or func.__name__

        # Get the tracer to use
        tracer_instance = tracer
        if tracer_instance is None:
            tracer_instance = get_tracer()
            if tracer_instance is None:
                raise RuntimeError(
                    "No global tracer set. Call prela.init() first or provide a tracer parameter."
                )

        # Check if function is async
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer_instance.span(
                    span_name, span_type=span_type, attributes=attributes
                ) as span:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    # Execute the async function
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer_instance.span(
                    span_name, span_type=span_type, attributes=attributes
                ) as span:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    # Execute the sync function
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator
