"""Context propagation for distributed tracing.

This module provides thread-safe and async-safe context management using
Python's contextvars module. It allows for proper trace context propagation
across async boundaries and thread pools.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token, copy_context
from functools import wraps
from typing import Any, Callable

from prela.core.span import Span

# Context variable for thread-safe and async-safe storage
_current_context: ContextVar[TraceContext | None] = ContextVar("_current_context", default=None)


class TraceContext:
    """A trace context manages the current trace and span stack.

    This class maintains the active trace ID, a stack of active spans,
    and baggage (inherited metadata) that propagates through the trace.
    """

    __slots__ = ("trace_id", "span_stack", "baggage", "sampled", "all_spans")

    def __init__(
        self,
        trace_id: str | None = None,
        sampled: bool = True,
        baggage: dict[str, str] | None = None,
    ) -> None:
        """Initialize a new trace context.

        Args:
            trace_id: Unique identifier for this trace (generates UUID if not provided)
            sampled: Whether this trace should be sampled/recorded
            baggage: Initial baggage metadata to propagate
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_stack: list[Span] = []
        self.baggage: dict[str, str] = baggage or {}
        self.sampled = sampled
        self.all_spans: list[Span] = []

    def current_span(self) -> Span | None:
        """Get the currently active span.

        Returns:
            The span at the top of the stack, or None if stack is empty
        """
        return self.span_stack[-1] if self.span_stack else None

    def push_span(self, span: Span) -> None:
        """Push a span onto the stack.

        Args:
            span: The span to make active
        """
        self.span_stack.append(span)

    def pop_span(self) -> Span | None:
        """Pop the current span from the stack.

        Returns:
            The popped span, or None if stack was empty
        """
        return self.span_stack.pop() if self.span_stack else None

    def add_completed_span(self, span: Span) -> None:
        """Add a completed span to the collection.

        Args:
            span: The completed span to add to the trace
        """
        self.all_spans.append(span)

    def set_baggage(self, key: str, value: str) -> None:
        """Set a baggage item that propagates through the trace.

        Args:
            key: Baggage key
            value: Baggage value
        """
        self.baggage[key] = value

    def get_baggage(self, key: str) -> str | None:
        """Get a baggage item.

        Args:
            key: Baggage key

        Returns:
            Baggage value, or None if not found
        """
        return self.baggage.get(key)

    def clear_baggage(self) -> None:
        """Clear all baggage items."""
        self.baggage.clear()


def get_current_context() -> TraceContext | None:
    """Get the current trace context.

    Returns:
        The active trace context, or None if no context is active
    """
    return _current_context.get()


def get_current_span() -> Span | None:
    """Get the currently active span.

    Returns:
        The active span, or None if no context or no active span
    """
    ctx = get_current_context()
    return ctx.current_span() if ctx else None


def get_current_trace_id() -> str | None:
    """Get the current trace ID.

    Returns:
        The active trace ID, or None if no context is active
    """
    ctx = get_current_context()
    return ctx.trace_id if ctx else None


def set_context(ctx: TraceContext) -> Token[TraceContext | None]:
    """Set the current trace context.

    Args:
        ctx: The trace context to set as active

    Returns:
        A token that can be used to reset the context
    """
    return _current_context.set(ctx)


def reset_context(token: Token[TraceContext | None]) -> None:
    """Reset the context to its previous value.

    Args:
        token: The token returned by set_context
    """
    _current_context.reset(token)


@contextmanager
def new_trace_context(
    trace_id: str | None = None, sampled: bool = True, baggage: dict[str, str] | None = None
) -> Iterator[TraceContext]:
    """Create a new trace context for the duration of the context manager.

    This context manager creates a new trace context, sets it as active,
    yields it for use, and automatically resets it on exit.

    Args:
        trace_id: Unique identifier for this trace (generates UUID if not provided)
        sampled: Whether this trace should be sampled/recorded
        baggage: Initial baggage metadata to propagate

    Yields:
        The newly created trace context

    Example:
        >>> with new_trace_context() as ctx:
        ...     span = Span(name="operation", trace_id=ctx.trace_id)
        ...     ctx.push_span(span)
        ...     # Do work
        ...     ctx.pop_span()
    """
    ctx = TraceContext(trace_id=trace_id, sampled=sampled, baggage=baggage)
    token = set_context(ctx)
    try:
        yield ctx
    finally:
        reset_context(token)


def copy_context_to_thread(func: Callable[..., Any]) -> Callable[..., Any]:
    """Create a wrapper that copies the current context to a new thread.

    This function captures the current contextvars context at call time
    and creates a wrapper that will run the function in that context.
    This is essential for maintaining trace continuity when using thread pools.

    IMPORTANT: Call this function INSIDE the context you want to propagate,
    BEFORE submitting to the thread pool.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that will run in the captured context

    Example:
        >>> def background_task():
        ...     span = get_current_span()
        ...     print(f"Span: {span}")
        >>>
        >>> with new_trace_context() as ctx:
        ...     # Capture context NOW, before submitting to pool
        ...     wrapped = copy_context_to_thread(background_task)
        ...     with ThreadPoolExecutor() as executor:
        ...         future = executor.submit(wrapped)
        ...         future.result()
    """
    # Capture the full context (including all contextvars) NOW
    # This happens in the calling thread
    ctx = copy_context()

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Run the function in the captured context
        # This ensures all contextvars are available in the worker thread
        return ctx.run(func, *args, **kwargs)

    return wrapper
