"""Tests for context propagation."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from prela.core.context import (
    TraceContext,
    copy_context_to_thread,
    get_current_context,
    get_current_span,
    new_trace_context,
    reset_context,
    set_context,
)
from prela.core.span import Span, SpanType


class TestTraceContext:
    """Tests for TraceContext class."""

    def test_create_context_with_defaults(self) -> None:
        """Test creating a context with default values."""
        ctx = TraceContext()

        assert ctx.trace_id is not None
        assert ctx.sampled is True
        assert ctx.baggage == {}
        assert ctx.span_stack == []

    def test_create_context_with_custom_values(self) -> None:
        """Test creating a context with custom values."""
        trace_id = "custom-trace-id"
        baggage = {"user_id": "123", "session": "abc"}

        ctx = TraceContext(trace_id=trace_id, sampled=False, baggage=baggage)

        assert ctx.trace_id == trace_id
        assert ctx.sampled is False
        assert ctx.baggage == baggage

    def test_context_generates_unique_trace_ids(self) -> None:
        """Test that each context gets a unique trace ID."""
        ctx1 = TraceContext()
        ctx2 = TraceContext()

        assert ctx1.trace_id != ctx2.trace_id

    def test_current_span_empty_stack(self) -> None:
        """Test current_span returns None when stack is empty."""
        ctx = TraceContext()

        assert ctx.current_span() is None

    def test_push_span(self) -> None:
        """Test pushing a span onto the stack."""
        ctx = TraceContext()
        span = Span(name="test-span", trace_id=ctx.trace_id)

        ctx.push_span(span)

        assert len(ctx.span_stack) == 1
        assert ctx.current_span() == span

    def test_push_multiple_spans(self) -> None:
        """Test pushing multiple spans creates a stack."""
        ctx = TraceContext()
        span1 = Span(name="span1", trace_id=ctx.trace_id)
        span2 = Span(name="span2", trace_id=ctx.trace_id)
        span3 = Span(name="span3", trace_id=ctx.trace_id)

        ctx.push_span(span1)
        ctx.push_span(span2)
        ctx.push_span(span3)

        assert len(ctx.span_stack) == 3
        assert ctx.current_span() == span3

    def test_pop_span(self) -> None:
        """Test popping a span from the stack."""
        ctx = TraceContext()
        span = Span(name="test-span", trace_id=ctx.trace_id)
        ctx.push_span(span)

        popped = ctx.pop_span()

        assert popped == span
        assert len(ctx.span_stack) == 0
        assert ctx.current_span() is None

    def test_pop_span_empty_stack(self) -> None:
        """Test popping from an empty stack returns None."""
        ctx = TraceContext()

        popped = ctx.pop_span()

        assert popped is None

    def test_pop_span_order(self) -> None:
        """Test that spans are popped in LIFO order."""
        ctx = TraceContext()
        span1 = Span(name="span1", trace_id=ctx.trace_id)
        span2 = Span(name="span2", trace_id=ctx.trace_id)
        span3 = Span(name="span3", trace_id=ctx.trace_id)

        ctx.push_span(span1)
        ctx.push_span(span2)
        ctx.push_span(span3)

        assert ctx.pop_span() == span3
        assert ctx.pop_span() == span2
        assert ctx.pop_span() == span1
        assert ctx.pop_span() is None

    def test_set_baggage(self) -> None:
        """Test setting baggage items."""
        ctx = TraceContext()

        ctx.set_baggage("user_id", "123")
        ctx.set_baggage("environment", "production")

        assert ctx.baggage["user_id"] == "123"
        assert ctx.baggage["environment"] == "production"

    def test_get_baggage(self) -> None:
        """Test getting baggage items."""
        ctx = TraceContext(baggage={"key": "value"})

        assert ctx.get_baggage("key") == "value"
        assert ctx.get_baggage("missing") is None

    def test_clear_baggage(self) -> None:
        """Test clearing all baggage."""
        ctx = TraceContext(baggage={"key1": "value1", "key2": "value2"})

        ctx.clear_baggage()

        assert ctx.baggage == {}


class TestContextManagement:
    """Tests for module-level context management functions."""

    def test_get_current_context_none_by_default(self) -> None:
        """Test that current context is None by default."""
        # Context should be None in a fresh test
        ctx = get_current_context()
        assert ctx is None

    def test_set_and_get_context(self) -> None:
        """Test setting and getting context."""
        new_ctx = TraceContext(trace_id="test-trace")
        token = set_context(new_ctx)

        try:
            ctx = get_current_context()
            assert ctx == new_ctx
            assert ctx.trace_id == "test-trace"
        finally:
            reset_context(token)

    def test_reset_context(self) -> None:
        """Test resetting context to previous value."""
        original_ctx = get_current_context()
        new_ctx = TraceContext(trace_id="test-trace")

        token = set_context(new_ctx)
        assert get_current_context() == new_ctx

        reset_context(token)
        assert get_current_context() == original_ctx

    def test_nested_contexts(self) -> None:
        """Test nested context setting and resetting."""
        ctx1 = TraceContext(trace_id="trace-1")
        ctx2 = TraceContext(trace_id="trace-2")
        ctx3 = TraceContext(trace_id="trace-3")

        token1 = set_context(ctx1)
        try:
            assert get_current_context() == ctx1

            token2 = set_context(ctx2)
            try:
                assert get_current_context() == ctx2

                token3 = set_context(ctx3)
                try:
                    assert get_current_context() == ctx3
                finally:
                    reset_context(token3)

                assert get_current_context() == ctx2
            finally:
                reset_context(token2)

            assert get_current_context() == ctx1
        finally:
            reset_context(token1)

    def test_get_current_span_no_context(self) -> None:
        """Test getting current span when no context is active."""
        span = get_current_span()
        assert span is None

    def test_get_current_span_with_context(self) -> None:
        """Test getting current span from active context."""
        ctx = TraceContext()
        test_span = Span(name="test-span", trace_id=ctx.trace_id)
        ctx.push_span(test_span)

        token = set_context(ctx)
        try:
            span = get_current_span()
            assert span == test_span
        finally:
            reset_context(token)


class TestNewTraceContext:
    """Tests for new_trace_context context manager."""

    def test_context_manager_creates_context(self) -> None:
        """Test that context manager creates and sets a context."""
        with new_trace_context() as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id is not None
            assert get_current_context() == ctx

    def test_context_manager_with_custom_trace_id(self) -> None:
        """Test context manager with custom trace ID."""
        with new_trace_context(trace_id="custom-trace") as ctx:
            assert ctx.trace_id == "custom-trace"

    def test_context_manager_with_sampling(self) -> None:
        """Test context manager with sampling disabled."""
        with new_trace_context(sampled=False) as ctx:
            assert ctx.sampled is False

    def test_context_manager_with_baggage(self) -> None:
        """Test context manager with initial baggage."""
        baggage = {"key": "value"}
        with new_trace_context(baggage=baggage) as ctx:
            assert ctx.baggage == baggage

    def test_context_manager_resets_on_exit(self) -> None:
        """Test that context is reset after exiting context manager."""
        original_ctx = get_current_context()

        with new_trace_context(trace_id="temp-trace") as ctx:
            assert get_current_context() == ctx

        assert get_current_context() == original_ctx

    def test_context_manager_resets_on_exception(self) -> None:
        """Test that context is reset even when exception occurs."""
        original_ctx = get_current_context()

        try:
            with new_trace_context(trace_id="temp-trace"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert get_current_context() == original_ctx

    def test_nested_context_managers(self) -> None:
        """Test nested context managers."""
        with new_trace_context(trace_id="outer") as outer_ctx:
            assert get_current_context() == outer_ctx

            with new_trace_context(trace_id="inner") as inner_ctx:
                assert get_current_context() == inner_ctx
                assert inner_ctx.trace_id == "inner"

            # Should restore outer context
            assert get_current_context() == outer_ctx

    def test_span_stack_in_context_manager(self) -> None:
        """Test using span stack within context manager."""
        with new_trace_context() as ctx:
            span1 = Span(name="span1", trace_id=ctx.trace_id)
            span2 = Span(name="span2", trace_id=ctx.trace_id)

            ctx.push_span(span1)
            assert get_current_span() == span1

            ctx.push_span(span2)
            assert get_current_span() == span2

            ctx.pop_span()
            assert get_current_span() == span1


class TestCopyContextToThread:
    """Tests for copy_context_to_thread decorator."""

    def test_decorator_preserves_context(self) -> None:
        """Test that decorator copies context to new thread."""
        result = {"span_name": None}

        def thread_function() -> None:
            span = get_current_span()
            result["span_name"] = span.name if span else None

        with new_trace_context() as ctx:
            test_span = Span(name="test-span", trace_id=ctx.trace_id)
            ctx.push_span(test_span)

            # Wrap the function with decorator BEFORE submitting
            # This captures the context at decoration time
            wrapped_func = copy_context_to_thread(thread_function)

            # Run in thread pool
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(wrapped_func)
                future.result()

        assert result["span_name"] == "test-span"

    def test_decorator_without_context(self) -> None:
        """Test decorator works when no context is active."""
        result = {"executed": False}

        @copy_context_to_thread
        def thread_function() -> None:
            result["executed"] = True
            assert get_current_context() is None

        # Run without setting a context
        thread_function()

        assert result["executed"] is True

    def test_decorator_with_return_value(self) -> None:
        """Test that wrapper preserves return values."""

        def get_trace_id() -> str | None:
            ctx = get_current_context()
            return ctx.trace_id if ctx else None

        with new_trace_context(trace_id="test-trace"):
            # Wrap function to capture context
            wrapped = copy_context_to_thread(get_trace_id)
            trace_id = wrapped()
            assert trace_id == "test-trace"

    def test_decorator_isolates_contexts(self) -> None:
        """Test that each thread gets its own context copy."""
        results: list[str | None] = []

        def capture_trace_id() -> str | None:
            ctx = get_current_context()
            return ctx.trace_id if ctx else None

        with new_trace_context(trace_id="main-trace"):
            # Wrap function to capture context
            wrapped_func = copy_context_to_thread(capture_trace_id)

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(wrapped_func) for _ in range(3)]
                results = [f.result() for f in futures]

        # All threads should see the same trace ID
        assert all(tid == "main-trace" for tid in results)


@pytest.mark.asyncio
class TestAsyncContextPropagation:
    """Tests for async context propagation."""

    async def test_context_propagates_to_async_function(self) -> None:
        """Test that context propagates to async functions."""

        async def async_operation() -> str | None:
            ctx = get_current_context()
            return ctx.trace_id if ctx else None

        with new_trace_context(trace_id="async-trace"):
            trace_id = await async_operation()
            assert trace_id == "async-trace"

    async def test_context_in_nested_async_calls(self) -> None:
        """Test context propagation through nested async calls."""

        async def inner_operation() -> Span | None:
            return get_current_span()

        async def outer_operation() -> Span | None:
            return await inner_operation()

        with new_trace_context() as ctx:
            test_span = Span(name="async-span", trace_id=ctx.trace_id)
            ctx.push_span(test_span)

            span = await outer_operation()
            assert span == test_span

    async def test_context_with_asyncio_gather(self) -> None:
        """Test context propagation with asyncio.gather."""

        async def get_trace_info() -> tuple[str | None, str | None]:
            await asyncio.sleep(0.01)  # Simulate async work
            ctx = get_current_context()
            span = get_current_span()
            return (ctx.trace_id if ctx else None, span.name if span else None)

        with new_trace_context(trace_id="gather-trace") as ctx:
            test_span = Span(name="gather-span", trace_id=ctx.trace_id)
            ctx.push_span(test_span)

            # Run multiple concurrent tasks
            results = await asyncio.gather(get_trace_info(), get_trace_info(), get_trace_info())

            # All tasks should see the same context
            for trace_id, span_name in results:
                assert trace_id == "gather-trace"
                assert span_name == "gather-span"

    async def test_context_isolation_between_tasks(self) -> None:
        """Test that different async tasks can have different contexts."""

        async def task_with_context(trace_id: str) -> str | None:
            with new_trace_context(trace_id=trace_id):
                await asyncio.sleep(0.01)
                ctx = get_current_context()
                return ctx.trace_id if ctx else None

        # Run tasks with different contexts concurrently
        results = await asyncio.gather(
            task_with_context("trace-1"),
            task_with_context("trace-2"),
            task_with_context("trace-3"),
        )

        assert results == ["trace-1", "trace-2", "trace-3"]

    async def test_span_stack_in_async_context(self) -> None:
        """Test span stack operations in async context."""

        async def async_operation(name: str) -> str | None:
            ctx = get_current_context()
            if ctx:
                span = Span(name=name, trace_id=ctx.trace_id)
                ctx.push_span(span)
                await asyncio.sleep(0.01)
                current = get_current_span()
                ctx.pop_span()
                return current.name if current else None
            return None

        with new_trace_context():
            result = await async_operation("async-span")
            assert result == "async-span"


class TestConcurrentAccess:
    """Tests for concurrent access to context."""

    def test_thread_isolation(self) -> None:
        """Test that each thread has isolated context."""
        results: dict[int, str | None] = {}

        def thread_function(thread_id: int, trace_id: str) -> None:
            with new_trace_context(trace_id=trace_id):
                # Simulate some work
                import time

                time.sleep(0.01)
                ctx = get_current_context()
                results[thread_id] = ctx.trace_id if ctx else None

        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_function, args=(i, f"trace-{i}"))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have seen its own trace ID
        for i in range(5):
            assert results[i] == f"trace-{i}"

    def test_parent_child_span_relationship(self) -> None:
        """Test parent-child span relationships in context."""
        with new_trace_context() as ctx:
            # Create parent span
            parent = Span(name="parent", trace_id=ctx.trace_id, span_type=SpanType.AGENT)
            ctx.push_span(parent)

            # Create child span with parent reference
            child = Span(
                name="child",
                trace_id=ctx.trace_id,
                parent_span_id=parent.span_id,
                span_type=SpanType.LLM,
            )
            ctx.push_span(child)

            # Verify hierarchy
            current = get_current_span()
            assert current == child
            assert current.parent_span_id == parent.span_id  # type: ignore[union-attr]

            # Pop child, parent should be current
            ctx.pop_span()
            assert get_current_span() == parent
