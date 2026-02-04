"""Tests for Tracer class."""

from __future__ import annotations

import asyncio

import pytest

from prela.core.sampler import AlwaysOffSampler, AlwaysOnSampler
from prela.core.span import SpanStatus, SpanType
from prela.core.tracer import Tracer, get_tracer, set_global_tracer, trace
from prela.exporters.base import BaseExporter, ExportResult


class MockExporter(BaseExporter):
    """Mock exporter for testing."""

    def __init__(self):
        self.exported_spans = []
        self.shutdown_called = False

    def export(self, spans: list) -> ExportResult:
        self.exported_spans.extend(spans)
        return ExportResult.SUCCESS

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_tracer_initialization():
    """Test tracer can be initialized with defaults."""
    tracer = Tracer()
    assert tracer.service_name == "default"
    assert tracer.exporter is None
    assert tracer.sampler is not None


def test_tracer_initialization_custom():
    """Test tracer initialization with custom values."""
    exporter = MockExporter()
    sampler = AlwaysOffSampler()
    tracer = Tracer(service_name="test-service", exporter=exporter, sampler=sampler)

    assert tracer.service_name == "test-service"
    assert tracer.exporter is exporter
    assert tracer.sampler is sampler


def test_tracer_span_creation():
    """Test creating a basic span."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test-service", exporter=exporter)

    with tracer.span("test-operation") as span:
        assert span.name == "test-operation"
        assert span.span_type == SpanType.CUSTOM
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.parent_span_id is None
        assert span.attributes.get("service.name") == "test-service"

    # Span should be exported after context exits
    assert len(exporter.exported_spans) == 1
    exported = exporter.exported_spans[0]
    assert exported.name == "test-operation"
    assert exported.ended_at is not None


def test_tracer_span_with_attributes():
    """Test creating a span with custom attributes."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    attrs = {"key1": "value1", "key2": 42}
    with tracer.span("test-operation", attributes=attrs) as span:
        assert span.attributes.get("key1") == "value1"
        assert span.attributes.get("key2") == 42
        assert span.attributes.get("service.name") == "default"


def test_tracer_span_with_type():
    """Test creating a span with custom type."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    with tracer.span("llm-call", span_type=SpanType.LLM) as span:
        assert span.span_type == SpanType.LLM


def test_tracer_nested_spans():
    """Test creating nested spans."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test-service", exporter=exporter)

    with tracer.span("parent") as parent:
        parent_id = parent.span_id
        trace_id = parent.trace_id

        with tracer.span("child") as child:
            assert child.parent_span_id == parent_id
            assert child.trace_id == trace_id  # Same trace

            with tracer.span("grandchild") as grandchild:
                assert grandchild.parent_span_id == child.span_id
                assert grandchild.trace_id == trace_id

    # All spans in the trace should be exported together when root completes
    assert len(exporter.exported_spans) == 3
    span_names = [s.name for s in exporter.exported_spans]
    assert "parent" in span_names
    assert "child" in span_names
    assert "grandchild" in span_names


def test_tracer_exception_capture():
    """Test that exceptions are captured in span."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    with pytest.raises(ValueError):
        with tracer.span("failing-operation") as span:
            raise ValueError("Test error")

    # Span should be exported with error status
    assert len(exporter.exported_spans) == 1
    exported = exporter.exported_spans[0]
    assert exported.status == SpanStatus.ERROR
    assert exported.status_message == "Test error"
    assert exported.attributes.get("error.type") == "ValueError"
    assert exported.attributes.get("error.message") == "Test error"


def test_tracer_sampling_always_on():
    """Test that AlwaysOnSampler samples all traces."""
    exporter = MockExporter()
    sampler = AlwaysOnSampler()
    tracer = Tracer(exporter=exporter, sampler=sampler)

    # Create multiple traces
    for i in range(5):
        with tracer.span(f"operation-{i}"):
            pass

    # All should be exported
    assert len(exporter.exported_spans) == 5


def test_tracer_sampling_always_off():
    """Test that AlwaysOffSampler samples no traces."""
    exporter = MockExporter()
    sampler = AlwaysOffSampler()
    tracer = Tracer(exporter=exporter, sampler=sampler)

    # Create multiple traces
    for i in range(5):
        with tracer.span(f"operation-{i}"):
            pass

    # None should be exported
    assert len(exporter.exported_spans) == 0


def test_tracer_get_current_span():
    """Test getting current span from tracer."""
    tracer = Tracer()

    # No span initially
    assert tracer.get_current_span() is None

    with tracer.span("outer") as outer:
        # Should return outer span
        current = tracer.get_current_span()
        assert current is not None
        assert current.span_id == outer.span_id

        with tracer.span("inner") as inner:
            # Should return inner span
            current = tracer.get_current_span()
            assert current is not None
            assert current.span_id == inner.span_id

        # Back to outer span
        current = tracer.get_current_span()
        assert current is not None
        assert current.span_id == outer.span_id

    # No span after exit
    assert tracer.get_current_span() is None


def test_tracer_set_global():
    """Test setting tracer as global."""
    tracer = Tracer(service_name="global-test")
    tracer.set_global()

    retrieved = get_tracer()
    assert retrieved is tracer
    assert retrieved.service_name == "global-test"


def test_set_global_tracer_function():
    """Test set_global_tracer function."""
    tracer = Tracer(service_name="function-test")
    set_global_tracer(tracer)

    retrieved = get_tracer()
    assert retrieved is tracer
    assert retrieved.service_name == "function-test"


def test_get_tracer_returns_none_initially():
    """Test get_tracer returns None when no global tracer is set."""
    # Reset global tracer
    set_global_tracer(None)
    assert get_tracer() is None


def test_tracer_shutdown():
    """Test tracer shutdown calls exporter shutdown."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    assert not exporter.shutdown_called
    tracer.shutdown()
    assert exporter.shutdown_called


def test_tracer_shutdown_no_exporter():
    """Test tracer shutdown works without exporter."""
    tracer = Tracer()
    tracer.shutdown()  # Should not raise


def test_tracer_span_attributes_can_be_modified():
    """Test that span attributes can be modified within context."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    with tracer.span("test") as span:
        span.set_attribute("added_key", "added_value")

    exported = exporter.exported_spans[0]
    assert exported.attributes.get("added_key") == "added_value"


def test_tracer_span_events_can_be_added():
    """Test that span events can be added within context."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    with tracer.span("test") as span:
        span.add_event("event1", {"key": "value"})
        span.add_event("event2")

    exported = exporter.exported_spans[0]
    assert len(exported.events) == 2
    assert exported.events[0].name == "event1"
    assert exported.events[1].name == "event2"


def test_tracer_multiple_independent_traces():
    """Test that multiple independent traces have different trace IDs."""
    exporter = MockExporter()
    tracer = Tracer(exporter=exporter)

    trace_ids = []

    for i in range(3):
        with tracer.span(f"trace-{i}") as span:
            trace_ids.append(span.trace_id)

    # All trace IDs should be unique
    assert len(set(trace_ids)) == 3


def test_tracer_no_export_without_exporter():
    """Test that spans work without an exporter."""
    tracer = Tracer()  # No exporter

    with tracer.span("test") as span:
        span.set_attribute("key", "value")

    # Should not raise, just no export happens


# ============================================================================
# @trace Decorator Tests
# ============================================================================


def test_trace_decorator_basic_sync():
    """Test @trace decorator on basic sync function."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("test_function")
    def my_function(x: int, y: int) -> int:
        return x + y

    result = my_function(2, 3)

    assert result == 5
    assert len(exporter.exported_spans) == 1
    span = exporter.exported_spans[0]
    assert span.name == "test_function"
    assert span.attributes.get("function.name") == "my_function"
    assert span.attributes.get("function.module") == "tests.test_tracer"


def test_trace_decorator_default_name():
    """Test @trace decorator uses function name by default."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace()
    def my_custom_function():
        return "result"

    result = my_custom_function()

    assert result == "result"
    assert len(exporter.exported_spans) == 1
    assert exporter.exported_spans[0].name == "my_custom_function"


def test_trace_decorator_with_span_type():
    """Test @trace decorator with custom span type."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("llm_call", span_type=SpanType.LLM)
    def call_llm():
        return "response"

    result = call_llm()

    assert result == "response"
    assert len(exporter.exported_spans) == 1
    span = exporter.exported_spans[0]
    assert span.span_type == SpanType.LLM


def test_trace_decorator_with_attributes():
    """Test @trace decorator with initial attributes."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("db_query", attributes={"db": "postgres", "table": "users"})
    def query_database():
        return [1, 2, 3]

    result = query_database()

    assert result == [1, 2, 3]
    assert len(exporter.exported_spans) == 1
    span = exporter.exported_spans[0]
    assert span.attributes.get("db") == "postgres"
    assert span.attributes.get("table") == "users"


def test_trace_decorator_captures_exception():
    """Test @trace decorator captures exceptions."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("failing_function")
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_function()

    assert len(exporter.exported_spans) == 1
    span = exporter.exported_spans[0]
    assert span.status == SpanStatus.ERROR
    assert span.status_message == "Test error"
    assert span.attributes.get("error.type") == "ValueError"


@pytest.mark.asyncio
async def test_trace_decorator_async_function():
    """Test @trace decorator on async function."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("async_function")
    async def async_operation(delay: float):
        await asyncio.sleep(delay)
        return "done"

    result = await async_operation(0.01)

    assert result == "done"
    assert len(exporter.exported_spans) == 1
    span = exporter.exported_spans[0]
    assert span.name == "async_function"
    assert span.attributes.get("function.name") == "async_operation"


@pytest.mark.asyncio
async def test_trace_decorator_async_exception():
    """Test @trace decorator captures async exceptions."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("failing_async")
    async def failing_async():
        await asyncio.sleep(0.01)
        raise RuntimeError("Async error")

    with pytest.raises(RuntimeError, match="Async error"):
        await failing_async()

    assert len(exporter.exported_spans) == 1
    span = exporter.exported_spans[0]
    assert span.status == SpanStatus.ERROR
    assert span.status_message == "Async error"


def test_trace_decorator_nested_functions():
    """Test @trace decorator with nested decorated functions."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("outer")
    def outer_function():
        return inner_function()

    @trace("inner")
    def inner_function():
        return "result"

    result = outer_function()

    assert result == "result"
    # All spans in the trace exported together when root completes
    assert len(exporter.exported_spans) == 2
    span_names = [s.name for s in exporter.exported_spans]
    assert "outer" in span_names
    assert "inner" in span_names


def test_trace_decorator_with_custom_tracer():
    """Test @trace decorator with custom tracer parameter."""
    exporter1 = MockExporter()
    exporter2 = MockExporter()
    tracer1 = Tracer(service_name="tracer1", exporter=exporter1)
    tracer2 = Tracer(service_name="tracer2", exporter=exporter2)
    tracer1.set_global()

    @trace("custom_tracer_func", tracer=tracer2)
    def custom_function():
        return "result"

    result = custom_function()

    assert result == "result"
    # Should export to tracer2, not global tracer1
    assert len(exporter1.exported_spans) == 0
    assert len(exporter2.exported_spans) == 1
    assert exporter2.exported_spans[0].attributes.get("service.name") == "tracer2"


def test_trace_decorator_no_global_tracer_raises():
    """Test @trace decorator raises when no global tracer."""
    # Reset global tracer
    set_global_tracer(None)

    with pytest.raises(RuntimeError, match="No global tracer set"):

        @trace("test")
        def my_function():
            pass

        my_function()


def test_trace_decorator_preserves_function_metadata():
    """Test @trace decorator preserves function metadata."""
    tracer = Tracer(service_name="test")
    tracer.set_global()

    @trace("documented_function")
    def documented_function(x: int, y: int) -> int:
        """This is a docstring."""
        return x + y

    # Check metadata is preserved
    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a docstring."


def test_trace_decorator_multiple_calls():
    """Test @trace decorator creates span for each call."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("repeated_function")
    def repeated_function(value: int):
        return value * 2

    # Call multiple times
    for i in range(5):
        result = repeated_function(i)
        assert result == i * 2

    # Should have 5 spans exported
    assert len(exporter.exported_spans) == 5
    for i, span in enumerate(exporter.exported_spans):
        assert span.name == "repeated_function"


@pytest.mark.asyncio
async def test_trace_decorator_async_concurrent():
    """Test @trace decorator works with concurrent async calls."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("concurrent_async")
    async def concurrent_operation(id: int):
        await asyncio.sleep(0.01)
        return f"result-{id}"

    # Run multiple async calls concurrently
    tasks = [concurrent_operation(i) for i in range(3)]
    results = await asyncio.gather(*tasks)

    assert results == ["result-0", "result-1", "result-2"]
    assert len(exporter.exported_spans) == 3


def test_trace_decorator_with_return_value():
    """Test @trace decorator returns correct value."""
    tracer = Tracer(service_name="test")
    tracer.set_global()

    @trace("return_test")
    def return_dict():
        return {"key": "value", "count": 42}

    result = return_dict()

    assert result == {"key": "value", "count": 42}
    assert isinstance(result, dict)


def test_trace_decorator_with_no_return():
    """Test @trace decorator works with functions that return None."""
    exporter = MockExporter()
    tracer = Tracer(service_name="test", exporter=exporter)
    tracer.set_global()

    @trace("void_function")
    def void_function():
        pass  # No return statement

    result = void_function()

    assert result is None
    assert len(exporter.exported_spans) == 1
