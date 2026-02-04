"""
Demonstration of the @trace decorator for automatic function tracing.

This example shows how to use the @trace decorator to automatically
create spans for function execution, both sync and async.

Run with:
    python examples/trace_decorator_demo.py
"""

import asyncio
import time

import prela


# ============================================================================
# Example 1: Basic Sync Function
# ============================================================================


def example_1_basic_sync():
    """Basic sync function tracing."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Sync Function Tracing")
    print("=" * 70)

    # Initialize with console exporter
    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("process_data")
    def process_data(items: list[int]) -> list[int]:
        """Process a list of items."""
        result = [item * 2 for item in items]
        time.sleep(0.1)  # Simulate work
        return result

    # Call the function - automatically traced!
    result = process_data([1, 2, 3, 4, 5])
    print(f"\nResult: {result}")


# ============================================================================
# Example 2: Default Name (Function Name)
# ============================================================================


def example_2_default_name():
    """Using function name as span name."""
    print("\n" + "=" * 70)
    print("Example 2: Default Name (Function Name)")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="normal")

    @prela.trace()  # No name specified - uses function name
    def calculate_total(prices: list[float]) -> float:
        """Calculate total price."""
        return sum(prices)

    total = calculate_total([10.99, 25.50, 8.75])
    print(f"\nTotal: ${total:.2f}")


# ============================================================================
# Example 3: Custom Span Type
# ============================================================================


def example_3_custom_span_type():
    """Using custom span type."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Span Type")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("fetch_documents", span_type=prela.SpanType.RETRIEVAL)
    def fetch_documents(query: str) -> list[dict]:
        """Fetch documents from database."""
        time.sleep(0.05)
        return [
            {"id": 1, "title": "Doc 1", "score": 0.95},
            {"id": 2, "title": "Doc 2", "score": 0.87},
        ]

    docs = fetch_documents("machine learning")
    print(f"\nFound {len(docs)} documents")


# ============================================================================
# Example 4: Initial Attributes
# ============================================================================


def example_4_initial_attributes():
    """Setting initial attributes on spans."""
    print("\n" + "=" * 70)
    print("Example 4: Initial Attributes")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace(
        "database_query",
        span_type=prela.SpanType.CUSTOM,
        attributes={"db.system": "postgresql", "db.table": "users"},
    )
    def query_users(limit: int = 10) -> list[dict]:
        """Query users from database."""
        time.sleep(0.05)
        return [{"id": i, "name": f"User {i}"} for i in range(limit)]

    users = query_users(limit=5)
    print(f"\nQueried {len(users)} users")


# ============================================================================
# Example 5: Adding Dynamic Attributes
# ============================================================================


def example_5_dynamic_attributes():
    """Adding attributes dynamically during execution."""
    print("\n" + "=" * 70)
    print("Example 5: Dynamic Attributes")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("calculate_metrics")
    def calculate_metrics(data: list[int]) -> dict:
        """Calculate metrics with dynamic attributes."""
        # Get current span to add attributes
        span = prela.get_current_span()

        # Add input info
        if span:
            span.set_attribute("input.count", len(data))
            span.set_attribute("input.min", min(data))
            span.set_attribute("input.max", max(data))

        # Calculate
        total = sum(data)
        average = total / len(data)

        # Add output info
        if span:
            span.set_attribute("output.total", total)
            span.set_attribute("output.average", average)

        return {"total": total, "average": average}

    result = calculate_metrics([10, 20, 30, 40, 50])
    print(f"\nMetrics: {result}")


# ============================================================================
# Example 6: Exception Handling
# ============================================================================


def example_6_exception_handling():
    """Automatic exception capture."""
    print("\n" + "=" * 70)
    print("Example 6: Exception Handling")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("divide_numbers")
    def divide_numbers(a: float, b: float) -> float:
        """Divide two numbers."""
        return a / b

    # Successful call
    result = divide_numbers(10, 2)
    print(f"\n10 / 2 = {result}")

    # Failed call - exception captured
    try:
        result = divide_numbers(10, 0)
    except ZeroDivisionError as e:
        print(f"\nCaught expected error: {e}")


# ============================================================================
# Example 7: Async Functions
# ============================================================================


async def example_7_async_functions():
    """Async function tracing."""
    print("\n" + "=" * 70)
    print("Example 7: Async Functions")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("async_fetch", span_type=prela.SpanType.RETRIEVAL)
    async def fetch_data(url: str) -> dict:
        """Async data fetching simulation."""
        await asyncio.sleep(0.1)
        return {"url": url, "status": 200, "data": "example"}

    # Call async function
    result = await fetch_data("https://api.example.com/data")
    print(f"\nFetched: {result}")


# ============================================================================
# Example 8: Nested Decorated Functions
# ============================================================================


def example_8_nested_functions():
    """Nested decorated functions create span hierarchy."""
    print("\n" + "=" * 70)
    print("Example 8: Nested Decorated Functions")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("process_pipeline")
    def process_pipeline(data: list[int]) -> list[int]:
        """Main processing pipeline."""
        # Each step is also traced
        validated = validate_data(data)
        transformed = transform_data(validated)
        enriched = enrich_data(transformed)
        return enriched

    @prela.trace("validate_data")
    def validate_data(data: list[int]) -> list[int]:
        """Validate input data."""
        time.sleep(0.02)
        return [x for x in data if x > 0]

    @prela.trace("transform_data")
    def transform_data(data: list[int]) -> list[int]:
        """Transform data."""
        time.sleep(0.02)
        return [x * 2 for x in data]

    @prela.trace("enrich_data")
    def enrich_data(data: list[int]) -> list[int]:
        """Enrich data."""
        time.sleep(0.02)
        return [x + 100 for x in data]

    result = process_pipeline([1, 2, 3, -1, 4, 5])
    print(f"\nPipeline result: {result}")


# ============================================================================
# Example 9: Concurrent Async Functions
# ============================================================================


async def example_9_concurrent_async():
    """Concurrent async function calls."""
    print("\n" + "=" * 70)
    print("Example 9: Concurrent Async Functions")
    print("=" * 70)

    prela.init(service_name="decorator-demo", exporter="console", verbosity="verbose")

    @prela.trace("async_task")
    async def async_task(task_id: int, delay: float) -> str:
        """Async task simulation."""
        span = prela.get_current_span()
        if span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("task.delay", delay)

        await asyncio.sleep(delay)
        return f"Task {task_id} complete"

    # Run multiple tasks concurrently
    tasks = [
        async_task(1, 0.1),
        async_task(2, 0.05),
        async_task(3, 0.08),
    ]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"  {result}")


# ============================================================================
# Example 10: Custom Tracer Instance
# ============================================================================


def example_10_custom_tracer():
    """Using custom tracer instance."""
    print("\n" + "=" * 70)
    print("Example 10: Custom Tracer Instance")
    print("=" * 70)

    # Create two tracers
    main_tracer = prela.init(service_name="main-service", exporter="console")
    custom_tracer = prela.Tracer(
        service_name="custom-service", exporter=prela.ConsoleExporter(verbosity="normal")
    )

    # Function using global tracer
    @prela.trace("global_function")
    def global_function():
        return "using global tracer"

    # Function using custom tracer
    @prela.trace("custom_function", tracer=custom_tracer)
    def custom_function():
        return "using custom tracer"

    print("\nCalling function with global tracer:")
    result1 = global_function()

    print("\nCalling function with custom tracer:")
    result2 = custom_function()


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PRELA @TRACE DECORATOR EXAMPLES")
    print("=" * 70)

    # Sync examples
    example_1_basic_sync()
    example_2_default_name()
    example_3_custom_span_type()
    example_4_initial_attributes()
    example_5_dynamic_attributes()
    example_6_exception_handling()
    example_8_nested_functions()
    example_10_custom_tracer()

    # Async examples
    print("\n" + "=" * 70)
    print("ASYNC EXAMPLES")
    print("=" * 70)
    asyncio.run(example_7_async_functions())
    asyncio.run(example_9_concurrent_async())

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
