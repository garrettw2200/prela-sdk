"""Demo of Prela CLI usage.

This example shows how to:
1. Initialize Prela with file exporter
2. Generate some traces
3. Use CLI commands to view traces

Run this script, then use the CLI:
    $ python examples/cli_demo.py
    $ prela trace list
    $ prela trace show <trace-id>
    $ prela trace search anthropic
"""

from __future__ import annotations

import prela

# Initialize Prela with file exporter
prela.init(
    service_name="cli-demo",
    exporter="file",
    directory="./demo-traces",
    sample_rate=1.0,
    auto_instrument=True,
)

print("Prela initialized with file exporter")
print("Generating sample traces...\n")

# Example 1: Simple manual span
tracer = prela.get_tracer()
with tracer.span("example_operation") as span:
    span.set_attribute("operation.type", "demo")
    span.set_attribute("operation.complexity", "simple")
    span.add_event("operation.started")

    # Simulate some work
    import time

    time.sleep(0.1)

    span.add_event("operation.completed")

print("✓ Created trace: example_operation")

# Example 2: Nested spans
with tracer.span("parent_task") as parent:
    parent.set_attribute("task.name", "data_processing")

    with tracer.span("subtask_1") as child1:
        child1.set_attribute("subtask.type", "fetch")
        time.sleep(0.05)

    with tracer.span("subtask_2") as child2:
        child2.set_attribute("subtask.type", "transform")
        time.sleep(0.05)

print("✓ Created trace: parent_task (with 2 child spans)")

# Example 3: Anthropic LLM call (if available)
try:
    from anthropic import Anthropic

    client = Anthropic()

    # This will be auto-instrumented
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello!"}],
    )

    print("✓ Created trace: anthropic.messages.create")
except ImportError:
    print("⚠ Anthropic not installed, skipping LLM example")
except Exception as e:
    print(f"⚠ Anthropic call failed: {e}")

# Example 4: Error span
try:
    with tracer.span("failing_operation") as span:
        span.set_attribute("operation.should_fail", True)
        raise ValueError("Intentional error for demo")
except ValueError:
    pass

print("✓ Created trace: failing_operation (with error)")

print("\n" + "=" * 60)
print("Sample traces generated!")
print("=" * 60)
print("\nNow try these CLI commands:\n")
print("  # List all traces")
print("  $ prela trace list\n")
print("  # List traces from last 5 minutes")
print("  $ prela trace list --since 5m\n")
print("  # Show detailed trace (copy trace ID from list)")
print("  $ prela trace show <trace-id>\n")
print("  # Search for traces")
print("  $ prela trace search anthropic")
print("  $ prela trace search error\n")
print("=" * 60)
