"""
Demo script showing how to use evaluation assertions.

This script demonstrates all available assertion types:
- Structural: ContainsAssertion, NotContainsAssertion, RegexAssertion, LengthAssertion, JSONValidAssertion
- Tool: ToolCalledAssertion, ToolArgsAssertion, ToolSequenceAssertion
- Semantic: SemanticSimilarityAssertion (requires sentence-transformers)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from prela.core.span import Span, SpanType
from prela.evals.assertions import (
    ContainsAssertion,
    JSONValidAssertion,
    LengthAssertion,
    NotContainsAssertion,
    RegexAssertion,
    ToolArgsAssertion,
    ToolCalledAssertion,
    ToolSequenceAssertion,
)


def print_result(assertion, output, trace=None):
    """Helper to print assertion results."""
    result = assertion.evaluate(output=output, expected=None, trace=trace)
    print(f"  {result}")
    return result


def demo_structural_assertions():
    """Demonstrate structural text assertions."""
    print("\n" + "=" * 70)
    print("STRUCTURAL ASSERTIONS")
    print("=" * 70)

    # ContainsAssertion
    print("\n1. ContainsAssertion - Check if output contains text")
    print("-" * 70)
    assertion = ContainsAssertion(text="success", case_sensitive=False)
    print(f"Assertion: {assertion}")
    print_result(assertion, output="Operation completed successfully!")
    print_result(assertion, output="Failed to complete operation")

    # NotContainsAssertion
    print("\n2. NotContainsAssertion - Check if output does NOT contain text")
    print("-" * 70)
    assertion = NotContainsAssertion(text="error", case_sensitive=True)
    print(f"Assertion: {assertion}")
    print_result(assertion, output="All tests passed!")
    print_result(assertion, output="Error: File not found")

    # RegexAssertion
    print("\n3. RegexAssertion - Match against regex pattern")
    print("-" * 70)
    assertion = RegexAssertion(pattern=r"\d{3}-\d{3}-\d{4}")
    print(f"Assertion: {assertion}")
    print_result(assertion, output="Call me at 555-123-4567")
    print_result(assertion, output="No phone number here")

    # LengthAssertion
    print("\n4. LengthAssertion - Check output length bounds")
    print("-" * 70)
    assertion = LengthAssertion(min_length=10, max_length=50)
    print(f"Assertion: {assertion}")
    print_result(assertion, output="This is a medium length response.")
    print_result(assertion, output="Too short")
    print_result(
        assertion,
        output="This is a very long response that exceeds the maximum length limit",
    )

    # JSONValidAssertion
    print("\n5. JSONValidAssertion - Validate JSON output")
    print("-" * 70)
    assertion = JSONValidAssertion()
    print(f"Assertion: {assertion}")
    print_result(assertion, output='{"status": "success", "count": 42}')
    print_result(assertion, output="{invalid json}")

    # JSONValidAssertion with schema (requires jsonschema)
    print("\n6. JSONValidAssertion with schema")
    print("-" * 70)
    try:
        import jsonschema

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name"],
        }
        assertion = JSONValidAssertion(schema=schema)
        print(f"Assertion: JSONValidAssertion(schema={json.dumps(schema, indent=2)})")
        print_result(assertion, output='{"name": "Alice", "age": 30}')
        print_result(assertion, output='{"age": 30}')  # Missing required field
    except ImportError:
        print("  (Skipped - jsonschema not installed)")


def demo_tool_assertions():
    """Demonstrate tool-related assertions."""
    print("\n" + "=" * 70)
    print("TOOL ASSERTIONS")
    print("=" * 70)

    # Create sample trace with tool spans
    trace = [
        Span(
            name="web_search",
            span_type=SpanType.TOOL,
            trace_id="trace-123",
            span_id="span-1",
            started_at=datetime.now(timezone.utc),
        ),
        Span(
            name="calculator",
            span_type=SpanType.TOOL,
            trace_id="trace-123",
            span_id="span-2",
            started_at=datetime.now(timezone.utc),
        ),
        Span(
            name="summarize",
            span_type=SpanType.TOOL,
            trace_id="trace-123",
            span_id="span-3",
            started_at=datetime.now(timezone.utc),
        ),
    ]

    # Add attributes to tool spans
    trace[0].set_attribute("tool.input.query", "Python tutorial")
    trace[0].set_attribute("tool.input.limit", 10)
    trace[1].set_attribute("tool.input.expression", "2 + 2")
    trace[2].set_attribute("tool.input.text", "Long article...")

    # ToolCalledAssertion
    print("\n7. ToolCalledAssertion - Check if tool was called")
    print("-" * 70)
    assertion = ToolCalledAssertion(tool_name="web_search")
    print(f"Assertion: {assertion}")
    print_result(assertion, output=None, trace=trace)

    assertion = ToolCalledAssertion(tool_name="nonexistent_tool")
    print(f"Assertion: {assertion}")
    print_result(assertion, output=None, trace=trace)

    # ToolArgsAssertion
    print("\n8. ToolArgsAssertion - Check tool arguments")
    print("-" * 70)
    assertion = ToolArgsAssertion(
        tool_name="web_search",
        expected_args={"query": "Python tutorial"},
        partial_match=True,
    )
    print(f"Assertion: {assertion}")
    print_result(assertion, output=None, trace=trace)

    assertion = ToolArgsAssertion(
        tool_name="web_search",
        expected_args={"query": "Java tutorial"},
    )
    print(f"Assertion: {assertion}")
    print_result(assertion, output=None, trace=trace)

    # ToolSequenceAssertion
    print("\n9. ToolSequenceAssertion - Check tool call order")
    print("-" * 70)
    assertion = ToolSequenceAssertion(
        sequence=["web_search", "calculator", "summarize"],
        strict=False,
    )
    print(f"Assertion: {assertion}")
    print_result(assertion, output=None, trace=trace)

    assertion = ToolSequenceAssertion(
        sequence=["calculator", "web_search"],  # Wrong order
    )
    print(f"Assertion: {assertion}")
    print_result(assertion, output=None, trace=trace)


def demo_semantic_assertions():
    """Demonstrate semantic similarity assertions."""
    print("\n" + "=" * 70)
    print("SEMANTIC ASSERTIONS (requires sentence-transformers)")
    print("=" * 70)

    try:
        from prela.evals.assertions import SemanticSimilarityAssertion

        # SemanticSimilarityAssertion
        print("\n10. SemanticSimilarityAssertion - Check semantic similarity")
        print("-" * 70)
        assertion = SemanticSimilarityAssertion(
            expected_text="The weather is nice today",
            threshold=0.7,
        )
        print(f"Assertion: {assertion}")

        # High similarity (different wording, same meaning)
        result = print_result(assertion, output="Today has beautiful weather")
        print(f"  Similarity score: {result.score:.3f}")

        # Low similarity (different meaning)
        result = print_result(assertion, output="I like pizza")
        print(f"  Similarity score: {result.score:.3f}")

        # Very high similarity (nearly identical)
        result = print_result(assertion, output="The weather is great today")
        print(f"  Similarity score: {result.score:.3f}")

    except ImportError:
        print("\n  (Skipped - sentence-transformers not installed)")
        print("  Install with: pip install sentence-transformers")


def demo_config_based_loading():
    """Demonstrate loading assertions from config."""
    print("\n" + "=" * 70)
    print("CONFIG-BASED ASSERTION LOADING")
    print("=" * 70)

    print("\n11. Loading assertions from configuration dictionaries")
    print("-" * 70)

    configs = [
        {
            "type": "contains",
            "config": {"text": "success", "case_sensitive": False},
        },
        {
            "type": "regex",
            "config": {"pattern": r"\d+"},
        },
        {
            "type": "length",
            "config": {"min_length": 5, "max_length": 100},
        },
        {
            "type": "tool_called",
            "config": {"tool_name": "web_search"},
        },
    ]

    # Map assertion types to classes
    assertion_types = {
        "contains": ContainsAssertion,
        "regex": RegexAssertion,
        "length": LengthAssertion,
        "tool_called": ToolCalledAssertion,
    }

    for item in configs:
        assertion_class = assertion_types[item["type"]]
        assertion = assertion_class.from_config(item["config"])
        print(f"  Loaded: {assertion}")


def demo_assertion_results():
    """Demonstrate working with assertion results."""
    print("\n" + "=" * 70)
    print("WORKING WITH ASSERTION RESULTS")
    print("=" * 70)

    print("\n12. Inspecting assertion results")
    print("-" * 70)

    assertion = ContainsAssertion(text="hello")
    result = assertion.evaluate(output="hello world", expected=None, trace=None)

    print(f"  Passed: {result.passed}")
    print(f"  Type: {result.assertion_type}")
    print(f"  Message: {result.message}")
    print(f"  Expected: {result.expected}")
    print(f"  Actual: {result.actual}")
    print(f"  Details: {result.details}")
    print(f"  String representation: {result}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("PRELA EVALUATION ASSERTIONS DEMO")
    print("=" * 70)

    demo_structural_assertions()
    demo_tool_assertions()
    demo_semantic_assertions()
    demo_config_based_loading()
    demo_assertion_results()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - sdk/prela/evals/assertions/")
    print("  - tests/test_evals/test_assertions.py")
    print()


if __name__ == "__main__":
    main()
