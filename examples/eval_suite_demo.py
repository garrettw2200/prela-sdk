"""Demo: Using the Prela evaluation framework.

This script demonstrates how to define and use eval cases and suites
for testing AI agents.
"""

from pathlib import Path

from prela.evals import EvalCase, EvalExpected, EvalInput, EvalSuite


def example_1_basic_case():
    """Example 1: Create a basic eval case."""
    print("\n=== Example 1: Basic Eval Case ===")

    case = EvalCase(
        id="test_basic_qa",
        name="Basic factual question",
        input=EvalInput(query="What is the capital of France?"),
        expected=EvalExpected(contains=["Paris"]),
    )

    print(f"Case ID: {case.id}")
    print(f"Case name: {case.name}")
    print(f"Input query: {case.input.query}")
    print(f"Expected to contain: {case.expected.contains}")
    print(f"Agent input format: {case.input.to_agent_input()}")


def example_2_advanced_case():
    """Example 2: Advanced case with multiple expectations."""
    print("\n=== Example 2: Advanced Eval Case ===")

    case = EvalCase(
        id="test_rag_quality",
        name="RAG quality test",
        input=EvalInput(
            query="Summarize the key findings from the Q4 2024 report",
            context={
                "document": "Q4 2024 report showed revenue growth of 15%...",
                "metadata": {"source": "financial_reports", "date": "2024-12-31"},
            },
        ),
        expected=EvalExpected(
            contains=["revenue", "15%", "growth"],
            not_contains=["error", "failed"],
        ),
        assertions=[
            {"type": "contains", "value": "revenue"},
            {"type": "semantic_similarity", "threshold": 0.8, "reference": "Revenue grew by 15%"},
            {"type": "latency", "max_ms": 5000},
        ],
        tags=["rag", "summarization", "financial"],
        timeout_seconds=10.0,
    )

    print(f"Case: {case.name}")
    print(f"Tags: {case.tags}")
    print(f"Assertions: {len(case.assertions)} configured")
    print(f"Expected to contain: {case.expected.contains}")
    print(f"Must not contain: {case.expected.not_contains}")


def example_3_suite_creation():
    """Example 3: Create a test suite with multiple cases."""
    print("\n=== Example 3: Create Test Suite ===")

    # Create test cases
    cases = [
        EvalCase(
            id="test_qa_geography",
            name="Geography question",
            input=EvalInput(query="What is the capital of France?"),
            expected=EvalExpected(contains=["Paris"]),
            tags=["qa", "geography"],
        ),
        EvalCase(
            id="test_qa_math",
            name="Math question",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
            assertions=[
                {"type": "contains", "value": "4"},
                {"type": "exact_match", "value": "4"},
            ],
            tags=["qa", "math"],
        ),
        EvalCase(
            id="test_qa_history",
            name="History question",
            input=EvalInput(query="When did World War II end?"),
            expected=EvalExpected(contains=["1945"]),
            tags=["qa", "history"],
        ),
    ]

    # Create suite
    suite = EvalSuite(
        name="Basic QA Suite",
        description="Tests for basic question answering across different domains",
        cases=cases,
        default_assertions=[
            {"type": "no_errors"},
            {"type": "latency", "max_ms": 5000},
        ],
    )

    print(f"Suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Number of cases: {len(suite)}")
    print(f"Default assertions: {len(suite.default_assertions)}")

    # Filter by tags
    qa_cases = suite.filter_by_tags(["qa"])
    print(f"\nCases with 'qa' tag: {len(qa_cases)}")

    geography_qa = suite.filter_by_tags(["qa", "geography"])
    print(f"Cases with both 'qa' and 'geography' tags: {len(geography_qa)}")


def example_4_yaml_serialization():
    """Example 4: Save and load suite from YAML."""
    print("\n=== Example 4: YAML Serialization ===")

    try:
        import yaml  # noqa: F401

        yaml_available = True
    except ImportError:
        yaml_available = False
        print("PyYAML not installed. Install with: pip install pyyaml")
        return

    # Create a comprehensive suite
    suite = EvalSuite(
        name="RAG Quality Suite",
        description="Tests for RAG pipeline quality",
        cases=[
            EvalCase(
                id="test_basic_qa",
                name="Basic factual question",
                input=EvalInput(query="What is the capital of France?"),
                expected=EvalExpected(contains=["Paris"]),
                assertions=[
                    {"type": "contains", "value": "Paris"},
                    {"type": "semantic_similarity", "threshold": 0.8},
                ],
                tags=["qa", "geography"],
            ),
            EvalCase(
                id="test_rag_retrieval",
                name="Document retrieval accuracy",
                input=EvalInput(
                    query="What were the key findings?",
                    context={"document_ids": ["doc1", "doc2"]},
                ),
                expected=EvalExpected(
                    contains=["finding", "result"],
                    metadata={"retrieved_docs": 2},
                ),
                tags=["rag", "retrieval"],
                timeout_seconds=15.0,
            ),
        ],
        default_assertions=[
            {"type": "no_errors"},
            {"type": "latency", "max_ms": 5000},
        ],
    )

    # Save to YAML
    yaml_path = Path("example_suite.yaml")
    suite.to_yaml(yaml_path)
    print(f"Suite saved to: {yaml_path}")

    # Load from YAML
    loaded_suite = EvalSuite.from_yaml(yaml_path)
    print(f"Suite loaded from YAML: {loaded_suite.name}")
    print(f"Number of cases: {len(loaded_suite)}")

    # Display YAML content
    print("\nYAML content:")
    with open(yaml_path, "r") as f:
        print(f.read())

    # Cleanup
    yaml_path.unlink()


def example_5_message_based_input():
    """Example 5: Using message-based input for chat agents."""
    print("\n=== Example 5: Message-Based Input ===")

    case = EvalCase(
        id="test_chat_agent",
        name="Multi-turn conversation",
        input=EvalInput(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to weather data."},
                {"role": "user", "content": "Can you tell me a joke instead?"},
            ]
        ),
        expected=EvalExpected(
            contains=["joke", "funny"], not_contains=["error", "cannot"]
        ),
        tags=["chat", "conversation"],
    )

    print(f"Case: {case.name}")
    print(f"Number of messages: {len(case.input.messages)}")
    print(f"Agent input format:")
    agent_input = case.input.to_agent_input()
    for i, msg in enumerate(agent_input["messages"]):
        print(f"  Message {i+1} ({msg['role']}): {msg['content'][:50]}...")


def example_6_tool_call_validation():
    """Example 6: Validating tool calls in agent output."""
    print("\n=== Example 6: Tool Call Validation ===")

    case = EvalCase(
        id="test_tool_usage",
        name="Correct tool selection",
        input=EvalInput(query="What's the weather in San Francisco?"),
        expected=EvalExpected(
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "San Francisco", "units": "fahrenheit"},
                }
            ]
        ),
        assertions=[
            {"type": "tool_called", "tool_name": "get_weather"},
            {
                "type": "tool_args_match",
                "tool_name": "get_weather",
                "expected_args": {"location": "San Francisco"},
            },
        ],
        tags=["tool_use", "weather"],
    )

    print(f"Case: {case.name}")
    print(f"Expected tool calls:")
    for tool_call in case.expected.tool_calls:
        print(f"  - {tool_call['name']} with args: {tool_call['args']}")


def example_7_suite_iteration():
    """Example 7: Iterating through suite cases."""
    print("\n=== Example 7: Suite Iteration ===")

    suite = EvalSuite(
        name="Example Suite",
        cases=[
            EvalCase(
                id=f"test_{i}",
                name=f"Test case {i}",
                input=EvalInput(query=f"Question {i}"),
                expected=EvalExpected(contains=[f"answer_{i}"]),
            )
            for i in range(1, 4)
        ],
    )

    print(f"Suite: {suite.name}")
    print(f"Total cases: {len(suite)}")

    # Iterate using for loop
    print("\nIterating through cases:")
    for case in suite:
        print(f"  - {case.id}: {case.name}")

    # Access by index
    print(f"\nFirst case: {suite[0].name}")
    print(f"Last case: {suite[-1].name}")

    # Get specific case by ID
    case = suite.get_case("test_2")
    if case:
        print(f"\nFound case by ID: {case.name}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Prela Evaluation Framework - Examples")
    print("=" * 70)

    example_1_basic_case()
    example_2_advanced_case()
    example_3_suite_creation()
    example_4_yaml_serialization()
    example_5_message_based_input()
    example_6_tool_call_validation()
    example_7_suite_iteration()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
