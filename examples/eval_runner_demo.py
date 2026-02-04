"""
Demo script showing how to use the EvalRunner to test AI agents.

This example demonstrates:
1. Creating evaluation test cases
2. Defining agent functions
3. Running evaluations with assertions
4. Viewing results
5. Parallel execution
6. Trace capture with Tracer integration
"""

from prela.evals import (
    CaseResult,
    EvalCase,
    EvalExpected,
    EvalInput,
    EvalRunner,
    EvalSuite,
)
from prela import init, get_tracer


# Example 1: Simple Math Agent Evaluation
def math_agent(input_data: EvalInput) -> str:
    """Simple agent that answers math questions."""
    query = input_data.query
    if "2+2" in query:
        return "The answer is 4"
    elif "3+3" in query:
        return "The answer is 6"
    elif "10-5" in query:
        return "The answer is 5"
    else:
        return "I don't know"


def example_1_basic_evaluation():
    """Basic evaluation with a math agent."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Math Agent Evaluation")
    print("=" * 80)

    # Create test cases
    cases = [
        EvalCase(
            id="test_addition_1",
            name="Test 2+2",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
            tags=["math", "addition"],
        ),
        EvalCase(
            id="test_addition_2",
            name="Test 3+3",
            input=EvalInput(query="What is 3+3?"),
            expected=EvalExpected(contains=["6"]),
            tags=["math", "addition"],
        ),
        EvalCase(
            id="test_subtraction",
            name="Test 10-5",
            input=EvalInput(query="What is 10-5?"),
            expected=EvalExpected(contains=["5"]),
            tags=["math", "subtraction"],
        ),
    ]

    # Create suite
    suite = EvalSuite(
        name="Math Agent Test Suite",
        description="Tests basic arithmetic capabilities",
        cases=cases,
    )

    # Run evaluation
    runner = EvalRunner(suite, math_agent)
    result = runner.run()

    # Print results
    print(result.summary())
    print(f"\nPass Rate: {result.pass_rate * 100:.1f}%")


# Example 2: Evaluation with Assertions
def example_2_with_assertions():
    """Evaluation using assertion configs instead of expected."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Evaluation with Assertion Configs")
    print("=" * 80)

    cases = [
        EvalCase(
            id="test_length",
            name="Response length check",
            input=EvalInput(query="What is 2+2?"),
            assertions=[
                {"type": "contains", "text": "4"},
                {"type": "length", "min_length": 5, "max_length": 50},
            ],
        ),
        EvalCase(
            id="test_no_errors",
            name="No error messages",
            input=EvalInput(query="What is 3+3?"),
            expected=EvalExpected(
                contains=["6"], not_contains=["error", "unknown"]
            ),
        ),
    ]

    suite = EvalSuite(name="Assertion Suite", cases=cases)
    runner = EvalRunner(suite, math_agent)
    result = runner.run()

    print(result.summary())


# Example 3: Default Assertions
def example_3_default_assertions():
    """Suite with default assertions applied to all cases."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Suite with Default Assertions")
    print("=" * 80)

    cases = [
        EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
        ),
        EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="What is 3+3?"),
            expected=EvalExpected(contains=["6"]),
        ),
    ]

    # All cases will check that response doesn't contain "error"
    suite = EvalSuite(
        name="Suite with Defaults",
        cases=cases,
        default_assertions=[{"type": "not_contains", "text": "error"}],
    )

    runner = EvalRunner(suite, math_agent)
    result = runner.run()

    print(result.summary())


# Example 4: Parallel Execution
def slow_agent(input_data: EvalInput) -> str:
    """Agent that simulates slow processing."""
    import time

    time.sleep(0.05)  # Simulate processing time
    return math_agent(input_data)


def example_4_parallel_execution():
    """Run evaluation cases in parallel."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Parallel Execution")
    print("=" * 80)

    cases = [
        EvalCase(
            id=f"test_{i}",
            name=f"Test case {i}",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
        )
        for i in range(10)
    ]

    suite = EvalSuite(name="Parallel Suite", cases=cases)

    # Sequential execution
    print("\nSequential execution:")
    import time

    start = time.time()
    runner = EvalRunner(suite, slow_agent, parallel=False)
    result = runner.run()
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s")
    print(f"Pass rate: {result.pass_rate * 100:.1f}%")

    # Parallel execution
    print("\nParallel execution (4 workers):")
    start = time.time()
    runner = EvalRunner(suite, slow_agent, parallel=True, max_workers=4)
    result = runner.run()
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")
    print(f"Pass rate: {result.pass_rate * 100:.1f}%")
    print(f"Speedup: {sequential_time / parallel_time:.1f}x")


# Example 5: Setup and Teardown
def example_5_setup_teardown():
    """Evaluation with setup and teardown hooks."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Setup and Teardown Hooks")
    print("=" * 80)

    # Shared state
    state = {"count": 0}

    def setup():
        print("  [Setup] Initializing evaluation...")
        state["count"] = 0

    def teardown():
        print(f"  [Teardown] Ran {state['count']} test cases")

    def counting_agent(input_data: EvalInput) -> str:
        state["count"] += 1
        return math_agent(input_data)

    cases = [
        EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
        ),
        EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="What is 3+3?"),
            expected=EvalExpected(contains=["6"]),
        ),
    ]

    suite = EvalSuite(
        name="Suite with Hooks", cases=cases, setup=setup, teardown=teardown
    )

    runner = EvalRunner(suite, counting_agent)
    result = runner.run()

    print(f"\n{result.summary()}")


# Example 6: Progress Callbacks
def example_6_progress_callbacks():
    """Evaluation with progress callbacks."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Progress Callbacks")
    print("=" * 80)

    completed = [0]

    def on_case_complete(result: CaseResult):
        completed[0] += 1
        status = "✓" if result.passed else "✗"
        print(
            f"  [{completed[0]}/3] {status} {result.case_name} ({result.duration_ms:.1f}ms)"
        )

    cases = [
        EvalCase(
            id=f"test_{i}",
            name=f"Test case {i}",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
        )
        for i in range(3)
    ]

    suite = EvalSuite(name="Progress Suite", cases=cases)

    print("\nRunning tests...")
    runner = EvalRunner(suite, math_agent, on_case_complete=on_case_complete)
    result = runner.run()

    print(f"\nFinal pass rate: {result.pass_rate * 100:.1f}%")


# Example 7: Tracer Integration
def example_7_tracer_integration():
    """Evaluation with trace capture."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Tracer Integration (Trace Capture)")
    print("=" * 80)

    # Initialize tracer with console exporter
    tracer = init(service_name="eval-demo", exporter="console", verbosity="minimal")

    cases = [
        EvalCase(
            id="traced_test",
            name="Traced Test Case",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
        ),
    ]

    suite = EvalSuite(name="Traced Suite", cases=cases)

    # Runner will capture trace_id for each case
    runner = EvalRunner(suite, math_agent, tracer=tracer)
    result = runner.run()

    print(f"\nTrace ID captured: {result.case_results[0].trace_id}")
    print(f"Pass rate: {result.pass_rate * 100:.1f}%")


# Example 8: Error Handling
def buggy_agent(input_data: EvalInput) -> str:
    """Agent that sometimes fails."""
    query = input_data.query
    if "crash" in query.lower():
        raise ValueError("Agent crashed!")
    return math_agent(input_data)


def example_8_error_handling():
    """Evaluation with agent errors."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Error Handling")
    print("=" * 80)

    cases = [
        EvalCase(
            id="good_test",
            name="Good test",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
        ),
        EvalCase(
            id="crash_test",
            name="Crash test",
            input=EvalInput(query="Please crash"),
            expected=EvalExpected(contains=["result"]),
        ),
    ]

    suite = EvalSuite(name="Error Suite", cases=cases)
    runner = EvalRunner(suite, buggy_agent)
    result = runner.run()

    print(result.summary())


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EVAL RUNNER DEMO")
    print("=" * 80)

    # Run all examples
    example_1_basic_evaluation()
    example_2_with_assertions()
    example_3_default_assertions()
    example_4_parallel_execution()
    example_5_setup_teardown()
    example_6_progress_callbacks()
    example_7_tracer_integration()
    example_8_error_handling()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Basic evaluation with EvalRunner")
    print("  ✓ Assertion configs (contains, length, not_contains)")
    print("  ✓ Default assertions applied to all cases")
    print("  ✓ Parallel execution with thread pools")
    print("  ✓ Setup and teardown hooks")
    print("  ✓ Progress callbacks")
    print("  ✓ Tracer integration for trace capture")
    print("  ✓ Error handling for agent failures")
    print()
