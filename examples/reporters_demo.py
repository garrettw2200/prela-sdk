"""Demo script showing how to use evaluation reporters.

This script demonstrates the three reporter types:
1. ConsoleReporter - Pretty terminal output with colors
2. JSONReporter - Structured JSON file output
3. JUnitReporter - JUnit XML for CI/CD integration
"""

from datetime import datetime, timezone
from pathlib import Path

from prela.evals.assertions.base import AssertionResult
from prela.evals.reporters import ConsoleReporter, JSONReporter, JUnitReporter
from prela.evals.runner import CaseResult, EvalRunResult


def create_sample_results() -> EvalRunResult:
    """Create sample evaluation results for demonstration."""

    # Create some test case results
    passed_case = CaseResult(
        case_id="test_geography_001",
        case_name="Capital of France",
        passed=True,
        duration_ms=145.3,
        assertion_results=[
            AssertionResult(
                passed=True,
                assertion_type="contains",
                message="Output contains 'Paris'",
                expected="Paris",
                actual="The capital of France is Paris.",
            ),
            AssertionResult(
                passed=True,
                assertion_type="semantic_similarity",
                message="High semantic similarity to expected answer",
                score=0.95,
                expected="Paris is the capital.",
                actual="The capital of France is Paris.",
            ),
        ],
        output="The capital of France is Paris.",
        trace_id="trace-abc-123",
    )

    failed_case = CaseResult(
        case_id="test_geography_002",
        case_name="Capital of Germany",
        passed=False,
        duration_ms=198.7,
        assertion_results=[
            AssertionResult(
                passed=False,
                assertion_type="contains",
                message="Output does not contain 'Berlin'",
                expected="Berlin",
                actual="The capital of Germany is Munich.",
            ),
            AssertionResult(
                passed=True,
                assertion_type="length",
                message="Length is within acceptable range",
            ),
        ],
        output="The capital of Germany is Munich.",
        trace_id="trace-def-456",
    )

    error_case = CaseResult(
        case_id="test_geography_003",
        case_name="Capital of Invalid Country",
        passed=False,
        duration_ms=23.1,
        assertion_results=[],
        output=None,
        error="ValueError: Country 'Atlantis' not found\n  File 'agent.py', line 42, in get_capital",
        trace_id="trace-ghi-789",
    )

    another_passed = CaseResult(
        case_id="test_geography_004",
        case_name="Capital of Japan",
        passed=True,
        duration_ms=132.9,
        assertion_results=[
            AssertionResult(
                passed=True,
                assertion_type="contains",
                message="Output contains 'Tokyo'",
                expected="Tokyo",
                actual="Tokyo is the capital of Japan.",
            )
        ],
        output="Tokyo is the capital of Japan.",
        trace_id="trace-jkl-012",
    )

    # Create evaluation run result
    started_at = datetime(2026, 1, 27, 14, 30, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 1, 27, 14, 30, 12, tzinfo=timezone.utc)

    return EvalRunResult(
        suite_name="Geography QA Suite",
        started_at=started_at,
        completed_at=completed_at,
        total_cases=4,
        passed_cases=2,
        failed_cases=2,
        pass_rate=0.5,
        case_results=[passed_case, failed_case, error_case, another_passed],
    )


def demo_console_reporter():
    """Demonstrate ConsoleReporter with different settings."""
    print("=" * 70)
    print("DEMO 1: Console Reporter (Verbose, with colors)")
    print("=" * 70)
    print()

    result = create_sample_results()

    # Default: verbose, with colors
    reporter = ConsoleReporter(verbose=True, use_colors=True)
    reporter.report(result)

    print()
    print()
    print("=" * 70)
    print("DEMO 2: Console Reporter (Non-verbose, plain text)")
    print("=" * 70)
    print()

    # Non-verbose, no colors
    reporter_plain = ConsoleReporter(verbose=False, use_colors=False)
    reporter_plain.report(result)


def demo_json_reporter():
    """Demonstrate JSONReporter."""
    print()
    print()
    print("=" * 70)
    print("DEMO 3: JSON Reporter")
    print("=" * 70)
    print()

    result = create_sample_results()

    # Create output directory
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)

    # Write to JSON file
    json_path = output_dir / "geography_qa_results.json"
    reporter = JSONReporter(json_path, indent=2)
    reporter.report(result)

    print(f"✓ JSON report written to: {json_path}")
    print()
    print("First 30 lines of JSON output:")
    print("-" * 70)

    # Show first 30 lines of the JSON file
    with open(json_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:30], 1):
            print(f"{i:2d}: {line}", end="")

    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")


def demo_junit_reporter():
    """Demonstrate JUnitReporter."""
    print()
    print()
    print("=" * 70)
    print("DEMO 4: JUnit XML Reporter")
    print("=" * 70)
    print()

    result = create_sample_results()

    # Create output directory
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)

    # Write to JUnit XML file
    junit_path = output_dir / "junit_results.xml"
    reporter = JUnitReporter(junit_path)
    reporter.report(result)

    print(f"✓ JUnit XML report written to: {junit_path}")
    print()
    print("JUnit XML output:")
    print("-" * 70)

    # Show the XML file
    with open(junit_path, "r") as f:
        print(f.read())


def demo_multiple_reporters():
    """Demonstrate using multiple reporters together."""
    print()
    print()
    print("=" * 70)
    print("DEMO 5: Using Multiple Reporters Together")
    print("=" * 70)
    print()

    result = create_sample_results()

    # Create output directory
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)

    # Use all three reporters
    reporters = [
        ConsoleReporter(verbose=False, use_colors=True),
        JSONReporter(output_dir / "multi_results.json"),
        JUnitReporter(output_dir / "multi_junit.xml"),
    ]

    print("Running evaluation with 3 reporters:")
    print(f"  1. Console (terminal output)")
    print(f"  2. JSON ({output_dir / 'multi_results.json'})")
    print(f"  3. JUnit ({output_dir / 'multi_junit.xml'})")
    print()

    for reporter in reporters:
        reporter.report(result)

    print()
    print("✓ All reports generated successfully!")


def demo_ci_integration():
    """Show example CI/CD integration commands."""
    print()
    print()
    print("=" * 70)
    print("DEMO 6: CI/CD Integration Examples")
    print("=" * 70)
    print()

    print("Example GitHub Actions workflow:")
    print("-" * 70)
    print("""
name: Run Evaluations

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install prela[evals]
          pip install -r requirements.txt

      - name: Run evaluations
        run: |
          python run_evals.py  # Your eval script using JUnitReporter

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: junit-results
          path: eval_results/junit.xml

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: eval_results/junit.xml
""")

    print()
    print("Example GitLab CI configuration:")
    print("-" * 70)
    print("""
test:
  script:
    - pip install prela[evals]
    - python run_evals.py
  artifacts:
    when: always
    reports:
      junit: eval_results/junit.xml
    paths:
      - eval_results/
""")


def main():
    """Run all demos."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PRELA EVALUATION REPORTERS DEMO" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_console_reporter()
    demo_json_reporter()
    demo_junit_reporter()
    demo_multiple_reporters()
    demo_ci_integration()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Three reporter types demonstrated:")
    print()
    print("1. ConsoleReporter")
    print("   - Beautiful terminal output with colors")
    print("   - Verbose/non-verbose modes")
    print("   - Perfect for development and debugging")
    print()
    print("2. JSONReporter")
    print("   - Structured JSON output")
    print("   - Programmatic access to results")
    print("   - Data analysis and historical tracking")
    print()
    print("3. JUnitReporter")
    print("   - JUnit XML format")
    print("   - CI/CD integration (Jenkins, GitHub Actions, GitLab)")
    print("   - Test result visualization in CI platforms")
    print()
    print("All output files are in: eval_results/")
    print()


if __name__ == "__main__":
    main()
