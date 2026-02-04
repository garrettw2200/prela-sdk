"""
Evaluation runner for executing test cases against AI agents.

This module provides the core infrastructure for running evaluation suites,
executing test cases, running assertions, and aggregating results.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from prela.core.clock import now
from prela.core.context import get_current_trace_id
from prela.core.tracer import Tracer

from .assertions.base import AssertionResult, BaseAssertion
from .case import EvalCase, EvalInput
from .suite import EvalSuite


@dataclass
class CaseResult:
    """Result of running a single eval case."""

    case_id: str
    case_name: str
    passed: bool
    duration_ms: float
    assertion_results: list[AssertionResult]
    output: Any = None
    error: str | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        """Validate fields."""
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")


@dataclass
class EvalRunResult:
    """Result of running an evaluation suite."""

    suite_name: str
    started_at: datetime
    completed_at: datetime
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    case_results: list[CaseResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate fields."""
        if self.total_cases < 0:
            raise ValueError("total_cases must be non-negative")
        if self.passed_cases < 0:
            raise ValueError("passed_cases must be non-negative")
        if self.failed_cases < 0:
            raise ValueError("failed_cases must be non-negative")
        if not 0.0 <= self.pass_rate <= 1.0:
            raise ValueError("pass_rate must be between 0.0 and 1.0")
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError("passed_cases + failed_cases must equal total_cases")

    def summary(self) -> str:
        """Return human-readable summary of the evaluation run.

        Returns:
            Multi-line string with summary statistics and case results.
        """
        lines = [
            f"Evaluation Suite: {self.suite_name}",
            f"Started: {self.started_at.isoformat()}",
            f"Completed: {self.completed_at.isoformat()}",
            f"Duration: {(self.completed_at - self.started_at).total_seconds():.2f}s",
            "",
            f"Total Cases: {self.total_cases}",
            f"Passed: {self.passed_cases} ({self.pass_rate * 100:.1f}%)",
            f"Failed: {self.failed_cases}",
            "",
            "Case Results:",
        ]

        for result in self.case_results:
            status = "✓" if result.passed else "✗"
            lines.append(
                f"  {status} {result.case_name} ({result.duration_ms:.1f}ms)"
            )
            if not result.passed:
                # Show failed assertions
                for assertion in result.assertion_results:
                    if not assertion.passed:
                        lines.append(f"      - {assertion.message}")

        return "\n".join(lines)


class EvalRunner:
    """Runner for executing evaluation suites against AI agents.

    The runner executes test cases, runs assertions, captures traces,
    and aggregates results. Supports parallel execution with thread pools.

    Example:
        >>> from prela.evals import EvalSuite, EvalRunner
        >>> from prela import get_tracer
        >>>
        >>> suite = EvalSuite.from_yaml("tests.yaml")
        >>> tracer = get_tracer()
        >>>
        >>> def my_agent(input_data):
        ...     # Your agent logic here
        ...     return "agent output"
        >>>
        >>> runner = EvalRunner(suite, my_agent, tracer=tracer)
        >>> result = runner.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        suite: EvalSuite,
        agent: Callable[[EvalInput], Any],
        tracer: Tracer | None = None,
        parallel: bool = False,
        max_workers: int = 4,
        on_case_complete: Callable[[CaseResult], None] | None = None,
    ):
        """Initialize the evaluation runner.

        Args:
            suite: The evaluation suite to run.
            agent: Callable that takes an EvalInput and returns agent output.
            tracer: Optional tracer for capturing execution traces.
            parallel: Whether to run cases in parallel using a thread pool.
            max_workers: Maximum number of worker threads if parallel=True.
            on_case_complete: Optional callback invoked after each case completes.
        """
        self.suite = suite
        self.agent = agent
        self.tracer = tracer
        self.parallel = parallel
        self.max_workers = max_workers
        self.on_case_complete = on_case_complete

    def run(self) -> EvalRunResult:
        """Run all test cases in the evaluation suite.

        Executes setup/teardown hooks, runs all cases (sequentially or in parallel),
        executes assertions, and aggregates results.

        Returns:
            EvalRunResult with aggregated statistics and individual case results.
        """
        started_at = now()

        # Run setup hook if provided
        if self.suite.setup:
            try:
                self.suite.setup()
            except Exception as e:
                # If setup fails, fail the entire run
                return EvalRunResult(
                    suite_name=self.suite.name,
                    started_at=started_at,
                    completed_at=now(),
                    total_cases=len(self.suite.cases),
                    passed_cases=0,
                    failed_cases=len(self.suite.cases),
                    pass_rate=0.0,
                    case_results=[
                        CaseResult(
                            case_id=case.id,
                            case_name=case.name,
                            passed=False,
                            duration_ms=0.0,
                            assertion_results=[],
                            error=f"Setup failed: {str(e)}",
                        )
                        for case in self.suite.cases
                    ],
                )

        # Run all cases
        if self.parallel:
            case_results = self._run_parallel()
        else:
            case_results = self._run_sequential()

        # Run teardown hook if provided
        if self.suite.teardown:
            try:
                self.suite.teardown()
            except Exception as e:
                # Log teardown errors but don't fail the run
                # (results are already collected)
                pass

        completed_at = now()

        # Aggregate results
        passed_cases = sum(1 for r in case_results if r.passed)
        failed_cases = len(case_results) - passed_cases
        pass_rate = passed_cases / len(case_results) if case_results else 0.0

        return EvalRunResult(
            suite_name=self.suite.name,
            started_at=started_at,
            completed_at=completed_at,
            total_cases=len(case_results),
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=pass_rate,
            case_results=case_results,
        )

    def _run_sequential(self) -> list[CaseResult]:
        """Run cases sequentially in the current thread."""
        results = []
        for case in self.suite.cases:
            result = self.run_case(case)
            results.append(result)
            if self.on_case_complete:
                try:
                    self.on_case_complete(result)
                except Exception:
                    # Don't let callback errors affect execution
                    pass
        return results

    def _run_parallel(self) -> list[CaseResult]:
        """Run cases in parallel using a thread pool.

        Note: Each case creates its own context via the tracer, so we don't
        need to propagate the parent context to worker threads.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all cases directly (no context wrapping needed)
            # Each case will create its own trace context if tracer is configured
            future_to_case = {
                executor.submit(self.run_case, case): case
                for case in self.suite.cases
            }

            # Collect results as they complete
            for future in as_completed(future_to_case):
                try:
                    result = future.result()
                    results.append(result)
                    if self.on_case_complete:
                        try:
                            self.on_case_complete(result)
                        except Exception:
                            pass
                except Exception as e:
                    # If case execution fails catastrophically, create error result
                    case = future_to_case[future]
                    results.append(
                        CaseResult(
                            case_id=case.id,
                            case_name=case.name,
                            passed=False,
                            duration_ms=0.0,
                            assertion_results=[],
                            error=f"Execution failed: {str(e)}",
                        )
                    )

        return results

    def run_case(self, case: EvalCase) -> CaseResult:
        """Run a single test case.

        Executes the agent with the case input, runs all assertions,
        captures the trace ID if a tracer is configured, and returns
        aggregated results.

        Args:
            case: The test case to run.

        Returns:
            CaseResult with pass/fail status and assertion results.
        """
        start_time = time.perf_counter_ns()
        output = None
        error = None
        trace_id = None

        # Execute agent
        try:
            # Get agent input from case
            agent_input = case.input

            # If tracer is configured, wrap execution in a span
            if self.tracer:
                from prela.core.span import SpanType

                with self.tracer.span(
                    name=f"eval.case.{case.id}",
                    span_type=SpanType.AGENT,
                    attributes={
                        "eval.case_id": case.id,
                        "eval.case_name": case.name,
                        "eval.tags": ",".join(case.tags) if case.tags else "",
                    },
                ):
                    output = self.agent(agent_input)
                    # Capture trace_id after span is created
                    trace_id = get_current_trace_id()
            else:
                output = self.agent(agent_input)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000

        # Run assertions
        assertion_results = []

        if error:
            # If agent execution failed, mark all assertions as failed
            # (can't run assertions without output)
            if case.expected:
                assertion_results.append(
                    AssertionResult(
                        assertion_type="execution",
                        passed=False,
                        message=f"Agent execution failed: {error}",
                    )
                )
        else:
            # Run assertions from case.expected (if provided)
            if case.expected:
                assertions = self._create_assertions_from_expected(case)
                for assertion in assertions:
                    try:
                        result = assertion.evaluate(output, case.expected, None)
                        assertion_results.append(result)
                    except Exception as e:
                        # If assertion itself crashes, mark as failed
                        assertion_results.append(
                            AssertionResult(
                                assertion_type=type(assertion).__name__,
                                passed=False,
                                message=f"Assertion error: {str(e)}",
                            )
                        )

            # Run assertions from case.assertions (if provided)
            if case.assertions:
                for assertion_config in case.assertions:
                    try:
                        assertion = create_assertion(assertion_config)
                        result = assertion.evaluate(output, case.expected, None)
                        assertion_results.append(result)
                    except Exception as e:
                        assertion_results.append(
                            AssertionResult(
                                assertion_type=assertion_config.get(
                                    "type", "unknown"
                                ),
                                passed=False,
                                message=f"Assertion error: {str(e)}",
                            )
                        )

            # Run default assertions from suite (if any)
            if self.suite.default_assertions:
                for assertion_config in self.suite.default_assertions:
                    try:
                        assertion = create_assertion(assertion_config)
                        result = assertion.evaluate(output, case.expected, None)
                        assertion_results.append(result)
                    except Exception as e:
                        assertion_results.append(
                            AssertionResult(
                                assertion_type=assertion_config.get(
                                    "type", "unknown"
                                ),
                                passed=False,
                                message=f"Default assertion error: {str(e)}",
                            )
                        )

        # Determine overall pass/fail
        passed = (not error) and all(r.passed for r in assertion_results)

        return CaseResult(
            case_id=case.id,
            case_name=case.name,
            passed=passed,
            duration_ms=duration_ms,
            assertion_results=assertion_results,
            output=output,
            error=error,
            trace_id=trace_id,
        )

    def _create_assertions_from_expected(
        self, case: EvalCase
    ) -> list[BaseAssertion]:
        """Create assertion objects from case.expected fields.

        This converts the EvalExpected fields (output, contains, not_contains, etc.)
        into actual assertion objects that can be run.

        Args:
            case: The eval case with expected output.

        Returns:
            List of assertion objects.
        """
        from .assertions.structural import ContainsAssertion, NotContainsAssertion

        assertions: list[BaseAssertion] = []

        if case.expected is None:
            return assertions

        # Exact output match
        if case.expected.output is not None:
            # For now, use contains with the exact string
            # TODO: Create a dedicated ExactMatchAssertion
            assertions.append(
                ContainsAssertion(text=case.expected.output, case_sensitive=True)
            )

        # Contains (all must be present)
        if case.expected.contains:
            for text in case.expected.contains:
                assertions.append(
                    ContainsAssertion(text=text, case_sensitive=True)
                )

        # Not contains (none must be present)
        if case.expected.not_contains:
            for text in case.expected.not_contains:
                assertions.append(
                    NotContainsAssertion(text=text, case_sensitive=True)
                )

        # TODO: Add tool_calls and metadata assertions when implemented

        return assertions


def create_assertion(config: dict) -> BaseAssertion:
    """Factory function to create assertion instances from configuration.

    This maps assertion type strings to concrete assertion classes and
    instantiates them with the provided configuration.

    Args:
        config: Dictionary with "type" key and type-specific parameters.

    Returns:
        Instantiated assertion object.

    Raises:
        ValueError: If assertion type is unknown or configuration is invalid.

    Example:
        >>> assertion = create_assertion({
        ...     "type": "contains",
        ...     "text": "hello",
        ...     "case_sensitive": False
        ... })
        >>> result = assertion.evaluate("Hello world", None, None)
        >>> assert result.passed
    """
    from .assertions.semantic import SemanticSimilarityAssertion
    from .assertions.structural import (
        ContainsAssertion,
        JSONValidAssertion,
        LengthAssertion,
        NotContainsAssertion,
        RegexAssertion,
    )
    from .assertions.tool import (
        ToolArgsAssertion,
        ToolCalledAssertion,
        ToolSequenceAssertion,
    )

    assertion_type = config.get("type")
    if not assertion_type:
        raise ValueError("Assertion config must have 'type' field")

    # Map type strings to classes
    registry: dict[str, type[BaseAssertion]] = {
        "contains": ContainsAssertion,
        "not_contains": NotContainsAssertion,
        "regex": RegexAssertion,
        "length": LengthAssertion,
        "json_valid": JSONValidAssertion,
        "semantic_similarity": SemanticSimilarityAssertion,
        "tool_called": ToolCalledAssertion,
        "tool_args": ToolArgsAssertion,
        "tool_sequence": ToolSequenceAssertion,
    }

    assertion_class = registry.get(assertion_type)
    if not assertion_class:
        raise ValueError(
            f"Unknown assertion type: {assertion_type}. "
            f"Available types: {', '.join(registry.keys())}"
        )

    # Extract parameters (everything except 'type')
    params = {k: v for k, v in config.items() if k != "type"}

    try:
        return assertion_class(**params)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for {assertion_type} assertion: {str(e)}"
        ) from e
