"""Tests for evaluation runner."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from prela.core.tracer import Tracer
from prela.evals import (
    CaseResult,
    EvalCase,
    EvalExpected,
    EvalInput,
    EvalRunResult,
    EvalRunner,
    EvalSuite,
    create_assertion,
)
from prela.evals.assertions.base import AssertionResult
from prela.evals.assertions.structural import ContainsAssertion
from prela.exporters.console import ConsoleExporter


# Test CaseResult


def test_case_result_init():
    """Test CaseResult initialization."""
    result = CaseResult(
        case_id="test_1",
        case_name="Test Case",
        passed=True,
        duration_ms=123.45,
        assertion_results=[],
    )
    assert result.case_id == "test_1"
    assert result.case_name == "Test Case"
    assert result.passed is True
    assert result.duration_ms == 123.45
    assert result.assertion_results == []
    assert result.output is None
    assert result.error is None
    assert result.trace_id is None


def test_case_result_with_optional_fields():
    """Test CaseResult with optional fields."""
    assertion = AssertionResult(
        assertion_type="exact_match", passed=True, message="Match found"
    )
    result = CaseResult(
        case_id="test_1",
        case_name="Test Case",
        passed=True,
        duration_ms=123.45,
        assertion_results=[assertion],
        output="output text",
        error="error text",
        trace_id="trace-123",
    )
    assert result.output == "output text"
    assert result.error == "error text"
    assert result.trace_id == "trace-123"
    assert len(result.assertion_results) == 1


def test_case_result_negative_duration():
    """Test CaseResult validation rejects negative duration."""
    with pytest.raises(ValueError, match="duration_ms must be non-negative"):
        CaseResult(
            case_id="test_1",
            case_name="Test",
            passed=True,
            duration_ms=-1.0,
            assertion_results=[],
        )


# Test EvalRunResult


def test_eval_run_result_init():
    """Test EvalRunResult initialization."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    result = EvalRunResult(
        suite_name="Test Suite",
        started_at=started,
        completed_at=completed,
        total_cases=10,
        passed_cases=8,
        failed_cases=2,
        pass_rate=0.8,
    )

    assert result.suite_name == "Test Suite"
    assert result.started_at == started
    assert result.completed_at == completed
    assert result.total_cases == 10
    assert result.passed_cases == 8
    assert result.failed_cases == 2
    assert result.pass_rate == 0.8
    assert result.case_results == []


def test_eval_run_result_validation_negative_values():
    """Test EvalRunResult validation rejects negative values."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="total_cases must be non-negative"):
        EvalRunResult(
            suite_name="Test",
            started_at=started,
            completed_at=completed,
            total_cases=-1,
            passed_cases=0,
            failed_cases=0,
            pass_rate=0.0,
        )


def test_eval_run_result_validation_pass_rate():
    """Test EvalRunResult validation for pass_rate bounds."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="pass_rate must be between 0.0 and 1.0"):
        EvalRunResult(
            suite_name="Test",
            started_at=started,
            completed_at=completed,
            total_cases=10,
            passed_cases=5,
            failed_cases=5,
            pass_rate=1.5,
        )


def test_eval_run_result_validation_case_count_mismatch():
    """Test EvalRunResult validation for case count consistency."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    with pytest.raises(
        ValueError, match="passed_cases \\+ failed_cases must equal total_cases"
    ):
        EvalRunResult(
            suite_name="Test",
            started_at=started,
            completed_at=completed,
            total_cases=10,
            passed_cases=3,
            failed_cases=3,  # 3 + 3 != 10
            pass_rate=0.6,
        )


def test_eval_run_result_summary():
    """Test EvalRunResult.summary() output."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    case_results = [
        CaseResult(
            case_id="test_1",
            case_name="Passing Test",
            passed=True,
            duration_ms=100.0,
            assertion_results=[
                AssertionResult(
                    assertion_type="exact_match",
                    passed=True,
                    message="Match found",
                )
            ],
        ),
        CaseResult(
            case_id="test_2",
            case_name="Failing Test",
            passed=False,
            duration_ms=200.0,
            assertion_results=[
                AssertionResult(
                    assertion_type="contains",
                    passed=False,
                    message="Expected 'foo' not found",
                )
            ],
        ),
    ]

    result = EvalRunResult(
        suite_name="Test Suite",
        started_at=started,
        completed_at=completed,
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        pass_rate=0.5,
        case_results=case_results,
    )

    summary = result.summary()
    assert "Test Suite" in summary
    assert "Total Cases: 2" in summary
    assert "Passed: 1 (50.0%)" in summary
    assert "Failed: 1" in summary
    assert "✓ Passing Test (100.0ms)" in summary
    assert "✗ Failing Test (200.0ms)" in summary
    assert "Expected 'foo' not found" in summary


def test_eval_run_result_summary_all_passed():
    """Test summary when all cases pass."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

    case_results = [
        CaseResult(
            case_id="test_1",
            case_name="Test 1",
            passed=True,
            duration_ms=100.0,
            assertion_results=[],
        )
    ]

    result = EvalRunResult(
        suite_name="Success Suite",
        started_at=started,
        completed_at=completed,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        case_results=case_results,
    )

    summary = result.summary()
    assert "Passed: 1 (100.0%)" in summary
    assert "Failed: 0" in summary


# Test create_assertion factory


def test_create_assertion_json_valid():
    """Test creating JSONValidAssertion."""
    from prela.evals.assertions.structural import JSONValidAssertion

    assertion = create_assertion({"type": "json_valid"})
    assert isinstance(assertion, JSONValidAssertion)


def test_create_assertion_contains():
    """Test creating ContainsAssertion."""
    assertion = create_assertion(
        {"type": "contains", "text": "world", "case_sensitive": False}
    )
    assert isinstance(assertion, ContainsAssertion)
    assert assertion.text == "world"
    assert assertion.case_sensitive is False


def test_create_assertion_missing_type():
    """Test error when type field is missing."""
    with pytest.raises(ValueError, match="must have 'type' field"):
        create_assertion({"substring": "hello"})


def test_create_assertion_unknown_type():
    """Test error for unknown assertion type."""
    with pytest.raises(ValueError, match="Unknown assertion type: unknown_type"):
        create_assertion({"type": "unknown_type"})


def test_create_assertion_invalid_params():
    """Test error when parameters are invalid for assertion type."""
    with pytest.raises(ValueError, match="Invalid parameters"):
        create_assertion({"type": "contains", "wrong_param": "value"})


# Test EvalRunner


def test_eval_runner_init():
    """Test EvalRunner initialization."""
    suite = EvalSuite(name="Test Suite", cases=[])
    agent = Mock()

    runner = EvalRunner(suite, agent)

    assert runner.suite == suite
    assert runner.agent == agent
    assert runner.tracer is None
    assert runner.parallel is False
    assert runner.max_workers == 4
    assert runner.on_case_complete is None


def test_eval_runner_init_with_options():
    """Test EvalRunner initialization with all options."""
    suite = EvalSuite(name="Test Suite", cases=[])
    agent = Mock()
    tracer = Mock()
    callback = Mock()

    runner = EvalRunner(
        suite,
        agent,
        tracer=tracer,
        parallel=True,
        max_workers=8,
        on_case_complete=callback,
    )

    assert runner.tracer == tracer
    assert runner.parallel is True
    assert runner.max_workers == 8
    assert runner.on_case_complete == callback


def test_eval_runner_run_empty_suite():
    """Test running an empty suite."""
    suite = EvalSuite(name="Empty Suite", cases=[])
    agent = Mock()

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.suite_name == "Empty Suite"
    assert result.total_cases == 0
    assert result.passed_cases == 0
    assert result.failed_cases == 0
    assert result.pass_rate == 0.0
    assert result.case_results == []


def test_eval_runner_run_single_case_success():
    """Test running a single passing case."""
    case = EvalCase(
        id="test_1",
        name="Test Case",
        input=EvalInput(query="What is 2+2?"),
        expected=EvalExpected(contains=["4"]),
    )
    suite = EvalSuite(name="Test Suite", cases=[case])

    # Agent that returns expected output
    agent = Mock(return_value="The answer is 4")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.total_cases == 1
    assert result.passed_cases == 1
    assert result.failed_cases == 0
    assert result.pass_rate == 1.0
    assert len(result.case_results) == 1
    assert result.case_results[0].passed is True
    assert result.case_results[0].case_id == "test_1"


def test_eval_runner_run_single_case_failure():
    """Test running a single failing case."""
    case = EvalCase(
        id="test_1",
        name="Test Case",
        input=EvalInput(query="What is 2+2?"),
        expected=EvalExpected(contains=["5"]),  # Wrong answer
    )
    suite = EvalSuite(name="Test Suite", cases=[case])

    agent = Mock(return_value="The answer is 4")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.total_cases == 1
    assert result.passed_cases == 0
    assert result.failed_cases == 1
    assert result.pass_rate == 0.0
    assert len(result.case_results) == 1
    assert result.case_results[0].passed is False


def test_eval_runner_run_multiple_cases():
    """Test running multiple cases."""
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
        EvalCase(
            id="test_3",
            name="Test 3",
            input=EvalInput(query="What is 5+5?"),
            expected=EvalExpected(contains=["10"]),
        ),
    ]
    suite = EvalSuite(name="Math Suite", cases=cases)

    # Agent that returns correct answers
    def agent(input_data):
        if "2+2" in input_data.query:
            return "4"
        elif "3+3" in input_data.query:
            return "6"
        elif "5+5" in input_data.query:
            return "10"
        return "unknown"

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.total_cases == 3
    assert result.passed_cases == 3
    assert result.failed_cases == 0
    assert result.pass_rate == 1.0


def test_eval_runner_run_mixed_results():
    """Test running cases with mixed pass/fail results."""
    cases = [
        EvalCase(
            id="test_1",
            name="Pass",
            input=EvalInput(query="2+2"),
            expected=EvalExpected(contains=["4"]),
        ),
        EvalCase(
            id="test_2",
            name="Fail",
            input=EvalInput(query="3+3"),
            expected=EvalExpected(contains=["7"]),  # Wrong
        ),
    ]
    suite = EvalSuite(name="Suite", cases=cases)

    agent = Mock(side_effect=["4", "6"])

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.total_cases == 2
    assert result.passed_cases == 1
    assert result.failed_cases == 1
    assert result.pass_rate == 0.5


def test_eval_runner_run_agent_exception():
    """Test handling agent execution errors."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["result"]),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(side_effect=ValueError("Agent error"))

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.total_cases == 1
    assert result.passed_cases == 0
    assert result.failed_cases == 1
    assert result.case_results[0].passed is False
    assert result.case_results[0].error is not None
    assert "ValueError: Agent error" in result.case_results[0].error


def test_eval_runner_run_with_tracer():
    """Test running with tracer to capture trace IDs."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["result"]),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="result")
    tracer = Tracer(service_name="test", exporter=ConsoleExporter())

    runner = EvalRunner(suite, agent, tracer=tracer)
    result = runner.run()

    assert result.total_cases == 1
    assert result.passed_cases == 1
    assert result.case_results[0].trace_id is not None


def test_eval_runner_run_with_setup_teardown():
    """Test running with setup and teardown hooks."""
    setup_called = Mock()
    teardown_called = Mock()

    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["result"]),
    )
    suite = EvalSuite(
        name="Suite", cases=[case], setup=setup_called, teardown=teardown_called
    )

    agent = Mock(return_value="result")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    setup_called.assert_called_once()
    teardown_called.assert_called_once()
    assert result.passed_cases == 1


def test_eval_runner_run_setup_failure():
    """Test handling setup failure."""

    def failing_setup():
        raise RuntimeError("Setup failed")

    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["result"]),
    )
    suite = EvalSuite(name="Suite", cases=[case], setup=failing_setup)

    agent = Mock(return_value="result")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.total_cases == 1
    assert result.passed_cases == 0
    assert result.failed_cases == 1
    assert "Setup failed" in result.case_results[0].error


def test_eval_runner_run_with_callback():
    """Test on_case_complete callback is invoked."""
    callback = Mock()

    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["result"]),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="result")

    runner = EvalRunner(suite, agent, on_case_complete=callback)
    result = runner.run()

    callback.assert_called_once()
    called_result = callback.call_args[0][0]
    assert isinstance(called_result, CaseResult)
    assert called_result.case_id == "test_1"


def test_eval_runner_run_parallel():
    """Test running cases in parallel."""
    cases = [
        EvalCase(
            id=f"test_{i}",
            name=f"Test {i}",
            input=EvalInput(query=f"Query {i}"),
            expected=EvalExpected(contains=[str(i)]),
        )
        for i in range(5)
    ]
    suite = EvalSuite(name="Parallel Suite", cases=cases)

    def agent(input_data):
        # Extract number from query
        number = input_data.query.split()[-1]
        time.sleep(0.01)  # Simulate work
        return number

    runner = EvalRunner(suite, agent, parallel=True, max_workers=3)
    result = runner.run()

    assert result.total_cases == 5
    assert result.passed_cases == 5
    assert result.failed_cases == 0


def test_eval_runner_run_case_with_assertions():
    """Test running case with assertion configs."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        assertions=[
            {"type": "contains", "text": "hello"},
            {"type": "contains", "text": "world"},
        ],
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="hello world")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.passed_cases == 1
    assert len(result.case_results[0].assertion_results) == 2
    assert all(r.passed for r in result.case_results[0].assertion_results)


def test_eval_runner_run_case_with_default_assertions():
    """Test running case with suite default assertions."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["hello"]),
    )
    suite = EvalSuite(
        name="Suite",
        cases=[case],
        default_assertions=[{"type": "contains", "text": "world"}],
    )

    agent = Mock(return_value="hello world")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.passed_cases == 1
    # Should have 1 from expected + 1 from default
    assert len(result.case_results[0].assertion_results) == 2


def test_eval_runner_run_case_exact_match():
    """Test running case with exact output match assertion."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(output="hello world"),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="hello world")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.passed_cases == 1
    # Output match uses ContainsAssertion for now
    assert result.case_results[0].assertion_results[0].assertion_type == "contains"


def test_eval_runner_run_case_not_contains():
    """Test running case with not_contains assertion."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(not_contains=["error", "failed"]),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="success")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.passed_cases == 1
    assert len(result.case_results[0].assertion_results) == 2


def test_eval_runner_run_case_assertion_error():
    """Test handling assertion execution errors."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        assertions=[{"type": "regex", "pattern": "[invalid"}],  # Invalid regex
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="test")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.passed_cases == 0
    assert result.failed_cases == 1
    assert "Assertion error" in result.case_results[0].assertion_results[0].message


def test_eval_runner_run_case_duration():
    """Test that case duration is captured."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["result"]),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    def agent(input_data):
        time.sleep(0.01)
        return "result"

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.case_results[0].duration_ms >= 10.0  # At least 10ms


def test_eval_runner_run_case_output_captured():
    """Test that agent output is captured in result."""
    case = EvalCase(
        id="test_1",
        name="Test",
        input=EvalInput(query="test"),
        expected=EvalExpected(contains=["hello"]),
    )
    suite = EvalSuite(name="Suite", cases=[case])

    agent = Mock(return_value="hello world")

    runner = EvalRunner(suite, agent)
    result = runner.run()

    assert result.case_results[0].output == "hello world"
