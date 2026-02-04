"""Tests for evaluation result reporters."""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prela.evals.assertions.base import AssertionResult
from prela.evals.reporters import ConsoleReporter, JSONReporter, JUnitReporter
from prela.evals.runner import CaseResult, EvalRunResult


# Fixtures
@pytest.fixture
def sample_passed_case() -> CaseResult:
    """Create a sample passed case result."""
    return CaseResult(
        case_id="test_001",
        case_name="Basic QA Test",
        passed=True,
        duration_ms=123.5,
        assertion_results=[
            AssertionResult(
                passed=True,
                assertion_type="contains",
                message="Output contains 'Paris'",
                expected="Paris",
                actual="The capital of France is Paris.",
            )
        ],
        output="The capital of France is Paris.",
        trace_id="trace-abc123",
    )


@pytest.fixture
def sample_failed_case() -> CaseResult:
    """Create a sample failed case result."""
    return CaseResult(
        case_id="test_002",
        case_name="Failed Assertion Test",
        passed=False,
        duration_ms=89.2,
        assertion_results=[
            AssertionResult(
                passed=False,
                assertion_type="contains",
                message="Output does not contain 'Berlin'",
                expected="Berlin",
                actual="The capital of France is Paris.",
            ),
            AssertionResult(
                passed=True,
                assertion_type="length",
                message="Length is within bounds",
            ),
        ],
        output="The capital of France is Paris.",
        trace_id="trace-def456",
    )


@pytest.fixture
def sample_error_case() -> CaseResult:
    """Create a sample case with execution error."""
    return CaseResult(
        case_id="test_003",
        case_name="Error Test",
        passed=False,
        duration_ms=45.1,
        assertion_results=[],
        error="ValueError: Invalid input\nTraceback...",
        trace_id="trace-ghi789",
    )


@pytest.fixture
def sample_eval_result(
    sample_passed_case, sample_failed_case, sample_error_case
) -> EvalRunResult:
    """Create a sample evaluation run result."""
    started_at = datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 1, 27, 10, 0, 5, tzinfo=timezone.utc)

    return EvalRunResult(
        suite_name="Geography QA Suite",
        started_at=started_at,
        completed_at=completed_at,
        total_cases=3,
        passed_cases=1,
        failed_cases=2,
        pass_rate=1 / 3,
        case_results=[sample_passed_case, sample_failed_case, sample_error_case],
    )


@pytest.fixture
def perfect_eval_result(sample_passed_case) -> EvalRunResult:
    """Create an evaluation result with all tests passing."""
    started_at = datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 1, 27, 10, 0, 2, tzinfo=timezone.utc)

    return EvalRunResult(
        suite_name="Perfect Suite",
        started_at=started_at,
        completed_at=completed_at,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        case_results=[sample_passed_case],
    )


# ConsoleReporter Tests
class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_initialization_defaults(self):
        """Test reporter initialization with default parameters."""
        reporter = ConsoleReporter()
        assert reporter.verbose is True
        # use_colors depends on whether rich is available
        assert isinstance(reporter.use_colors, bool)

    def test_initialization_custom(self):
        """Test reporter initialization with custom parameters."""
        reporter = ConsoleReporter(verbose=False, use_colors=False)
        assert reporter.verbose is False
        assert reporter.use_colors is False

    def test_report_plain_output(self, sample_eval_result, capsys):
        """Test plain text output (no colors)."""
        reporter = ConsoleReporter(use_colors=False)
        reporter.report(sample_eval_result)

        captured = capsys.readouterr()
        output = captured.out

        # Check for key elements in output
        assert "Geography QA Suite" in output
        assert "Total: 3" in output
        assert "Passed: 1" in output
        assert "Failed: 2" in output
        assert "33.3%" in output  # Pass rate
        assert "Basic QA Test" in output
        assert "Failed Assertion Test" in output
        assert "Error Test" in output

    def test_report_shows_failed_details_verbose(
        self, sample_eval_result, capsys
    ):
        """Test that verbose mode shows failed test details."""
        reporter = ConsoleReporter(verbose=True, use_colors=False)
        reporter.report(sample_eval_result)

        captured = capsys.readouterr()
        output = captured.out

        # Should show failed assertion details
        assert "Output does not contain 'Berlin'" in output
        assert "ValueError: Invalid input" in output

    def test_report_hides_failed_details_non_verbose(
        self, sample_eval_result, capsys
    ):
        """Test that non-verbose mode hides detailed failures."""
        reporter = ConsoleReporter(verbose=False, use_colors=False)
        reporter.report(sample_eval_result)

        captured = capsys.readouterr()
        output = captured.out

        # Should NOT show detailed failure information
        # (only summary and case names)
        assert "Failed Test Details" not in output

    def test_report_perfect_suite(self, perfect_eval_result, capsys):
        """Test output for a suite with all tests passing."""
        reporter = ConsoleReporter(use_colors=False)
        reporter.report(perfect_eval_result)

        captured = capsys.readouterr()
        output = captured.out

        assert "Perfect Suite" in output
        assert "Passed: 1 (100.0%)" in output
        assert "Failed: 0" in output
        # Should not show "Failed Test Details" section
        assert "Failed Test Details" not in output

    def test_truncate_long_values(self):
        """Test that long values are truncated in output."""
        reporter = ConsoleReporter(use_colors=False)

        # Test truncation
        long_value = "x" * 200
        truncated = reporter._truncate(long_value, max_length=50)
        assert len(truncated) == 50
        assert truncated.endswith("...")

        # Test no truncation for short values
        short_value = "short"
        not_truncated = reporter._truncate(short_value, max_length=50)
        assert not_truncated == short_value

    def test_report_with_rich_available(self, sample_eval_result):
        """Test that rich output works when rich is available."""
        try:
            from rich.console import Console

            reporter = ConsoleReporter(use_colors=True)
            # Should not raise an error
            reporter.report(sample_eval_result)
        except ImportError:
            pytest.skip("rich library not available")


# JSONReporter Tests
class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_initialization(self):
        """Test reporter initialization."""
        reporter = JSONReporter("output.json")
        assert reporter.output_path == Path("output.json")
        assert reporter.indent == 2

    def test_initialization_custom_indent(self):
        """Test initialization with custom indent."""
        reporter = JSONReporter("output.json", indent=4)
        assert reporter.indent == 4

    def test_report_creates_file(self, sample_eval_result):
        """Test that report creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JSONReporter(output_path)

            reporter.report(sample_eval_result)

            # File should exist
            assert output_path.exists()

            # File should be valid JSON
            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["suite_name"] == "Geography QA Suite"

    def test_report_creates_parent_directories(self, sample_eval_result):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "results.json"
            reporter = JSONReporter(output_path)

            reporter.report(sample_eval_result)

            # File should exist
            assert output_path.exists()

    def test_json_structure(self, sample_eval_result):
        """Test the structure of the generated JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JSONReporter(output_path)

            reporter.report(sample_eval_result)

            with open(output_path, "r") as f:
                data = json.load(f)

            # Check top-level structure
            assert "suite_name" in data
            assert "started_at" in data
            assert "completed_at" in data
            assert "duration_seconds" in data
            assert "summary" in data
            assert "case_results" in data

            # Check summary
            summary = data["summary"]
            assert summary["total_cases"] == 3
            assert summary["passed_cases"] == 1
            assert summary["failed_cases"] == 2
            assert 0.33 <= summary["pass_rate"] <= 0.34

            # Check case results
            assert len(data["case_results"]) == 3

            # Check first case result structure
            case = data["case_results"][0]
            assert "case_id" in case
            assert "case_name" in case
            assert "passed" in case
            assert "duration_ms" in case
            assert "trace_id" in case
            assert "output" in case
            assert "error" in case
            assert "assertions" in case

    def test_json_assertion_details(self, sample_failed_case):
        """Test that assertion details are included in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            # Create minimal result with just the failed case
            result = EvalRunResult(
                suite_name="Test Suite",
                started_at=datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc),
                completed_at=datetime(
                    2026, 1, 27, 10, 0, 1, tzinfo=timezone.utc
                ),
                total_cases=1,
                passed_cases=0,
                failed_cases=1,
                pass_rate=0.0,
                case_results=[sample_failed_case],
            )

            reporter = JSONReporter(output_path)
            reporter.report(result)

            with open(output_path, "r") as f:
                data = json.load(f)

            # Check assertion structure
            assertions = data["case_results"][0]["assertions"]
            assert len(assertions) == 2

            failed_assertion = assertions[0]
            assert failed_assertion["assertion_type"] == "contains"
            assert failed_assertion["passed"] is False
            assert "Berlin" in failed_assertion["message"]
            assert failed_assertion["expected"] == "Berlin"
            assert "Paris" in failed_assertion["actual"]

    def test_compact_json_output(self, sample_eval_result):
        """Test compact JSON output with no indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JSONReporter(output_path, indent=None)

            reporter.report(sample_eval_result)

            # Read file content
            with open(output_path, "r") as f:
                content = f.read()

            # Compact JSON should not have extra newlines
            assert "\n  " not in content  # No indentation

    def test_serialize_complex_output(self):
        """Test serialization of complex output types."""
        reporter = JSONReporter("output.json")

        # Test various types
        assert reporter._serialize_output(None) is None
        assert reporter._serialize_output(42) == 42
        assert reporter._serialize_output("text") == "text"
        assert reporter._serialize_output([1, 2, 3]) == [1, 2, 3]
        assert reporter._serialize_output({"key": "value"}) == {"key": "value"}

        # Test custom objects (should convert to string)
        class CustomObj:
            def __str__(self):
                return "custom"

        assert reporter._serialize_output(CustomObj()) == "custom"


# JUnitReporter Tests
class TestJUnitReporter:
    """Tests for JUnitReporter."""

    def test_initialization(self):
        """Test reporter initialization."""
        reporter = JUnitReporter("junit.xml")
        assert reporter.output_path == Path("junit.xml")

    def test_report_creates_file(self, sample_eval_result):
        """Test that report creates an XML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "junit.xml"
            reporter = JUnitReporter(output_path)

            reporter.report(sample_eval_result)

            # File should exist
            assert output_path.exists()

            # File should be valid XML
            tree = ET.parse(output_path)
            root = tree.getroot()
            assert root.tag == "testsuite"

    def test_report_creates_parent_directories(self, sample_eval_result):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test-results" / "junit.xml"
            reporter = JUnitReporter(output_path)

            reporter.report(sample_eval_result)

            # File should exist
            assert output_path.exists()

    def test_junit_xml_structure(self, sample_eval_result):
        """Test the structure of the generated JUnit XML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "junit.xml"
            reporter = JUnitReporter(output_path)

            reporter.report(sample_eval_result)

            tree = ET.parse(output_path)
            root = tree.getroot()

            # Check testsuite attributes
            assert root.tag == "testsuite"
            assert root.attrib["name"] == "Geography QA Suite"
            assert root.attrib["tests"] == "3"
            assert root.attrib["failures"] == "2"
            assert root.attrib["errors"] == "0"
            assert "time" in root.attrib
            assert "timestamp" in root.attrib

            # Check testcase elements
            testcases = root.findall("testcase")
            assert len(testcases) == 3

            # Check first testcase (passed)
            assert testcases[0].attrib["name"] == "Basic QA Test"
            assert testcases[0].attrib["classname"] == "Geography QA Suite"
            assert "time" in testcases[0].attrib

            # Should not have failure or error elements
            assert testcases[0].find("failure") is None
            assert testcases[0].find("error") is None

    def test_junit_xml_failed_assertion(self, sample_failed_case):
        """Test that failed assertions are properly recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "junit.xml"

            result = EvalRunResult(
                suite_name="Test Suite",
                started_at=datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc),
                completed_at=datetime(
                    2026, 1, 27, 10, 0, 1, tzinfo=timezone.utc
                ),
                total_cases=1,
                passed_cases=0,
                failed_cases=1,
                pass_rate=0.0,
                case_results=[sample_failed_case],
            )

            reporter = JUnitReporter(output_path)
            reporter.report(result)

            tree = ET.parse(output_path)
            root = tree.getroot()
            testcase = root.find("testcase")

            # Should have failure element
            failure = testcase.find("failure")
            assert failure is not None
            assert failure.attrib["type"] == "AssertionFailure"
            assert "Berlin" in failure.attrib["message"]

            # Check failure details
            assert "contains" in failure.text
            assert "Berlin" in failure.text

    def test_junit_xml_execution_error(self, sample_error_case):
        """Test that execution errors are properly recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "junit.xml"

            result = EvalRunResult(
                suite_name="Test Suite",
                started_at=datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc),
                completed_at=datetime(
                    2026, 1, 27, 10, 0, 1, tzinfo=timezone.utc
                ),
                total_cases=1,
                passed_cases=0,
                failed_cases=1,
                pass_rate=0.0,
                case_results=[sample_error_case],
            )

            reporter = JUnitReporter(output_path)
            reporter.report(result)

            tree = ET.parse(output_path)
            root = tree.getroot()
            testcase = root.find("testcase")

            # Should have error element (not failure)
            error = testcase.find("error")
            assert error is not None
            assert error.attrib["type"] == "ExecutionError"
            assert "ValueError" in error.attrib["message"]
            assert "ValueError: Invalid input" in error.text

    def test_junit_xml_system_out(self, sample_passed_case):
        """Test that system-out includes trace_id and output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "junit.xml"

            result = EvalRunResult(
                suite_name="Test Suite",
                started_at=datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc),
                completed_at=datetime(
                    2026, 1, 27, 10, 0, 1, tzinfo=timezone.utc
                ),
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
                pass_rate=1.0,
                case_results=[sample_passed_case],
            )

            reporter = JUnitReporter(output_path)
            reporter.report(result)

            tree = ET.parse(output_path)
            root = tree.getroot()
            testcase = root.find("testcase")

            # Should have system-out element
            system_out = testcase.find("system-out")
            assert system_out is not None
            assert "trace-abc123" in system_out.text
            assert "Paris" in system_out.text

    def test_junit_xml_pretty_formatted(self, sample_eval_result):
        """Test that XML is pretty-formatted with indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "junit.xml"
            reporter = JUnitReporter(output_path)

            reporter.report(sample_eval_result)

            # Read file content
            with open(output_path, "r") as f:
                content = f.read()

            # Should have XML declaration
            assert '<?xml version=' in content

            # Should have indentation (2 spaces)
            assert "\n  <testcase" in content

    def test_truncate_message(self):
        """Test message truncation for XML attributes."""
        reporter = JUnitReporter("junit.xml")

        # Test truncation
        long_msg = "x" * 300
        truncated = reporter._truncate_message(long_msg, max_length=200)
        assert len(truncated) == 200
        assert truncated.endswith("...")

        # Test no truncation
        short_msg = "short"
        not_truncated = reporter._truncate_message(short_msg, max_length=200)
        assert not_truncated == short_msg

    def test_format_failure_message_single(self):
        """Test formatting of failure message for single assertion."""
        reporter = JUnitReporter("junit.xml")

        assertion = AssertionResult(
            passed=False,
            assertion_type="contains",
            message="Output does not contain 'expected'",
        )

        msg = reporter._format_failure_message([assertion])
        assert msg == "Output does not contain 'expected'"

    def test_format_failure_message_multiple(self):
        """Test formatting of failure message for multiple assertions."""
        reporter = JUnitReporter("junit.xml")

        assertions = [
            AssertionResult(
                passed=False,
                assertion_type="contains",
                message="Missing text",
            ),
            AssertionResult(
                passed=False, assertion_type="length", message="Too long"
            ),
        ]

        msg = reporter._format_failure_message(assertions)
        assert "2 assertions failed" in msg
        assert "contains" in msg
        assert "length" in msg
