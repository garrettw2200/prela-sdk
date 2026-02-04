"""JSON reporter for evaluation results.

This module provides a reporter that writes evaluation results to a JSON file,
suitable for programmatic access, data analysis, or integration with other tools.
"""

from __future__ import annotations

import json
from pathlib import Path

from prela.evals.runner import EvalRunResult


class JSONReporter:
    """Reporter that writes evaluation results to a JSON file.

    Outputs a structured JSON file containing all evaluation data:
    - Suite metadata (name, timestamps, duration)
    - Summary statistics (total, passed, failed, pass rate)
    - Individual case results with assertion details
    - Full error messages and stack traces

    The JSON format is designed for:
    - Programmatic analysis of test results
    - Integration with data processing pipelines
    - Historical comparison of evaluation runs
    - CI/CD artifact storage

    Example:
        >>> from prela.evals import EvalRunner
        >>> from prela.evals.reporters import JSONReporter
        >>>
        >>> runner = EvalRunner(suite, agent)
        >>> result = runner.run()
        >>>
        >>> reporter = JSONReporter("results/eval_run_123.json")
        >>> reporter.report(result)
        # Creates results/eval_run_123.json with full results
    """

    def __init__(self, output_path: str | Path, indent: int = 2):
        """Initialize the JSON reporter.

        Args:
            output_path: Path where the JSON file will be written.
                         Parent directories will be created if they don't exist.
            indent: Number of spaces for JSON indentation (default: 2).
                    Set to None for compact output.
        """
        self.output_path = Path(output_path)
        self.indent = indent

    def report(self, result: EvalRunResult) -> None:
        """Write the evaluation results to a JSON file.

        Creates parent directories if they don't exist. Overwrites
        any existing file at the output path.

        Args:
            result: The evaluation run result to write.

        Raises:
            OSError: If unable to write to the output path.
        """
        # Create parent directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert result to dict
        data = self._result_to_dict(result)

        # Write JSON file
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False)

    def _result_to_dict(self, result: EvalRunResult) -> dict:
        """Convert EvalRunResult to a JSON-serializable dictionary.

        Args:
            result: The evaluation run result.

        Returns:
            Dictionary with all result data in JSON-compatible format.
        """
        duration_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()

        return {
            "suite_name": result.suite_name,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "duration_seconds": duration_seconds,
            "summary": {
                "total_cases": result.total_cases,
                "passed_cases": result.passed_cases,
                "failed_cases": result.failed_cases,
                "pass_rate": result.pass_rate,
            },
            "case_results": [
                self._case_result_to_dict(case_result)
                for case_result in result.case_results
            ],
        }

    def _case_result_to_dict(self, case_result) -> dict:
        """Convert CaseResult to a JSON-serializable dictionary.

        Args:
            case_result: A CaseResult instance.

        Returns:
            Dictionary with all case result data.
        """
        return {
            "case_id": case_result.case_id,
            "case_name": case_result.case_name,
            "passed": case_result.passed,
            "duration_ms": case_result.duration_ms,
            "trace_id": case_result.trace_id,
            "output": self._serialize_output(case_result.output),
            "error": case_result.error,
            "assertions": [
                self._assertion_result_to_dict(assertion)
                for assertion in case_result.assertion_results
            ],
        }

    def _assertion_result_to_dict(self, assertion_result) -> dict:
        """Convert AssertionResult to a JSON-serializable dictionary.

        Args:
            assertion_result: An AssertionResult instance.

        Returns:
            Dictionary with all assertion result data.
        """
        return {
            "assertion_type": assertion_result.assertion_type,
            "passed": assertion_result.passed,
            "message": assertion_result.message,
            "score": assertion_result.score,
            "expected": self._serialize_output(assertion_result.expected),
            "actual": self._serialize_output(assertion_result.actual),
            "details": assertion_result.details,
        }

    def _serialize_output(self, output) -> any:
        """Serialize output values for JSON.

        Handles common non-JSON-serializable types by converting them
        to strings. For complex objects, returns their string representation.

        Args:
            output: The output value to serialize.

        Returns:
            JSON-serializable version of the output.
        """
        if output is None:
            return None

        # Basic JSON-serializable types
        if isinstance(output, (bool, int, float, str, list, dict)):
            # For lists and dicts, recursively serialize contents
            if isinstance(output, list):
                return [self._serialize_output(item) for item in output]
            elif isinstance(output, dict):
                return {
                    str(key): self._serialize_output(value)
                    for key, value in output.items()
                }
            return output

        # Convert other types to string
        return str(output)
