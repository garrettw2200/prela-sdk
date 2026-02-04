"""JUnit XML reporter for evaluation results.

This module provides a reporter that generates JUnit-compatible XML files,
enabling integration with CI/CD systems like Jenkins, GitLab CI, GitHub Actions,
and other tools that parse JUnit test reports.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from prela.evals.runner import EvalRunResult


class JUnitReporter:
    """Reporter that generates JUnit XML format for CI/CD integration.

    Creates a JUnit XML file that can be consumed by continuous integration
    systems for test result visualization, trend analysis, and failure reporting.

    The XML format follows the JUnit schema with:
    - <testsuite> root element with summary statistics
    - <testcase> elements for each test case
    - <failure> elements for failed assertions
    - <error> elements for execution errors
    - <system-out> for additional output/trace information

    Supported CI/CD platforms:
    - Jenkins (JUnit plugin)
    - GitLab CI/CD (junit report artifacts)
    - GitHub Actions (test reporters)
    - Azure DevOps (publish test results)
    - CircleCI (store_test_results)

    Example:
        >>> from prela.evals import EvalRunner
        >>> from prela.evals.reporters import JUnitReporter
        >>>
        >>> runner = EvalRunner(suite, agent)
        >>> result = runner.run()
        >>>
        >>> reporter = JUnitReporter("test-results/junit.xml")
        >>> reporter.report(result)
        # Creates JUnit XML at test-results/junit.xml
    """

    def __init__(self, output_path: str | Path):
        """Initialize the JUnit XML reporter.

        Args:
            output_path: Path where the JUnit XML file will be written.
                         Parent directories will be created if they don't exist.
        """
        self.output_path = Path(output_path)

    def report(self, result: EvalRunResult) -> None:
        """Generate and write JUnit XML for the evaluation results.

        Creates parent directories if they don't exist. Overwrites
        any existing file at the output path.

        Args:
            result: The evaluation run result to convert to JUnit XML.

        Raises:
            OSError: If unable to write to the output path.
        """
        # Create parent directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build XML structure
        xml_root = self._build_xml(result)

        # Write to file with pretty formatting
        self._write_xml(xml_root)

    def _build_xml(self, result: EvalRunResult) -> ET.Element:
        """Build JUnit XML element tree from evaluation result.

        Args:
            result: The evaluation run result.

        Returns:
            XML root element (<testsuite>).
        """
        # Calculate duration in seconds
        duration_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()

        # Create root testsuite element
        testsuite = ET.Element(
            "testsuite",
            attrib={
                "name": result.suite_name,
                "tests": str(result.total_cases),
                "failures": str(result.failed_cases),
                "errors": "0",  # We track errors as failures
                "skipped": "0",
                "time": f"{duration_seconds:.3f}",
                "timestamp": result.started_at.isoformat(),
            },
        )

        # Add testcase elements
        for case_result in result.case_results:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                attrib={
                    "name": case_result.case_name,
                    "classname": result.suite_name,
                    "time": f"{case_result.duration_ms / 1000:.3f}",
                },
            )

            # If case failed due to execution error, add <error> element
            if case_result.error:
                error = ET.SubElement(
                    testcase,
                    "error",
                    attrib={
                        "type": "ExecutionError",
                        "message": self._truncate_message(case_result.error),
                    },
                )
                error.text = case_result.error

            # If case failed assertions, add <failure> elements
            elif not case_result.passed:
                # Collect all failed assertions
                failed_assertions = [
                    a for a in case_result.assertion_results if not a.passed
                ]

                if failed_assertions:
                    # Create a single failure element with all failed assertions
                    failure_message = self._format_failure_message(
                        failed_assertions
                    )
                    failure = ET.SubElement(
                        testcase,
                        "failure",
                        attrib={
                            "type": "AssertionFailure",
                            "message": self._truncate_message(failure_message),
                        },
                    )
                    failure.text = self._format_failure_details(
                        failed_assertions
                    )

            # Add system-out with trace_id and output if available
            system_out_parts = []
            if case_result.trace_id:
                system_out_parts.append(f"Trace ID: {case_result.trace_id}")
            if case_result.output is not None:
                output_str = str(case_result.output)
                if len(output_str) > 1000:
                    output_str = output_str[:1000] + "... (truncated)"
                system_out_parts.append(f"Output: {output_str}")

            if system_out_parts:
                system_out = ET.SubElement(testcase, "system-out")
                system_out.text = "\n".join(system_out_parts)

        return testsuite

    def _write_xml(self, root: ET.Element) -> None:
        """Write XML element tree to file with pretty formatting.

        Args:
            root: The root XML element to write.
        """
        # Pretty-print the XML
        self._indent(root)

        # Create ElementTree and write to file
        tree = ET.ElementTree(root)
        tree.write(
            self.output_path,
            encoding="utf-8",
            xml_declaration=True,
            method="xml",
        )

    def _indent(self, elem: ET.Element, level: int = 0) -> None:
        """Add indentation to XML elements for pretty printing.

        Modifies the element tree in-place to add newlines and indentation.

        Args:
            elem: The XML element to indent.
            level: Current indentation level (number of tabs).
        """
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    def _format_failure_message(self, failed_assertions: list) -> str:
        """Format a summary message for failed assertions.

        Args:
            failed_assertions: List of AssertionResult objects that failed.

        Returns:
            Summary string listing all failed assertion types.
        """
        if not failed_assertions:
            return "Test case failed"

        if len(failed_assertions) == 1:
            return failed_assertions[0].message

        # Multiple failures
        assertion_types = [a.assertion_type for a in failed_assertions]
        return f"{len(failed_assertions)} assertions failed: {', '.join(assertion_types)}"

    def _format_failure_details(self, failed_assertions: list) -> str:
        """Format detailed failure information for all failed assertions.

        Args:
            failed_assertions: List of AssertionResult objects that failed.

        Returns:
            Detailed multi-line string with all failure information.
        """
        lines = []
        for i, assertion in enumerate(failed_assertions, 1):
            lines.append(f"Assertion {i}: {assertion.assertion_type}")
            lines.append(f"  Message: {assertion.message}")

            if assertion.expected is not None:
                expected_str = str(assertion.expected)
                if len(expected_str) > 200:
                    expected_str = expected_str[:200] + "... (truncated)"
                lines.append(f"  Expected: {expected_str}")

            if assertion.actual is not None:
                actual_str = str(assertion.actual)
                if len(actual_str) > 200:
                    actual_str = actual_str[:200] + "... (truncated)"
                lines.append(f"  Actual: {actual_str}")

            if assertion.score is not None:
                lines.append(f"  Score: {assertion.score:.3f}")

            if assertion.details:
                lines.append(f"  Details: {assertion.details}")

            lines.append("")  # Blank line between assertions

        return "\n".join(lines)

    def _truncate_message(self, message: str, max_length: int = 200) -> str:
        """Truncate long error messages for the message attribute.

        Args:
            message: The message to truncate.
            max_length: Maximum length before truncation.

        Returns:
            Truncated string with "..." suffix if needed.
        """
        if len(message) > max_length:
            return message[: max_length - 3] + "..."
        return message
