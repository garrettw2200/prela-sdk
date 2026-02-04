"""
Tool-related assertions for verifying agent tool usage.
"""

from __future__ import annotations

from typing import Any

from prela.core.span import Span, SpanType
from prela.evals.assertions.base import AssertionResult, BaseAssertion


class ToolCalledAssertion(BaseAssertion):
    """Assert that a specific tool was called during execution.

    This assertion examines the trace to verify that a tool span with the
    specified name exists.

    Example:
        >>> assertion = ToolCalledAssertion(tool_name="web_search")
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, tool_name: str):
        """Initialize tool called assertion.

        Args:
            tool_name: Name of the tool that should have been called
        """
        self.tool_name = tool_name

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if the specified tool was called in the trace."""
        if trace is None or len(trace) == 0:
            return AssertionResult(
                passed=False,
                assertion_type="tool_called",
                message=f"No trace available to check for tool '{self.tool_name}'",
                expected=f"tool '{self.tool_name}' called",
                actual="no trace",
                details={},
            )

        # Look for tool spans with matching name
        tool_spans = [
            span for span in trace
            if span.span_type == SpanType.TOOL and span.name == self.tool_name
        ]

        passed = len(tool_spans) > 0

        if passed:
            message = f"Tool '{self.tool_name}' was called {len(tool_spans)} time(s)"
            details = {
                "call_count": len(tool_spans),
                "span_ids": [span.span_id for span in tool_spans],
            }
        else:
            # List available tools to help debugging
            available_tools = {
                span.name for span in trace
                if span.span_type == SpanType.TOOL
            }
            message = f"Tool '{self.tool_name}' was not called"
            details = {
                "call_count": 0,
                "available_tools": list(available_tools),
            }

        return AssertionResult(
            passed=passed,
            assertion_type="tool_called",
            message=message,
            expected=f"tool '{self.tool_name}' called",
            actual=f"{len(tool_spans)} calls" if passed else "not called",
            details=details,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ToolCalledAssertion:
        """Create from configuration.

        Config format:
            {
                "tool_name": "web_search"
            }
        """
        if "tool_name" not in config:
            raise ValueError("ToolCalledAssertion requires 'tool_name' in config")

        return cls(tool_name=config["tool_name"])

    def __repr__(self) -> str:
        return f"ToolCalledAssertion(tool_name={self.tool_name!r})"


class ToolArgsAssertion(BaseAssertion):
    """Assert that a tool was called with expected arguments.

    This assertion verifies both that the tool was called and that it was
    called with specific argument values.

    Example:
        >>> assertion = ToolArgsAssertion(
        ...     tool_name="web_search",
        ...     expected_args={"query": "Python tutorial"}
        ... )
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(
        self,
        tool_name: str,
        expected_args: dict[str, Any],
        partial_match: bool = True,
    ):
        """Initialize tool args assertion.

        Args:
            tool_name: Name of the tool to check
            expected_args: Expected argument key-value pairs
            partial_match: If True, only check that expected_args are present
                          (allow additional args). If False, require exact match.
        """
        self.tool_name = tool_name
        self.expected_args = expected_args
        self.partial_match = partial_match

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if tool was called with expected arguments."""
        if trace is None or len(trace) == 0:
            return AssertionResult(
                passed=False,
                assertion_type="tool_args",
                message=f"No trace available to check tool '{self.tool_name}' arguments",
                expected=f"tool '{self.tool_name}' with args {self.expected_args}",
                actual="no trace",
                details={},
            )

        # Find tool spans with matching name
        tool_spans = [
            span for span in trace
            if span.span_type == SpanType.TOOL and span.name == self.tool_name
        ]

        if not tool_spans:
            return AssertionResult(
                passed=False,
                assertion_type="tool_args",
                message=f"Tool '{self.tool_name}' was not called",
                expected=f"tool '{self.tool_name}' with args {self.expected_args}",
                actual="tool not called",
                details={},
            )

        # Check each tool call for matching arguments
        matches = []
        for span in tool_spans:
            # Look for tool input in attributes (common patterns)
            actual_args = {}
            for key, value in span.attributes.items():
                if key.startswith("tool.input.") or key.startswith("input."):
                    arg_name = key.split(".")[-1]
                    actual_args[arg_name] = value

            # Also check for generic "input" attribute
            if "input" in span.attributes and isinstance(span.attributes["input"], dict):
                actual_args.update(span.attributes["input"])

            # Check if this call matches expected args
            if self.partial_match:
                # Check that all expected args are present with correct values
                match = all(
                    actual_args.get(key) == value
                    for key, value in self.expected_args.items()
                )
            else:
                # Require exact match
                match = actual_args == self.expected_args

            if match:
                matches.append((span, actual_args))

        passed = len(matches) > 0

        if passed:
            message = f"Tool '{self.tool_name}' was called with expected arguments ({len(matches)} time(s))"
            details = {
                "match_count": len(matches),
                "span_ids": [span.span_id for span, _ in matches],
                "matched_args": [args for _, args in matches],
            }
        else:
            # Show actual args from first call for debugging
            first_span_args = {}
            if tool_spans:
                for key, value in tool_spans[0].attributes.items():
                    if key.startswith("tool.input.") or key.startswith("input."):
                        arg_name = key.split(".")[-1]
                        first_span_args[arg_name] = value

            message = f"Tool '{self.tool_name}' was called but not with expected arguments"
            details = {
                "match_count": 0,
                "call_count": len(tool_spans),
                "first_call_args": first_span_args,
            }

        return AssertionResult(
            passed=passed,
            assertion_type="tool_args",
            message=message,
            expected=self.expected_args,
            actual=matches[0][1] if matches else first_span_args,
            details=details,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ToolArgsAssertion:
        """Create from configuration.

        Config format:
            {
                "tool_name": "web_search",
                "expected_args": {"query": "Python"},
                "partial_match": true  # optional, default: true
            }
        """
        if "tool_name" not in config:
            raise ValueError("ToolArgsAssertion requires 'tool_name' in config")
        if "expected_args" not in config:
            raise ValueError("ToolArgsAssertion requires 'expected_args' in config")

        return cls(
            tool_name=config["tool_name"],
            expected_args=config["expected_args"],
            partial_match=config.get("partial_match", True),
        )

    def __repr__(self) -> str:
        return (
            f"ToolArgsAssertion(tool_name={self.tool_name!r}, "
            f"expected_args={self.expected_args}, "
            f"partial_match={self.partial_match})"
        )


class ToolSequenceAssertion(BaseAssertion):
    """Assert that tools were called in a specific order.

    This assertion verifies that tools appear in the trace in the expected
    sequence, though other tools may appear between them.

    Example:
        >>> assertion = ToolSequenceAssertion(
        ...     sequence=["web_search", "calculator", "summarize"]
        ... )
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, sequence: list[str], strict: bool = False):
        """Initialize tool sequence assertion.

        Args:
            sequence: Expected sequence of tool names
            strict: If True, no other tools can appear between expected ones.
                   If False, other tools are allowed between expected sequence.
        """
        if not sequence:
            raise ValueError("sequence cannot be empty")

        self.sequence = sequence
        self.strict = strict

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if tools were called in the expected sequence."""
        if trace is None or len(trace) == 0:
            return AssertionResult(
                passed=False,
                assertion_type="tool_sequence",
                message="No trace available to check tool sequence",
                expected=f"sequence: {self.sequence}",
                actual="no trace",
                details={},
            )

        # Extract tool call sequence from trace (ordered by started_at)
        tool_spans = [
            span for span in sorted(trace, key=lambda s: s.started_at)
            if span.span_type == SpanType.TOOL
        ]

        if not tool_spans:
            return AssertionResult(
                passed=False,
                assertion_type="tool_sequence",
                message="No tool calls found in trace",
                expected=f"sequence: {self.sequence}",
                actual="no tools called",
                details={},
            )

        actual_sequence = [span.name for span in tool_spans]

        # Check sequence
        if self.strict:
            # Strict mode: must match exactly
            passed = actual_sequence == self.sequence
            if passed:
                message = f"Tool sequence matches exactly: {self.sequence}"
            else:
                message = f"Tool sequence does not match. Expected {self.sequence}, got {actual_sequence}"
        else:
            # Non-strict: check subsequence
            seq_idx = 0
            for tool_name in actual_sequence:
                if seq_idx < len(self.sequence) and tool_name == self.sequence[seq_idx]:
                    seq_idx += 1

            passed = seq_idx == len(self.sequence)

            if passed:
                message = f"Tool sequence {self.sequence} found in correct order"
            else:
                found = self.sequence[:seq_idx]
                missing = self.sequence[seq_idx:]
                message = f"Tool sequence incomplete. Found {found}, missing {missing}"

        return AssertionResult(
            passed=passed,
            assertion_type="tool_sequence",
            message=message,
            expected=self.sequence,
            actual=actual_sequence,
            details={
                "strict": self.strict,
                "expected_length": len(self.sequence),
                "actual_length": len(actual_sequence),
            },
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ToolSequenceAssertion:
        """Create from configuration.

        Config format:
            {
                "sequence": ["tool1", "tool2", "tool3"],
                "strict": false  # optional, default: false
            }
        """
        if "sequence" not in config:
            raise ValueError("ToolSequenceAssertion requires 'sequence' in config")

        return cls(
            sequence=config["sequence"],
            strict=config.get("strict", False),
        )

    def __repr__(self) -> str:
        return f"ToolSequenceAssertion(sequence={self.sequence}, strict={self.strict})"
