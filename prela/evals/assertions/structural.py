"""
Structural assertions for text and data format validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from prela.core.span import Span
from prela.evals.assertions.base import AssertionResult, BaseAssertion


class ContainsAssertion(BaseAssertion):
    """Assert that output contains specified text.

    Example:
        >>> assertion = ContainsAssertion(text="error", case_sensitive=False)
        >>> result = assertion.evaluate(output="Error occurred", expected=None, trace=None)
        >>> assert result.passed
    """

    def __init__(self, text: str, case_sensitive: bool = True):
        """Initialize contains assertion.

        Args:
            text: Text that must be present in output
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.text = text
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if output contains the specified text."""
        output_str = str(output)
        text = self.text

        if not self.case_sensitive:
            output_str = output_str.lower()
            text = text.lower()

        passed = text in output_str

        if passed:
            message = f"Output contains '{self.text}'"
        else:
            message = f"Output does not contain '{self.text}'"

        return AssertionResult(
            passed=passed,
            assertion_type="contains",
            message=message,
            expected=self.text,
            actual=output_str[:100] + "..." if len(output_str) > 100 else output_str,
            details={"case_sensitive": self.case_sensitive},
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ContainsAssertion:
        """Create from configuration.

        Config format:
            {
                "text": "required text",
                "case_sensitive": true  # optional, default: true
            }
        """
        if "text" not in config:
            raise ValueError("ContainsAssertion requires 'text' in config")

        return cls(
            text=config["text"],
            case_sensitive=config.get("case_sensitive", True),
        )

    def __repr__(self) -> str:
        return f"ContainsAssertion(text={self.text!r}, case_sensitive={self.case_sensitive})"


class NotContainsAssertion(BaseAssertion):
    """Assert that output does NOT contain specified text.

    Example:
        >>> assertion = NotContainsAssertion(text="error")
        >>> result = assertion.evaluate(output="Success!", expected=None, trace=None)
        >>> assert result.passed
    """

    def __init__(self, text: str, case_sensitive: bool = True):
        """Initialize not-contains assertion.

        Args:
            text: Text that must NOT be present in output
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.text = text
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if output does not contain the specified text."""
        output_str = str(output)
        text = self.text

        if not self.case_sensitive:
            output_str = output_str.lower()
            text = text.lower()

        passed = text not in output_str

        if passed:
            message = f"Output correctly does not contain '{self.text}'"
        else:
            message = f"Output incorrectly contains '{self.text}'"

        return AssertionResult(
            passed=passed,
            assertion_type="not_contains",
            message=message,
            expected=f"not containing '{self.text}'",
            actual=output_str[:100] + "..." if len(output_str) > 100 else output_str,
            details={"case_sensitive": self.case_sensitive},
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NotContainsAssertion:
        """Create from configuration.

        Config format:
            {
                "text": "forbidden text",
                "case_sensitive": true  # optional, default: true
            }
        """
        if "text" not in config:
            raise ValueError("NotContainsAssertion requires 'text' in config")

        return cls(
            text=config["text"],
            case_sensitive=config.get("case_sensitive", True),
        )

    def __repr__(self) -> str:
        return f"NotContainsAssertion(text={self.text!r}, case_sensitive={self.case_sensitive})"


class RegexAssertion(BaseAssertion):
    """Assert that output matches a regular expression pattern.

    Example:
        >>> assertion = RegexAssertion(pattern=r"\\d{3}-\\d{4}")
        >>> result = assertion.evaluate(output="Call 555-1234", expected=None, trace=None)
        >>> assert result.passed
    """

    def __init__(self, pattern: str, flags: int = 0):
        """Initialize regex assertion.

        Args:
            pattern: Regular expression pattern to match
            flags: Optional regex flags (e.g., re.IGNORECASE)
        """
        self.pattern = pattern
        self.flags = flags
        self._compiled = re.compile(pattern, flags)

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if output matches the regex pattern."""
        output_str = str(output)
        match = self._compiled.search(output_str)
        passed = match is not None

        if passed:
            matched_text = match.group(0) if match else ""
            message = f"Output matches pattern '{self.pattern}' (matched: '{matched_text}')"
            details = {
                "matched_text": matched_text,
                "match_start": match.start(),
                "match_end": match.end(),
            }
        else:
            message = f"Output does not match pattern '{self.pattern}'"
            details = {}

        return AssertionResult(
            passed=passed,
            assertion_type="regex",
            message=message,
            expected=self.pattern,
            actual=output_str[:100] + "..." if len(output_str) > 100 else output_str,
            details=details,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RegexAssertion:
        """Create from configuration.

        Config format:
            {
                "pattern": "\\d{3}-\\d{4}",
                "flags": 2  # optional, e.g., re.IGNORECASE
            }
        """
        if "pattern" not in config:
            raise ValueError("RegexAssertion requires 'pattern' in config")

        return cls(
            pattern=config["pattern"],
            flags=config.get("flags", 0),
        )

    def __repr__(self) -> str:
        return f"RegexAssertion(pattern={self.pattern!r}, flags={self.flags})"


class LengthAssertion(BaseAssertion):
    """Assert that output length is within specified bounds.

    Example:
        >>> assertion = LengthAssertion(min_length=10, max_length=100)
        >>> result = assertion.evaluate(output="Hello, world!", expected=None, trace=None)
        >>> assert result.passed
    """

    def __init__(self, min_length: int | None = None, max_length: int | None = None):
        """Initialize length assertion.

        Args:
            min_length: Minimum acceptable length (inclusive)
            max_length: Maximum acceptable length (inclusive)

        Raises:
            ValueError: If both min_length and max_length are None
        """
        if min_length is None and max_length is None:
            raise ValueError("At least one of min_length or max_length must be specified")

        if min_length is not None and min_length < 0:
            raise ValueError("min_length must be non-negative")

        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be non-negative")

        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError("min_length cannot be greater than max_length")

        self.min_length = min_length
        self.max_length = max_length

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if output length is within bounds."""
        output_str = str(output)
        actual_length = len(output_str)

        passed = True
        reasons = []

        if self.min_length is not None and actual_length < self.min_length:
            passed = False
            reasons.append(f"too short (< {self.min_length})")

        if self.max_length is not None and actual_length > self.max_length:
            passed = False
            reasons.append(f"too long (> {self.max_length})")

        if passed:
            if self.min_length is not None and self.max_length is not None:
                message = f"Output length {actual_length} is within bounds [{self.min_length}, {self.max_length}]"
            elif self.min_length is not None:
                message = f"Output length {actual_length} is >= {self.min_length}"
            else:
                message = f"Output length {actual_length} is <= {self.max_length}"
        else:
            message = f"Output length {actual_length} is {', '.join(reasons)}"

        expected_desc = []
        if self.min_length is not None:
            expected_desc.append(f"min: {self.min_length}")
        if self.max_length is not None:
            expected_desc.append(f"max: {self.max_length}")

        return AssertionResult(
            passed=passed,
            assertion_type="length",
            message=message,
            expected=", ".join(expected_desc),
            actual=actual_length,
            details={
                "min_length": self.min_length,
                "max_length": self.max_length,
            },
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LengthAssertion:
        """Create from configuration.

        Config format:
            {
                "min_length": 10,  # optional
                "max_length": 100  # optional
            }
        """
        return cls(
            min_length=config.get("min_length"),
            max_length=config.get("max_length"),
        )

    def __repr__(self) -> str:
        return f"LengthAssertion(min_length={self.min_length}, max_length={self.max_length})"


class JSONValidAssertion(BaseAssertion):
    """Assert that output is valid JSON, optionally matching a schema.

    Example:
        >>> assertion = JSONValidAssertion()
        >>> result = assertion.evaluate(output='{"key": "value"}', expected=None, trace=None)
        >>> assert result.passed
    """

    def __init__(self, schema: dict[str, Any] | None = None):
        """Initialize JSON validation assertion.

        Args:
            schema: Optional JSON schema to validate against (using jsonschema library)
        """
        self.schema = schema

        # Only import jsonschema if schema validation is requested
        if schema is not None:
            try:
                import jsonschema
                self._validator = jsonschema.Draft7Validator(schema)
            except ImportError:
                raise ImportError(
                    "jsonschema library required for schema validation. "
                    "Install with: pip install jsonschema"
                )
        else:
            self._validator = None

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if output is valid JSON and optionally matches schema."""
        output_str = str(output)

        # First, check if it's valid JSON
        try:
            parsed = json.loads(output_str)
        except json.JSONDecodeError as e:
            return AssertionResult(
                passed=False,
                assertion_type="json_valid",
                message=f"Output is not valid JSON: {e.msg}",
                expected="valid JSON",
                actual=output_str[:100] + "..." if len(output_str) > 100 else output_str,
                details={"error": str(e), "position": e.pos},
            )

        # If no schema, we're done
        if self._validator is None:
            return AssertionResult(
                passed=True,
                assertion_type="json_valid",
                message="Output is valid JSON",
                expected="valid JSON",
                actual=parsed,
                details={"type": type(parsed).__name__},
            )

        # Validate against schema
        errors = list(self._validator.iter_errors(parsed))

        if not errors:
            return AssertionResult(
                passed=True,
                assertion_type="json_valid",
                message="Output is valid JSON and matches schema",
                expected="valid JSON matching schema",
                actual=parsed,
                details={"schema_valid": True},
            )
        else:
            error_messages = [f"{e.json_path}: {e.message}" for e in errors[:3]]
            if len(errors) > 3:
                error_messages.append(f"... and {len(errors) - 3} more errors")

            return AssertionResult(
                passed=False,
                assertion_type="json_valid",
                message=f"Output is valid JSON but does not match schema: {'; '.join(error_messages)}",
                expected="valid JSON matching schema",
                actual=parsed,
                details={
                    "schema_valid": False,
                    "error_count": len(errors),
                    "errors": error_messages,
                },
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JSONValidAssertion:
        """Create from configuration.

        Config format:
            {
                "schema": {  # optional
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        """
        return cls(schema=config.get("schema"))

    def __repr__(self) -> str:
        return f"JSONValidAssertion(schema={self.schema is not None})"
