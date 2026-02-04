"""
Base classes for evaluation assertions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from prela.core.span import Span


@dataclass
class AssertionResult:
    """Result of an assertion evaluation.

    Attributes:
        passed: Whether the assertion passed
        assertion_type: Type of assertion (e.g., "contains", "semantic_similarity")
        message: Human-readable message describing the result
        score: Optional score between 0-1 for partial credit assertions
        expected: Expected value (if applicable)
        actual: Actual value that was evaluated
        details: Additional details about the evaluation
    """

    passed: bool
    assertion_type: str
    message: str
    score: float | None = None
    expected: Any = None
    actual: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f"{status} [{self.assertion_type}] {self.message}"
        if self.score is not None:
            msg += f" (score: {self.score:.2f})"
        return msg


class BaseAssertion(ABC):
    """Base class for all assertions.

    Assertions evaluate agent outputs and traces to determine if they meet
    expected criteria. Subclasses should implement the evaluate() method
    to perform the actual check.
    """

    @abstractmethod
    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Evaluate the assertion against the output and trace.

        Args:
            output: The actual output from the agent/function under test
            expected: The expected output (format depends on assertion type)
            trace: Optional list of spans from the traced execution

        Returns:
            AssertionResult with pass/fail status and details
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> BaseAssertion:
        """Create an assertion instance from configuration dict.

        Args:
            config: Configuration dictionary with assertion-specific parameters

        Returns:
            Configured assertion instance

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"{self.__class__.__name__}()"
