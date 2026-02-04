"""
n8n-specific assertions for workflow evaluation.

This module provides specialized assertions designed for testing n8n workflows,
including node completion checks, output validation, performance assertions,
and AI-specific validations.
"""

from __future__ import annotations

from typing import Any, Optional

from prela.core.span import Span
from prela.evals.assertions.base import AssertionResult, BaseAssertion


class N8nNodeCompleted(BaseAssertion):
    """Assert that a specific node completed successfully.

    Example:
        >>> assertion = N8nNodeCompleted(node_name="Data Processor")
        >>> result = assertion.evaluate(execution_result, None, None)
        >>> assert result.passed
    """

    def __init__(self, node_name: str):
        """Initialize node completion assertion.

        Args:
            node_name: Name of the node to check
        """
        self.node_name = node_name

    def evaluate(
        self, output: Any, expected: Any | None, trace: list[Span] | None
    ) -> AssertionResult:
        """Evaluate if the node completed successfully.

        Args:
            output: Execution result from n8n
            expected: Not used
            trace: Not used

        Returns:
            AssertionResult with pass/fail status
        """
        execution_result = output if isinstance(output, dict) else {}

        for node in execution_result.get("nodes", []):
            if node.get("name") == self.node_name:
                status = node.get("status")
                passed = status == "success"
                return AssertionResult(
                    passed=passed,
                    assertion_type="n8n_node_completed",
                    message=f"Node '{self.node_name}' {'completed successfully' if passed else f'failed with status: {status}'}",
                    expected="success",
                    actual=status,
                )

        return AssertionResult(
            passed=False,
            assertion_type="n8n_node_completed",
            message=f"Node '{self.node_name}' not found in execution",
            expected="node present",
            actual="node not found",
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> N8nNodeCompleted:
        """Create assertion from configuration dict.

        Args:
            config: Dictionary with 'node_name' key

        Returns:
            N8nNodeCompleted instance
        """
        return cls(node_name=config["node_name"])


class N8nNodeOutput(BaseAssertion):
    """Assert on the output of a specific node using path notation.

    Example:
        >>> assertion = N8nNodeOutput(
        ...     node_name="API Call",
        ...     path="response.status",
        ...     expected_value=200
        ... )
        >>> result = assertion.evaluate(execution_result, None, None)
    """

    def __init__(self, node_name: str, path: str, expected_value: Any):
        """Initialize node output assertion.

        Args:
            node_name: Name of the node to check
            path: Dot-separated path to value (e.g., "response.data.id")
            expected_value: Expected value at the path
        """
        self.node_name = node_name
        self.path = path
        self.expected_value = expected_value

    def evaluate(
        self, output: Any, expected: Any | None, trace: list[Span] | None
    ) -> AssertionResult:
        """Evaluate if node output at path matches expected value.

        Args:
            output: Execution result from n8n
            expected: Not used (expected_value from __init__ is used)
            trace: Not used

        Returns:
            AssertionResult with pass/fail status
        """
        execution_result = output if isinstance(output, dict) else {}

        node_data = self._get_node(execution_result)
        if not node_data:
            return AssertionResult(
                passed=False,
                assertion_type="n8n_node_output",
                message=f"Node '{self.node_name}' not found",
                expected=f"{self.path} = {self.expected_value}",
                actual="node not found",
            )

        actual = self._get_path(node_data.get("output", {}), self.path)
        passed = actual == self.expected_value

        return AssertionResult(
            passed=passed,
            assertion_type="n8n_node_output",
            message=f"Node '{self.node_name}' output at '{self.path}' {'matches' if passed else 'does not match'}",
            expected=self.expected_value,
            actual=actual,
        )

    def _get_node(self, result: dict) -> Optional[dict]:
        """Find node by name in execution result."""
        for node in result.get("nodes", []):
            if node.get("name") == self.node_name:
                return node
        return None

    def _get_path(self, data: Any, path: str) -> Any:
        """Extract value from nested dict using dot notation."""
        parts = path.split(".")
        for part in parts:
            if isinstance(data, dict):
                data = data.get(part)
            else:
                return None
        return data

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> N8nNodeOutput:
        """Create assertion from configuration dict.

        Args:
            config: Dictionary with 'node_name', 'path', and 'expected_value' keys

        Returns:
            N8nNodeOutput instance
        """
        return cls(
            node_name=config["node_name"],
            path=config["path"],
            expected_value=config["expected_value"],
        )


class N8nWorkflowDuration(BaseAssertion):
    """Assert workflow completed within time limit.

    Example:
        >>> assertion = N8nWorkflowDuration(max_seconds=5.0)
        >>> result = assertion.evaluate(execution_result, None, None)
    """

    def __init__(self, max_seconds: float):
        """Initialize workflow duration assertion.

        Args:
            max_seconds: Maximum allowed execution time in seconds
        """
        self.max_seconds = max_seconds
        self.max_ms = max_seconds * 1000

    def evaluate(
        self, output: Any, expected: Any | None, trace: list[Span] | None
    ) -> AssertionResult:
        """Evaluate if workflow duration is within limit.

        Args:
            output: Execution result from n8n
            expected: Not used
            trace: Not used

        Returns:
            AssertionResult with pass/fail status
        """
        execution_result = output if isinstance(output, dict) else {}

        duration_ms = execution_result.get("duration_ms", float("inf"))
        passed = duration_ms <= self.max_ms

        return AssertionResult(
            passed=passed,
            assertion_type="n8n_workflow_duration",
            message=f"Workflow duration: {duration_ms:.1f}ms {'within' if passed else 'exceeds'} limit of {self.max_ms}ms",
            expected=f"<= {self.max_ms}ms",
            actual=f"{duration_ms:.1f}ms",
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> N8nWorkflowDuration:
        """Create assertion from configuration dict.

        Args:
            config: Dictionary with 'max_seconds' key

        Returns:
            N8nWorkflowDuration instance
        """
        return cls(max_seconds=config["max_seconds"])


class N8nAINodeTokens(BaseAssertion):
    """Assert AI node token usage is within budget.

    Example:
        >>> assertion = N8nAINodeTokens(node_name="GPT-4", max_tokens=1000)
        >>> result = assertion.evaluate(execution_result, None, None)
    """

    def __init__(self, node_name: str, max_tokens: int):
        """Initialize AI node token assertion.

        Args:
            node_name: Name of the AI node to check
            max_tokens: Maximum allowed token count
        """
        self.node_name = node_name
        self.max_tokens = max_tokens

    def evaluate(
        self, output: Any, expected: Any | None, trace: list[Span] | None
    ) -> AssertionResult:
        """Evaluate if AI node token usage is within budget.

        Args:
            output: Execution result from n8n
            expected: Not used
            trace: Not used

        Returns:
            AssertionResult with pass/fail status
        """
        execution_result = output if isinstance(output, dict) else {}

        for node in execution_result.get("nodes", []):
            if node.get("name") == self.node_name:
                tokens = node.get("total_tokens", 0)
                passed = tokens <= self.max_tokens

                return AssertionResult(
                    passed=passed,
                    assertion_type="n8n_ai_node_tokens",
                    message=f"Node '{self.node_name}' used {tokens} tokens {'within' if passed else 'exceeds'} budget of {self.max_tokens}",
                    expected=f"<= {self.max_tokens}",
                    actual=str(tokens),
                )

        return AssertionResult(
            passed=False,
            assertion_type="n8n_ai_node_tokens",
            message=f"AI node '{self.node_name}' not found in execution",
            expected=f"node with <= {self.max_tokens} tokens",
            actual="node not found",
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> N8nAINodeTokens:
        """Create assertion from configuration dict.

        Args:
            config: Dictionary with 'node_name' and 'max_tokens' keys

        Returns:
            N8nAINodeTokens instance
        """
        return cls(node_name=config["node_name"], max_tokens=config["max_tokens"])


class N8nWorkflowStatus(BaseAssertion):
    """Assert workflow completed with expected status.

    Example:
        >>> assertion = N8nWorkflowStatus(expected_status="success")
        >>> result = assertion.evaluate(execution_result, None, None)
    """

    def __init__(self, expected_status: str = "success"):
        """Initialize workflow status assertion.

        Args:
            expected_status: Expected workflow status (default: "success")
        """
        self.expected_status = expected_status

    def evaluate(
        self, output: Any, expected: Any | None, trace: list[Span] | None
    ) -> AssertionResult:
        """Evaluate if workflow status matches expected.

        Args:
            output: Execution result from n8n
            expected: Not used (expected_status from __init__ is used)
            trace: Not used

        Returns:
            AssertionResult with pass/fail status
        """
        execution_result = output if isinstance(output, dict) else {}

        actual_status = execution_result.get("status", "unknown")
        passed = actual_status == self.expected_status

        return AssertionResult(
            passed=passed,
            assertion_type="n8n_workflow_status",
            message=f"Workflow status: {actual_status} ({'matches' if passed else 'does not match'} expected: {self.expected_status})",
            expected=self.expected_status,
            actual=actual_status,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> N8nWorkflowStatus:
        """Create assertion from configuration dict.

        Args:
            config: Dictionary with 'expected_status' key

        Returns:
            N8nWorkflowStatus instance
        """
        return cls(expected_status=config.get("expected_status", "success"))


# Convenience factory functions


def node_completed(node_name: str) -> N8nNodeCompleted:
    """Create assertion that node completed successfully.

    Args:
        node_name: Name of the node to check

    Returns:
        N8nNodeCompleted assertion

    Example:
        >>> from prela.evals.n8n.assertions import node_completed
        >>> assertion = node_completed("Data Processor")
    """
    return N8nNodeCompleted(node_name)


def node_output(node_name: str, path: str, expected_value: Any) -> N8nNodeOutput:
    """Create assertion for node output at path.

    Args:
        node_name: Name of the node to check
        path: Dot-separated path to value
        expected_value: Expected value at the path

    Returns:
        N8nNodeOutput assertion

    Example:
        >>> from prela.evals.n8n.assertions import node_output
        >>> assertion = node_output("API Call", "response.status", 200)
    """
    return N8nNodeOutput(node_name, path, expected_value)


def duration_under(seconds: float) -> N8nWorkflowDuration:
    """Create assertion for workflow duration limit.

    Args:
        seconds: Maximum allowed duration in seconds

    Returns:
        N8nWorkflowDuration assertion

    Example:
        >>> from prela.evals.n8n.assertions import duration_under
        >>> assertion = duration_under(5.0)
    """
    return N8nWorkflowDuration(seconds)


def tokens_under(node_name: str, max_tokens: int) -> N8nAINodeTokens:
    """Create assertion for AI node token budget.

    Args:
        node_name: Name of the AI node
        max_tokens: Maximum allowed tokens

    Returns:
        N8nAINodeTokens assertion

    Example:
        >>> from prela.evals.n8n.assertions import tokens_under
        >>> assertion = tokens_under("GPT-4", 1000)
    """
    return N8nAINodeTokens(node_name, max_tokens)


def workflow_completed() -> N8nWorkflowStatus:
    """Create assertion that workflow completed successfully.

    Returns:
        N8nWorkflowStatus assertion with expected_status="success"

    Example:
        >>> from prela.evals.n8n.assertions import workflow_completed
        >>> assertion = workflow_completed()
    """
    return N8nWorkflowStatus(expected_status="success")


def workflow_status(expected_status: str) -> N8nWorkflowStatus:
    """Create assertion for specific workflow status.

    Args:
        expected_status: Expected workflow status

    Returns:
        N8nWorkflowStatus assertion

    Example:
        >>> from prela.evals.n8n.assertions import workflow_status
        >>> assertion = workflow_status("error")
    """
    return N8nWorkflowStatus(expected_status=expected_status)
