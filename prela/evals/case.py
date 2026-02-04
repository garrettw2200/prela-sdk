"""Eval case data structures for defining test cases.

This module provides the core data structures for defining evaluation cases:
- EvalInput: What goes into the agent
- EvalExpected: What we compare against
- EvalCase: Complete test case with input, expected output, and assertions

Example:
    >>> from prela.evals import EvalCase, EvalInput, EvalExpected
    >>> case = EvalCase(
    ...     id="test_qa",
    ...     name="Basic QA test",
    ...     input=EvalInput(query="What is 2+2?"),
    ...     expected=EvalExpected(contains=["4"])
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalInput:
    """Input data for an eval case.

    Represents what goes into the agent being tested. Can be a simple query,
    a list of messages, or custom context data.

    Attributes:
        query: Simple string query/prompt (for basic use cases)
        messages: List of message dicts (for chat-based agents)
        context: Additional context data (e.g., retrieved documents, metadata)

    Example:
        >>> # Simple query
        >>> input1 = EvalInput(query="What is the capital of France?")
        >>>
        >>> # Chat messages
        >>> input2 = EvalInput(messages=[
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>>
        >>> # Query with context
        >>> input3 = EvalInput(
        ...     query="Summarize the document",
        ...     context={"document": "Long text here..."}
        ... )
    """

    query: str | None = None
    messages: list[dict] | None = None
    context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate that at least one input type is provided."""
        if self.query is None and self.messages is None:
            raise ValueError("EvalInput must have either 'query' or 'messages'")

    def to_agent_input(self) -> dict[str, Any]:
        """Convert to format that agent expects.

        Returns:
            Dictionary with all non-None input fields.

        Example:
            >>> input = EvalInput(query="Hello", context={"user_id": "123"})
            >>> input.to_agent_input()
            {'query': 'Hello', 'context': {'user_id': '123'}}
        """
        result: dict[str, Any] = {}

        if self.query is not None:
            result["query"] = self.query

        if self.messages is not None:
            result["messages"] = self.messages

        if self.context is not None:
            result["context"] = self.context

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalInput:
        """Create EvalInput from dictionary.

        Args:
            data: Dictionary with 'query', 'messages', and/or 'context' keys

        Returns:
            EvalInput instance

        Example:
            >>> data = {"query": "Hello", "context": {"key": "value"}}
            >>> input = EvalInput.from_dict(data)
        """
        return cls(
            query=data.get("query"),
            messages=data.get("messages"),
            context=data.get("context"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the input.
        """
        result: dict[str, Any] = {}

        if self.query is not None:
            result["query"] = self.query

        if self.messages is not None:
            result["messages"] = self.messages

        if self.context is not None:
            result["context"] = self.context

        return result


@dataclass
class EvalExpected:
    """Expected output for an eval case.

    Defines what the agent's output should look like. Supports multiple
    validation strategies:
    - Exact output match
    - Contains/not_contains substring checks
    - Tool call validation
    - Custom metadata checks

    Attributes:
        output: Exact expected output string
        contains: List of substrings that must appear in output
        not_contains: List of substrings that must NOT appear in output
        tool_calls: Expected tool calls (list of dicts with 'name', 'args', etc.)
        metadata: Expected metadata fields (e.g., final_answer, confidence)

    Example:
        >>> # Exact match
        >>> expected1 = EvalExpected(output="The answer is 42")
        >>>
        >>> # Substring checks
        >>> expected2 = EvalExpected(
        ...     contains=["Paris", "capital"],
        ...     not_contains=["London", "Berlin"]
        ... )
        >>>
        >>> # Tool call validation
        >>> expected3 = EvalExpected(tool_calls=[
        ...     {"name": "search", "args": {"query": "weather"}}
        ... ])
    """

    output: str | None = None
    contains: list[str] | None = None
    not_contains: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate that at least one expectation is provided."""
        if (
            self.output is None
            and self.contains is None
            and self.not_contains is None
            and self.tool_calls is None
            and self.metadata is None
        ):
            raise ValueError(
                "EvalExpected must have at least one expectation "
                "(output, contains, not_contains, tool_calls, or metadata)"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalExpected:
        """Create EvalExpected from dictionary.

        Args:
            data: Dictionary with expected output specifications

        Returns:
            EvalExpected instance

        Example:
            >>> data = {"contains": ["Paris"], "not_contains": ["London"]}
            >>> expected = EvalExpected.from_dict(data)
        """
        return cls(
            output=data.get("output"),
            contains=data.get("contains"),
            not_contains=data.get("not_contains"),
            tool_calls=data.get("tool_calls"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the expected output.
        """
        result: dict[str, Any] = {}

        if self.output is not None:
            result["output"] = self.output

        if self.contains is not None:
            result["contains"] = self.contains

        if self.not_contains is not None:
            result["not_contains"] = self.not_contains

        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls

        if self.metadata is not None:
            result["metadata"] = self.metadata

        return result


@dataclass
class EvalCase:
    """Complete evaluation test case.

    Represents a single test case with input, expected output, and assertions.
    Eval cases are the building blocks of eval suites.

    Attributes:
        id: Unique identifier for this test case
        name: Human-readable test case name
        input: Input data for the agent
        expected: Expected output (optional, can use assertions instead)
        assertions: List of assertion configurations (dicts with 'type', 'value', etc.)
        tags: Tags for filtering/grouping test cases
        timeout_seconds: Maximum execution time for this test case
        metadata: Additional metadata for this test case

    Example:
        >>> case = EvalCase(
        ...     id="test_basic_qa",
        ...     name="Basic factual question",
        ...     input=EvalInput(query="What is the capital of France?"),
        ...     expected=EvalExpected(contains=["Paris"]),
        ...     assertions=[
        ...         {"type": "contains", "value": "Paris"},
        ...         {"type": "semantic_similarity", "threshold": 0.8}
        ...     ],
        ...     tags=["qa", "geography"],
        ...     timeout_seconds=10.0
        ... )
    """

    id: str
    name: str
    input: EvalInput
    expected: EvalExpected | None = None
    assertions: list[dict[str, Any]] | None = None
    tags: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate test case configuration."""
        if not self.id:
            raise ValueError("EvalCase must have a non-empty 'id'")

        if not self.name:
            raise ValueError("EvalCase must have a non-empty 'name'")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        # Validate that we have at least expected or assertions
        if self.expected is None and (self.assertions is None or len(self.assertions) == 0):
            raise ValueError("EvalCase must have either 'expected' or 'assertions'")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        """Create EvalCase from dictionary.

        Args:
            data: Dictionary with test case specification

        Returns:
            EvalCase instance

        Example:
            >>> data = {
            ...     "id": "test_1",
            ...     "name": "Test case 1",
            ...     "input": {"query": "Hello"},
            ...     "expected": {"contains": ["Hi"]},
            ...     "tags": ["greeting"]
            ... }
            >>> case = EvalCase.from_dict(data)
        """
        # Parse input
        input_data = data.get("input")
        if input_data is None:
            raise ValueError("EvalCase must have 'input' field")

        if isinstance(input_data, EvalInput):
            input_obj = input_data
        else:
            input_obj = EvalInput.from_dict(input_data)

        # Parse expected (optional)
        expected_data = data.get("expected")
        expected_obj: EvalExpected | None = None
        if expected_data is not None:
            if isinstance(expected_data, EvalExpected):
                expected_obj = expected_data
            else:
                expected_obj = EvalExpected.from_dict(expected_data)

        return cls(
            id=data["id"],
            name=data["name"],
            input=input_obj,
            expected=expected_obj,
            assertions=data.get("assertions"),
            tags=data.get("tags", []),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the test case.

        Example:
            >>> case = EvalCase(
            ...     id="test_1",
            ...     name="Test",
            ...     input=EvalInput(query="Hello"),
            ...     expected=EvalExpected(contains=["Hi"])
            ... )
            >>> data = case.to_dict()
            >>> data["id"]
            'test_1'
        """
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "input": self.input.to_dict(),
            "timeout_seconds": self.timeout_seconds,
        }

        if self.expected is not None:
            result["expected"] = self.expected.to_dict()

        if self.assertions is not None and len(self.assertions) > 0:
            result["assertions"] = self.assertions

        if len(self.tags) > 0:
            result["tags"] = self.tags

        if len(self.metadata) > 0:
            result["metadata"] = self.metadata

        return result
