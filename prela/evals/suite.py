"""Eval suite for organizing and managing test cases.

This module provides the EvalSuite class for organizing multiple eval cases,
with support for YAML serialization, setup/teardown hooks, and default assertions.

Example:
    >>> from prela.evals import EvalSuite, EvalCase, EvalInput, EvalExpected
    >>> suite = EvalSuite(
    ...     name="RAG Quality Suite",
    ...     description="Tests for RAG pipeline quality",
    ...     cases=[
    ...         EvalCase(
    ...             id="test_basic_qa",
    ...             name="Basic QA test",
    ...             input=EvalInput(query="What is 2+2?"),
    ...             expected=EvalExpected(contains=["4"])
    ...         )
    ...     ]
    ... )
    >>> suite.to_yaml("suite.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from prela.evals.case import EvalCase


@dataclass
class EvalSuite:
    """Collection of eval cases with shared configuration.

    An eval suite organizes multiple test cases with:
    - Shared setup/teardown hooks
    - Default assertions applied to all cases
    - YAML serialization for easy configuration
    - Tagging and filtering capabilities

    Attributes:
        name: Suite name (e.g., "RAG Quality Suite")
        description: Human-readable description of what this suite tests
        cases: List of eval cases in this suite
        default_assertions: Assertions applied to all cases (unless overridden)
        setup: Callable run before executing the suite (e.g., start services)
        teardown: Callable run after executing the suite (e.g., cleanup)
        metadata: Additional metadata for the suite

    Example:
        >>> suite = EvalSuite(
        ...     name="RAG Quality Suite",
        ...     description="Tests for RAG pipeline quality",
        ...     cases=[
        ...         EvalCase(
        ...             id="test_basic_qa",
        ...             name="Basic factual question",
        ...             input=EvalInput(query="What is the capital of France?"),
        ...             expected=EvalExpected(contains=["Paris"])
        ...         )
        ...     ],
        ...     default_assertions=[
        ...         {"type": "latency", "max_ms": 5000},
        ...         {"type": "no_errors"}
        ...     ]
        ... )
    """

    name: str
    description: str = ""
    cases: list[EvalCase] = field(default_factory=list)
    default_assertions: list[dict[str, Any]] | None = None
    setup: Callable[[], None] | None = None
    teardown: Callable[[], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate suite configuration."""
        if not self.name:
            raise ValueError("EvalSuite must have a non-empty 'name'")

    def add_case(self, case: EvalCase) -> None:
        """Add a test case to the suite.

        Args:
            case: Eval case to add

        Example:
            >>> suite = EvalSuite(name="My Suite")
            >>> case = EvalCase(
            ...     id="test_1",
            ...     name="Test",
            ...     input=EvalInput(query="Hello"),
            ...     expected=EvalExpected(contains=["Hi"])
            ... )
            >>> suite.add_case(case)
        """
        self.cases.append(case)

    def get_case(self, case_id: str) -> EvalCase | None:
        """Get a test case by ID.

        Args:
            case_id: ID of the test case to retrieve

        Returns:
            EvalCase if found, None otherwise

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[...])
            >>> case = suite.get_case("test_basic_qa")
        """
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    def filter_by_tags(self, tags: list[str]) -> list[EvalCase]:
        """Filter test cases by tags.

        Returns cases that have ALL specified tags.

        Args:
            tags: List of tags to filter by

        Returns:
            List of matching test cases

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[...])
            >>> qa_cases = suite.filter_by_tags(["qa"])
            >>> geography_qa = suite.filter_by_tags(["qa", "geography"])
        """
        return [case for case in self.cases if all(tag in case.tags for tag in tags)]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalSuite:
        """Create EvalSuite from dictionary.

        Args:
            data: Dictionary with suite specification

        Returns:
            EvalSuite instance

        Example:
            >>> data = {
            ...     "name": "My Suite",
            ...     "description": "Test suite",
            ...     "cases": [
            ...         {
            ...             "id": "test_1",
            ...             "name": "Test",
            ...             "input": {"query": "Hello"},
            ...             "expected": {"contains": ["Hi"]}
            ...         }
            ...     ]
            ... }
            >>> suite = EvalSuite.from_dict(data)
        """
        # Parse cases
        cases_data = data.get("cases", [])
        cases = [EvalCase.from_dict(case_data) for case_data in cases_data]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            cases=cases,
            default_assertions=data.get("default_assertions"),
            metadata=data.get("metadata", {}),
            # Note: setup/teardown can't be serialized, only set programmatically
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the suite.

        Note:
            setup and teardown callables are not serialized.

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[...])
            >>> data = suite.to_dict()
            >>> data["name"]
            'My Suite'
        """
        result: dict[str, Any] = {
            "name": self.name,
        }

        if self.description:
            result["description"] = self.description

        if len(self.cases) > 0:
            result["cases"] = [case.to_dict() for case in self.cases]

        if self.default_assertions is not None and len(self.default_assertions) > 0:
            result["default_assertions"] = self.default_assertions

        if len(self.metadata) > 0:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalSuite:
        """Load eval suite from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            EvalSuite instance

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails

        Example:
            >>> suite = EvalSuite.from_yaml("tests/suite.yaml")
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save eval suite to YAML file.

        Args:
            path: Path to save YAML file

        Raises:
            ImportError: If PyYAML is not installed

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[...])
            >>> suite.to_yaml("suite.yaml")
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def __len__(self) -> int:
        """Return number of test cases in suite.

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[case1, case2])
            >>> len(suite)
            2
        """
        return len(self.cases)

    def __iter__(self):
        """Iterate over test cases.

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[case1, case2])
            >>> for case in suite:
            ...     print(case.name)
        """
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvalCase:
        """Get test case by index.

        Args:
            index: Index of the test case

        Returns:
            EvalCase at the specified index

        Example:
            >>> suite = EvalSuite(name="My Suite", cases=[case1, case2])
            >>> first_case = suite[0]
        """
        return self.cases[index]
