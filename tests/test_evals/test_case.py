"""Tests for prela.evals.case module."""

from __future__ import annotations

import pytest

from prela.evals.case import EvalCase, EvalExpected, EvalInput


class TestEvalInput:
    """Tests for EvalInput class."""

    def test_init_with_query(self):
        """Test initialization with query."""
        input_obj = EvalInput(query="What is 2+2?")
        assert input_obj.query == "What is 2+2?"
        assert input_obj.messages is None
        assert input_obj.context is None

    def test_init_with_messages(self):
        """Test initialization with messages."""
        messages = [{"role": "user", "content": "Hello"}]
        input_obj = EvalInput(messages=messages)
        assert input_obj.query is None
        assert input_obj.messages == messages
        assert input_obj.context is None

    def test_init_with_query_and_context(self):
        """Test initialization with query and context."""
        context = {"user_id": "123", "session": "abc"}
        input_obj = EvalInput(query="Hello", context=context)
        assert input_obj.query == "Hello"
        assert input_obj.context == context

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123"}
        input_obj = EvalInput(query="Hello", messages=messages, context=context)
        assert input_obj.query == "Hello"
        assert input_obj.messages == messages
        assert input_obj.context == context

    def test_init_without_query_or_messages(self):
        """Test that initialization fails without query or messages."""
        with pytest.raises(ValueError, match="must have either 'query' or 'messages'"):
            EvalInput(context={"key": "value"})

    def test_to_agent_input_with_query(self):
        """Test to_agent_input with query."""
        input_obj = EvalInput(query="Hello")
        result = input_obj.to_agent_input()
        assert result == {"query": "Hello"}

    def test_to_agent_input_with_messages(self):
        """Test to_agent_input with messages."""
        messages = [{"role": "user", "content": "Hello"}]
        input_obj = EvalInput(messages=messages)
        result = input_obj.to_agent_input()
        assert result == {"messages": messages}

    def test_to_agent_input_with_all_fields(self):
        """Test to_agent_input with all fields."""
        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123"}
        input_obj = EvalInput(query="Hello", messages=messages, context=context)
        result = input_obj.to_agent_input()
        assert result == {"query": "Hello", "messages": messages, "context": context}

    def test_from_dict_with_query(self):
        """Test from_dict with query."""
        data = {"query": "What is 2+2?"}
        input_obj = EvalInput.from_dict(data)
        assert input_obj.query == "What is 2+2?"
        assert input_obj.messages is None
        assert input_obj.context is None

    def test_from_dict_with_messages(self):
        """Test from_dict with messages."""
        messages = [{"role": "user", "content": "Hello"}]
        data = {"messages": messages}
        input_obj = EvalInput.from_dict(data)
        assert input_obj.query is None
        assert input_obj.messages == messages

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123"}
        data = {"query": "Hello", "messages": messages, "context": context}
        input_obj = EvalInput.from_dict(data)
        assert input_obj.query == "Hello"
        assert input_obj.messages == messages
        assert input_obj.context == context

    def test_to_dict_with_query(self):
        """Test to_dict with query."""
        input_obj = EvalInput(query="Hello")
        result = input_obj.to_dict()
        assert result == {"query": "Hello"}

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields."""
        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123"}
        input_obj = EvalInput(query="Hello", messages=messages, context=context)
        result = input_obj.to_dict()
        assert result == {"query": "Hello", "messages": messages, "context": context}

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works."""
        messages = [{"role": "user", "content": "Hello"}]
        context = {"user_id": "123"}
        input_obj = EvalInput(query="Hello", messages=messages, context=context)
        data = input_obj.to_dict()
        restored = EvalInput.from_dict(data)
        assert restored.query == input_obj.query
        assert restored.messages == input_obj.messages
        assert restored.context == input_obj.context


class TestEvalExpected:
    """Tests for EvalExpected class."""

    def test_init_with_output(self):
        """Test initialization with exact output."""
        expected = EvalExpected(output="The answer is 42")
        assert expected.output == "The answer is 42"
        assert expected.contains is None
        assert expected.not_contains is None

    def test_init_with_contains(self):
        """Test initialization with contains."""
        expected = EvalExpected(contains=["Paris", "capital"])
        assert expected.output is None
        assert expected.contains == ["Paris", "capital"]

    def test_init_with_not_contains(self):
        """Test initialization with not_contains."""
        expected = EvalExpected(not_contains=["London", "Berlin"])
        assert expected.not_contains == ["London", "Berlin"]

    def test_init_with_tool_calls(self):
        """Test initialization with tool_calls."""
        tool_calls = [{"name": "search", "args": {"query": "weather"}}]
        expected = EvalExpected(tool_calls=tool_calls)
        assert expected.tool_calls == tool_calls

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"confidence": 0.9, "source": "database"}
        expected = EvalExpected(metadata=metadata)
        assert expected.metadata == metadata

    def test_init_with_multiple_expectations(self):
        """Test initialization with multiple expectations."""
        expected = EvalExpected(
            contains=["Paris"], not_contains=["London"], metadata={"confidence": 0.9}
        )
        assert expected.contains == ["Paris"]
        assert expected.not_contains == ["London"]
        assert expected.metadata == {"confidence": 0.9}

    def test_init_without_expectations(self):
        """Test that initialization fails without any expectations."""
        with pytest.raises(ValueError, match="must have at least one expectation"):
            EvalExpected()

    def test_from_dict_with_output(self):
        """Test from_dict with output."""
        data = {"output": "The answer is 42"}
        expected = EvalExpected.from_dict(data)
        assert expected.output == "The answer is 42"

    def test_from_dict_with_contains(self):
        """Test from_dict with contains."""
        data = {"contains": ["Paris", "capital"]}
        expected = EvalExpected.from_dict(data)
        assert expected.contains == ["Paris", "capital"]

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        data = {
            "output": "Answer",
            "contains": ["Paris"],
            "not_contains": ["London"],
            "tool_calls": [{"name": "search"}],
            "metadata": {"confidence": 0.9},
        }
        expected = EvalExpected.from_dict(data)
        assert expected.output == "Answer"
        assert expected.contains == ["Paris"]
        assert expected.not_contains == ["London"]
        assert expected.tool_calls == [{"name": "search"}]
        assert expected.metadata == {"confidence": 0.9}

    def test_to_dict_with_output(self):
        """Test to_dict with output."""
        expected = EvalExpected(output="The answer is 42")
        result = expected.to_dict()
        assert result == {"output": "The answer is 42"}

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields."""
        expected = EvalExpected(
            output="Answer",
            contains=["Paris"],
            not_contains=["London"],
            tool_calls=[{"name": "search"}],
            metadata={"confidence": 0.9},
        )
        result = expected.to_dict()
        assert result == {
            "output": "Answer",
            "contains": ["Paris"],
            "not_contains": ["London"],
            "tool_calls": [{"name": "search"}],
            "metadata": {"confidence": 0.9},
        }

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works."""
        expected = EvalExpected(
            contains=["Paris"], not_contains=["London"], metadata={"confidence": 0.9}
        )
        data = expected.to_dict()
        restored = EvalExpected.from_dict(data)
        assert restored.contains == expected.contains
        assert restored.not_contains == expected.not_contains
        assert restored.metadata == expected.metadata


class TestEvalCase:
    """Tests for EvalCase class."""

    def test_init_with_minimal_fields(self):
        """Test initialization with minimal fields."""
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        assert case.id == "test_1"
        assert case.name == "Test case"
        assert case.input.query == "Hello"
        assert case.expected.contains == ["Hi"]
        assert case.assertions is None
        assert case.tags == []
        assert case.timeout_seconds == 30.0

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        assertions = [{"type": "contains", "value": "Paris"}]
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            assertions=assertions,
            tags=["qa", "greeting"],
            timeout_seconds=10.0,
            metadata={"priority": "high"},
        )
        assert case.assertions == assertions
        assert case.tags == ["qa", "greeting"]
        assert case.timeout_seconds == 10.0
        assert case.metadata == {"priority": "high"}

    def test_init_with_assertions_only(self):
        """Test initialization with assertions instead of expected."""
        assertions = [{"type": "contains", "value": "Paris"}]
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            assertions=assertions,
        )
        assert case.expected is None
        assert case.assertions == assertions

    def test_init_without_id(self):
        """Test that initialization fails without id."""
        with pytest.raises(ValueError, match="must have a non-empty 'id'"):
            EvalCase(
                id="",
                name="Test",
                input=EvalInput(query="Hello"),
                expected=EvalExpected(contains=["Hi"]),
            )

    def test_init_without_name(self):
        """Test that initialization fails without name."""
        with pytest.raises(ValueError, match="must have a non-empty 'name'"):
            EvalCase(
                id="test_1",
                name="",
                input=EvalInput(query="Hello"),
                expected=EvalExpected(contains=["Hi"]),
            )

    def test_init_without_expected_or_assertions(self):
        """Test that initialization fails without expected or assertions."""
        with pytest.raises(
            ValueError, match="must have either 'expected' or 'assertions'"
        ):
            EvalCase(id="test_1", name="Test", input=EvalInput(query="Hello"))

    def test_init_with_negative_timeout(self):
        """Test that initialization fails with negative timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            EvalCase(
                id="test_1",
                name="Test",
                input=EvalInput(query="Hello"),
                expected=EvalExpected(contains=["Hi"]),
                timeout_seconds=-1.0,
            )

    def test_init_with_zero_timeout(self):
        """Test that initialization fails with zero timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            EvalCase(
                id="test_1",
                name="Test",
                input=EvalInput(query="Hello"),
                expected=EvalExpected(contains=["Hi"]),
                timeout_seconds=0.0,
            )

    def test_from_dict_with_minimal_fields(self):
        """Test from_dict with minimal fields."""
        data = {
            "id": "test_1",
            "name": "Test case",
            "input": {"query": "Hello"},
            "expected": {"contains": ["Hi"]},
        }
        case = EvalCase.from_dict(data)
        assert case.id == "test_1"
        assert case.name == "Test case"
        assert case.input.query == "Hello"
        assert case.expected.contains == ["Hi"]

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        data = {
            "id": "test_1",
            "name": "Test case",
            "input": {"query": "Hello"},
            "expected": {"contains": ["Hi"]},
            "assertions": [{"type": "contains", "value": "Paris"}],
            "tags": ["qa", "greeting"],
            "timeout_seconds": 10.0,
            "metadata": {"priority": "high"},
        }
        case = EvalCase.from_dict(data)
        assert case.id == "test_1"
        assert case.name == "Test case"
        assert case.assertions == [{"type": "contains", "value": "Paris"}]
        assert case.tags == ["qa", "greeting"]
        assert case.timeout_seconds == 10.0
        assert case.metadata == {"priority": "high"}

    def test_from_dict_with_input_object(self):
        """Test from_dict with EvalInput object instead of dict."""
        data = {
            "id": "test_1",
            "name": "Test case",
            "input": EvalInput(query="Hello"),
            "expected": {"contains": ["Hi"]},
        }
        case = EvalCase.from_dict(data)
        assert case.input.query == "Hello"

    def test_from_dict_with_expected_object(self):
        """Test from_dict with EvalExpected object instead of dict."""
        data = {
            "id": "test_1",
            "name": "Test case",
            "input": {"query": "Hello"},
            "expected": EvalExpected(contains=["Hi"]),
        }
        case = EvalCase.from_dict(data)
        assert case.expected.contains == ["Hi"]

    def test_from_dict_without_input(self):
        """Test that from_dict fails without input."""
        data = {
            "id": "test_1",
            "name": "Test case",
            "expected": {"contains": ["Hi"]},
        }
        with pytest.raises(ValueError, match="must have 'input' field"):
            EvalCase.from_dict(data)

    def test_to_dict_with_minimal_fields(self):
        """Test to_dict with minimal fields."""
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        result = case.to_dict()
        assert result == {
            "id": "test_1",
            "name": "Test case",
            "input": {"query": "Hello"},
            "expected": {"contains": ["Hi"]},
            "timeout_seconds": 30.0,
        }

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields."""
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            assertions=[{"type": "contains", "value": "Paris"}],
            tags=["qa", "greeting"],
            timeout_seconds=10.0,
            metadata={"priority": "high"},
        )
        result = case.to_dict()
        assert result == {
            "id": "test_1",
            "name": "Test case",
            "input": {"query": "Hello"},
            "expected": {"contains": ["Hi"]},
            "assertions": [{"type": "contains", "value": "Paris"}],
            "tags": ["qa", "greeting"],
            "timeout_seconds": 10.0,
            "metadata": {"priority": "high"},
        }

    def test_to_dict_with_assertions_only(self):
        """Test to_dict with assertions instead of expected."""
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            assertions=[{"type": "contains", "value": "Paris"}],
        )
        result = case.to_dict()
        assert "expected" not in result
        assert result["assertions"] == [{"type": "contains", "value": "Paris"}]

    def test_to_dict_excludes_empty_fields(self):
        """Test that to_dict excludes empty tags, assertions, metadata."""
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            tags=[],
            assertions=None,
            metadata={},
        )
        result = case.to_dict()
        assert "tags" not in result
        assert "assertions" not in result
        assert "metadata" not in result

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works."""
        case = EvalCase(
            id="test_1",
            name="Test case",
            input=EvalInput(query="Hello", context={"user_id": "123"}),
            expected=EvalExpected(contains=["Hi"], not_contains=["Bye"]),
            assertions=[{"type": "contains", "value": "Paris"}],
            tags=["qa", "greeting"],
            timeout_seconds=10.0,
            metadata={"priority": "high"},
        )
        data = case.to_dict()
        restored = EvalCase.from_dict(data)
        assert restored.id == case.id
        assert restored.name == case.name
        assert restored.input.query == case.input.query
        assert restored.input.context == case.input.context
        assert restored.expected.contains == case.expected.contains
        assert restored.expected.not_contains == case.expected.not_contains
        assert restored.assertions == case.assertions
        assert restored.tags == case.tags
        assert restored.timeout_seconds == case.timeout_seconds
        assert restored.metadata == case.metadata
