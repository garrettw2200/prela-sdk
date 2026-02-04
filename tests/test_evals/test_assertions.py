"""
Tests for evaluation assertions.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

from prela.core.span import Span, SpanType, SpanStatus
from prela.evals.assertions import (
    AssertionResult,
    BaseAssertion,
    ContainsAssertion,
    JSONValidAssertion,
    LengthAssertion,
    NotContainsAssertion,
    RegexAssertion,
    ToolArgsAssertion,
    ToolCalledAssertion,
    ToolSequenceAssertion,
)


# ============================================================================
# AssertionResult Tests
# ============================================================================


class TestAssertionResult:
    """Tests for AssertionResult dataclass."""

    def test_init_minimal(self):
        """Test creating result with minimal fields."""
        result = AssertionResult(
            passed=True,
            assertion_type="test",
            message="Test passed",
        )
        assert result.passed is True
        assert result.assertion_type == "test"
        assert result.message == "Test passed"
        assert result.score is None
        assert result.expected is None
        assert result.actual is None
        assert result.details == {}

    def test_init_full(self):
        """Test creating result with all fields."""
        result = AssertionResult(
            passed=False,
            assertion_type="test",
            message="Test failed",
            score=0.5,
            expected="foo",
            actual="bar",
            details={"reason": "mismatch"},
        )
        assert result.passed is False
        assert result.score == 0.5
        assert result.expected == "foo"
        assert result.actual == "bar"
        assert result.details == {"reason": "mismatch"}

    def test_str_passed(self):
        """Test string representation for passed result."""
        result = AssertionResult(
            passed=True,
            assertion_type="contains",
            message="Output contains 'hello'",
        )
        s = str(result)
        assert "✓ PASS" in s
        assert "[contains]" in s
        assert "Output contains 'hello'" in s

    def test_str_failed(self):
        """Test string representation for failed result."""
        result = AssertionResult(
            passed=False,
            assertion_type="regex",
            message="Pattern not found",
        )
        s = str(result)
        assert "✗ FAIL" in s
        assert "[regex]" in s
        assert "Pattern not found" in s

    def test_str_with_score(self):
        """Test string representation with score."""
        result = AssertionResult(
            passed=True,
            assertion_type="semantic",
            message="Semantically similar",
            score=0.87,
        )
        s = str(result)
        assert "(score: 0.87)" in s


# ============================================================================
# ContainsAssertion Tests
# ============================================================================


class TestContainsAssertion:
    """Tests for ContainsAssertion."""

    def test_init(self):
        """Test initialization."""
        assertion = ContainsAssertion(text="hello", case_sensitive=True)
        assert assertion.text == "hello"
        assert assertion.case_sensitive is True

    def test_from_config(self):
        """Test creating from config."""
        config = {"text": "hello", "case_sensitive": False}
        assertion = ContainsAssertion.from_config(config)
        assert assertion.text == "hello"
        assert assertion.case_sensitive is False

    def test_from_config_default_case_sensitive(self):
        """Test default case_sensitive is True."""
        config = {"text": "hello"}
        assertion = ContainsAssertion.from_config(config)
        assert assertion.case_sensitive is True

    def test_from_config_missing_text(self):
        """Test error when text is missing."""
        with pytest.raises(ValueError, match="requires 'text'"):
            ContainsAssertion.from_config({})

    def test_evaluate_pass_case_sensitive(self):
        """Test evaluation passes with exact case match."""
        assertion = ContainsAssertion(text="hello", case_sensitive=True)
        result = assertion.evaluate(output="hello world", expected=None, trace=None)
        assert result.passed is True
        assert result.assertion_type == "contains"
        assert "contains" in result.message.lower()

    def test_evaluate_fail_case_sensitive(self):
        """Test evaluation fails with case mismatch."""
        assertion = ContainsAssertion(text="hello", case_sensitive=True)
        result = assertion.evaluate(output="Hello world", expected=None, trace=None)
        assert result.passed is False

    def test_evaluate_pass_case_insensitive(self):
        """Test evaluation passes with case insensitive match."""
        assertion = ContainsAssertion(text="hello", case_sensitive=False)
        result = assertion.evaluate(output="HELLO WORLD", expected=None, trace=None)
        assert result.passed is True

    def test_evaluate_fail_not_found(self):
        """Test evaluation fails when text not found."""
        assertion = ContainsAssertion(text="goodbye", case_sensitive=False)
        result = assertion.evaluate(output="hello world", expected=None, trace=None)
        assert result.passed is False
        assert "does not contain" in result.message.lower()

    def test_repr(self):
        """Test repr."""
        assertion = ContainsAssertion(text="hello", case_sensitive=False)
        r = repr(assertion)
        assert "ContainsAssertion" in r
        assert "hello" in r
        assert "False" in r


# ============================================================================
# NotContainsAssertion Tests
# ============================================================================


class TestNotContainsAssertion:
    """Tests for NotContainsAssertion."""

    def test_evaluate_pass_not_found(self):
        """Test evaluation passes when text not found."""
        assertion = NotContainsAssertion(text="error", case_sensitive=True)
        result = assertion.evaluate(output="Success!", expected=None, trace=None)
        assert result.passed is True
        assert "does not contain" in result.message.lower()

    def test_evaluate_fail_found(self):
        """Test evaluation fails when text is found."""
        assertion = NotContainsAssertion(text="error", case_sensitive=True)
        result = assertion.evaluate(output="error occurred", expected=None, trace=None)
        assert result.passed is False
        assert "incorrectly contains" in result.message.lower()

    def test_evaluate_case_insensitive(self):
        """Test case insensitive matching."""
        assertion = NotContainsAssertion(text="ERROR", case_sensitive=False)
        result = assertion.evaluate(output="error occurred", expected=None, trace=None)
        assert result.passed is False


# ============================================================================
# RegexAssertion Tests
# ============================================================================


class TestRegexAssertion:
    """Tests for RegexAssertion."""

    def test_init(self):
        """Test initialization."""
        assertion = RegexAssertion(pattern=r"\d+", flags=0)
        assert assertion.pattern == r"\d+"
        assert assertion.flags == 0

    def test_from_config(self):
        """Test creating from config."""
        config = {"pattern": r"\d{3}-\d{4}", "flags": re.IGNORECASE}
        assertion = RegexAssertion.from_config(config)
        assert assertion.pattern == r"\d{3}-\d{4}"
        assert assertion.flags == re.IGNORECASE

    def test_from_config_missing_pattern(self):
        """Test error when pattern is missing."""
        with pytest.raises(ValueError, match="requires 'pattern'"):
            RegexAssertion.from_config({})

    def test_evaluate_pass(self):
        """Test evaluation passes when pattern matches."""
        assertion = RegexAssertion(pattern=r"\d{3}-\d{4}")
        result = assertion.evaluate(output="Call 555-1234", expected=None, trace=None)
        assert result.passed is True
        assert result.details["matched_text"] == "555-1234"

    def test_evaluate_fail(self):
        """Test evaluation fails when pattern doesn't match."""
        assertion = RegexAssertion(pattern=r"\d{3}-\d{4}")
        result = assertion.evaluate(output="No phone number", expected=None, trace=None)
        assert result.passed is False

    def test_evaluate_with_flags(self):
        """Test evaluation with regex flags."""
        assertion = RegexAssertion(pattern=r"hello", flags=re.IGNORECASE)
        result = assertion.evaluate(output="HELLO WORLD", expected=None, trace=None)
        assert result.passed is True


# ============================================================================
# LengthAssertion Tests
# ============================================================================


class TestLengthAssertion:
    """Tests for LengthAssertion."""

    def test_init_both_bounds(self):
        """Test initialization with both bounds."""
        assertion = LengthAssertion(min_length=10, max_length=100)
        assert assertion.min_length == 10
        assert assertion.max_length == 100

    def test_init_min_only(self):
        """Test initialization with only min."""
        assertion = LengthAssertion(min_length=10)
        assert assertion.min_length == 10
        assert assertion.max_length is None

    def test_init_max_only(self):
        """Test initialization with only max."""
        assertion = LengthAssertion(max_length=100)
        assert assertion.min_length is None
        assert assertion.max_length == 100

    def test_init_neither(self):
        """Test error when neither bound specified."""
        with pytest.raises(ValueError, match="At least one"):
            LengthAssertion()

    def test_init_negative_min(self):
        """Test error with negative min."""
        with pytest.raises(ValueError, match="non-negative"):
            LengthAssertion(min_length=-1)

    def test_init_negative_max(self):
        """Test error with negative max."""
        with pytest.raises(ValueError, match="non-negative"):
            LengthAssertion(max_length=-1)

    def test_init_min_greater_than_max(self):
        """Test error when min > max."""
        with pytest.raises(ValueError, match="cannot be greater"):
            LengthAssertion(min_length=100, max_length=10)

    def test_evaluate_pass_within_bounds(self):
        """Test evaluation passes when within bounds."""
        assertion = LengthAssertion(min_length=5, max_length=20)
        result = assertion.evaluate(output="hello world", expected=None, trace=None)
        assert result.passed is True
        assert result.actual == 11

    def test_evaluate_fail_too_short(self):
        """Test evaluation fails when too short."""
        assertion = LengthAssertion(min_length=20)
        result = assertion.evaluate(output="hi", expected=None, trace=None)
        assert result.passed is False
        assert "too short" in result.message

    def test_evaluate_fail_too_long(self):
        """Test evaluation fails when too long."""
        assertion = LengthAssertion(max_length=5)
        result = assertion.evaluate(output="hello world", expected=None, trace=None)
        assert result.passed is False
        assert "too long" in result.message

    def test_evaluate_edge_case_exact_min(self):
        """Test exact minimum length passes."""
        assertion = LengthAssertion(min_length=5)
        result = assertion.evaluate(output="hello", expected=None, trace=None)
        assert result.passed is True

    def test_evaluate_edge_case_exact_max(self):
        """Test exact maximum length passes."""
        assertion = LengthAssertion(max_length=5)
        result = assertion.evaluate(output="hello", expected=None, trace=None)
        assert result.passed is True


# ============================================================================
# JSONValidAssertion Tests
# ============================================================================


class TestJSONValidAssertion:
    """Tests for JSONValidAssertion."""

    def test_init_no_schema(self):
        """Test initialization without schema."""
        assertion = JSONValidAssertion()
        assert assertion.schema is None
        assert assertion._validator is None

    def test_init_with_schema(self):
        """Test initialization with schema (requires jsonschema)."""
        pytest.importorskip("jsonschema")
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assertion = JSONValidAssertion(schema=schema)
        assert assertion.schema == schema
        assert assertion._validator is not None

    def test_evaluate_valid_json(self):
        """Test evaluation passes for valid JSON."""
        assertion = JSONValidAssertion()
        result = assertion.evaluate(
            output='{"key": "value"}',
            expected=None,
            trace=None,
        )
        assert result.passed is True
        assert result.actual == {"key": "value"}

    def test_evaluate_invalid_json(self):
        """Test evaluation fails for invalid JSON."""
        assertion = JSONValidAssertion()
        result = assertion.evaluate(
            output='{invalid json}',
            expected=None,
            trace=None,
        )
        assert result.passed is False
        assert "not valid JSON" in result.message

    def test_evaluate_with_schema_valid(self):
        """Test evaluation with schema validation (valid)."""
        pytest.importorskip("jsonschema")
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assertion = JSONValidAssertion(schema=schema)
        result = assertion.evaluate(
            output='{"name": "Alice"}',
            expected=None,
            trace=None,
        )
        assert result.passed is True

    def test_evaluate_with_schema_invalid(self):
        """Test evaluation with schema validation (invalid)."""
        pytest.importorskip("jsonschema")
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assertion = JSONValidAssertion(schema=schema)
        result = assertion.evaluate(
            output='{"age": 30}',
            expected=None,
            trace=None,
        )
        assert result.passed is False
        assert "does not match schema" in result.message


# ============================================================================
# ToolCalledAssertion Tests
# ============================================================================


class TestToolCalledAssertion:
    """Tests for ToolCalledAssertion."""

    def create_tool_span(self, tool_name: str) -> Span:
        """Helper to create a tool span."""
        return Span(
            name=tool_name,
            span_type=SpanType.TOOL,
            trace_id="trace-123",
            span_id=f"span-{tool_name}",
            started_at=datetime.now(timezone.utc),
        )

    def test_init(self):
        """Test initialization."""
        assertion = ToolCalledAssertion(tool_name="web_search")
        assert assertion.tool_name == "web_search"

    def test_from_config(self):
        """Test creating from config."""
        config = {"tool_name": "calculator"}
        assertion = ToolCalledAssertion.from_config(config)
        assert assertion.tool_name == "calculator"

    def test_from_config_missing_tool_name(self):
        """Test error when tool_name is missing."""
        with pytest.raises(ValueError, match="requires 'tool_name'"):
            ToolCalledAssertion.from_config({})

    def test_evaluate_no_trace(self):
        """Test evaluation fails when no trace provided."""
        assertion = ToolCalledAssertion(tool_name="web_search")
        result = assertion.evaluate(output=None, expected=None, trace=None)
        assert result.passed is False
        assert "no trace" in result.message.lower()

    def test_evaluate_empty_trace(self):
        """Test evaluation fails with empty trace."""
        assertion = ToolCalledAssertion(tool_name="web_search")
        result = assertion.evaluate(output=None, expected=None, trace=[])
        assert result.passed is False

    def test_evaluate_pass_tool_found(self):
        """Test evaluation passes when tool is found."""
        assertion = ToolCalledAssertion(tool_name="web_search")
        span = self.create_tool_span("web_search")
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is True
        assert "was called" in result.message
        assert result.details["call_count"] == 1

    def test_evaluate_fail_tool_not_found(self):
        """Test evaluation fails when tool is not found."""
        assertion = ToolCalledAssertion(tool_name="calculator")
        span = self.create_tool_span("web_search")
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is False
        assert "was not called" in result.message
        assert "web_search" in result.details["available_tools"]

    def test_evaluate_multiple_calls(self):
        """Test detection of multiple tool calls."""
        assertion = ToolCalledAssertion(tool_name="web_search")
        spans = [
            self.create_tool_span("web_search"),
            self.create_tool_span("web_search"),
        ]
        result = assertion.evaluate(output=None, expected=None, trace=spans)
        assert result.passed is True
        assert result.details["call_count"] == 2


# ============================================================================
# ToolArgsAssertion Tests
# ============================================================================


class TestToolArgsAssertion:
    """Tests for ToolArgsAssertion."""

    def create_tool_span_with_args(self, tool_name: str, args: dict) -> Span:
        """Helper to create a tool span with arguments."""
        span = Span(
            name=tool_name,
            span_type=SpanType.TOOL,
            trace_id="trace-123",
            span_id=f"span-{tool_name}",
            started_at=datetime.now(timezone.utc),
        )
        # Add args as attributes
        for key, value in args.items():
            span.set_attribute(f"tool.input.{key}", value)
        return span

    def test_init(self):
        """Test initialization."""
        assertion = ToolArgsAssertion(
            tool_name="web_search",
            expected_args={"query": "Python"},
        )
        assert assertion.tool_name == "web_search"
        assert assertion.expected_args == {"query": "Python"}
        assert assertion.partial_match is True

    def test_from_config(self):
        """Test creating from config."""
        config = {
            "tool_name": "calculator",
            "expected_args": {"x": 5, "y": 10},
            "partial_match": False,
        }
        assertion = ToolArgsAssertion.from_config(config)
        assert assertion.tool_name == "calculator"
        assert assertion.expected_args == {"x": 5, "y": 10}
        assert assertion.partial_match is False

    def test_evaluate_pass_partial_match(self):
        """Test evaluation passes with partial match."""
        assertion = ToolArgsAssertion(
            tool_name="web_search",
            expected_args={"query": "Python"},
            partial_match=True,
        )
        span = self.create_tool_span_with_args(
            "web_search",
            {"query": "Python", "limit": 10},
        )
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is True

    def test_evaluate_fail_missing_arg(self):
        """Test evaluation fails when expected arg is missing."""
        assertion = ToolArgsAssertion(
            tool_name="web_search",
            expected_args={"query": "Python", "limit": 10},
            partial_match=True,
        )
        span = self.create_tool_span_with_args("web_search", {"query": "Python"})
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is False

    def test_evaluate_fail_wrong_value(self):
        """Test evaluation fails when arg has wrong value."""
        assertion = ToolArgsAssertion(
            tool_name="web_search",
            expected_args={"query": "Python"},
        )
        span = self.create_tool_span_with_args("web_search", {"query": "Java"})
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is False

    def test_evaluate_exact_match(self):
        """Test evaluation with exact match requirement."""
        assertion = ToolArgsAssertion(
            tool_name="calculator",
            expected_args={"x": 5, "y": 10},
            partial_match=False,
        )
        span = self.create_tool_span_with_args("calculator", {"x": 5, "y": 10})
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is True

    def test_evaluate_exact_match_fail_extra_arg(self):
        """Test exact match fails with extra args."""
        assertion = ToolArgsAssertion(
            tool_name="calculator",
            expected_args={"x": 5},
            partial_match=False,
        )
        span = self.create_tool_span_with_args("calculator", {"x": 5, "y": 10})
        result = assertion.evaluate(output=None, expected=None, trace=[span])
        assert result.passed is False


# ============================================================================
# ToolSequenceAssertion Tests
# ============================================================================


class TestToolSequenceAssertion:
    """Tests for ToolSequenceAssertion."""

    def create_tool_span(self, tool_name: str, offset_seconds: int = 0) -> Span:
        """Helper to create a tool span with timestamp offset."""
        from datetime import timedelta
        base_time = datetime.now(timezone.utc)
        adjusted_time = base_time + timedelta(seconds=offset_seconds)

        span = Span(
            name=tool_name,
            span_type=SpanType.TOOL,
            trace_id="trace-123",
            span_id=f"span-{tool_name}-{offset_seconds}",
            started_at=adjusted_time,
        )
        return span

    def test_init(self):
        """Test initialization."""
        assertion = ToolSequenceAssertion(sequence=["tool1", "tool2"])
        assert assertion.sequence == ["tool1", "tool2"]
        assert assertion.strict is False

    def test_init_empty_sequence(self):
        """Test error with empty sequence."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ToolSequenceAssertion(sequence=[])

    def test_from_config(self):
        """Test creating from config."""
        config = {"sequence": ["tool1", "tool2", "tool3"], "strict": True}
        assertion = ToolSequenceAssertion.from_config(config)
        assert assertion.sequence == ["tool1", "tool2", "tool3"]
        assert assertion.strict is True

    def test_evaluate_pass_non_strict(self):
        """Test evaluation passes in non-strict mode."""
        assertion = ToolSequenceAssertion(
            sequence=["search", "calculate"],
            strict=False,
        )
        spans = [
            self.create_tool_span("search", 0),
            self.create_tool_span("other", 1),
            self.create_tool_span("calculate", 2),
        ]
        result = assertion.evaluate(output=None, expected=None, trace=spans)
        assert result.passed is True

    def test_evaluate_fail_wrong_order(self):
        """Test evaluation fails with wrong order."""
        assertion = ToolSequenceAssertion(sequence=["search", "calculate"])
        spans = [
            self.create_tool_span("calculate", 0),
            self.create_tool_span("search", 1),
        ]
        result = assertion.evaluate(output=None, expected=None, trace=spans)
        assert result.passed is False

    def test_evaluate_pass_strict(self):
        """Test evaluation passes in strict mode."""
        assertion = ToolSequenceAssertion(
            sequence=["search", "calculate"],
            strict=True,
        )
        spans = [
            self.create_tool_span("search", 0),
            self.create_tool_span("calculate", 1),
        ]
        result = assertion.evaluate(output=None, expected=None, trace=spans)
        assert result.passed is True

    def test_evaluate_fail_strict_extra_tool(self):
        """Test evaluation fails in strict mode with extra tools."""
        assertion = ToolSequenceAssertion(
            sequence=["search", "calculate"],
            strict=True,
        )
        spans = [
            self.create_tool_span("search", 0),
            self.create_tool_span("other", 1),
            self.create_tool_span("calculate", 2),
        ]
        result = assertion.evaluate(output=None, expected=None, trace=spans)
        assert result.passed is False

    def test_evaluate_fail_incomplete_sequence(self):
        """Test evaluation fails when sequence is incomplete."""
        assertion = ToolSequenceAssertion(sequence=["search", "calculate", "summarize"])
        spans = [
            self.create_tool_span("search", 0),
            self.create_tool_span("calculate", 1),
        ]
        result = assertion.evaluate(output=None, expected=None, trace=spans)
        assert result.passed is False
        assert "incomplete" in result.message.lower()


# ============================================================================
# SemanticSimilarityAssertion Tests (Optional)
# ============================================================================


# Check if sentence_transformers is available
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed",
)
class TestSemanticSimilarityAssertion:
    """Tests for SemanticSimilarityAssertion."""

    def test_init(self):
        """Test initialization."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        assertion = SemanticSimilarityAssertion(
            expected_text="Hello world",
            threshold=0.8,
        )
        assert assertion.expected_text == "Hello world"
        assert assertion.threshold == 0.8
        assert assertion.model_name == "all-MiniLM-L6-v2"

    def test_init_invalid_threshold(self):
        """Test error with invalid threshold."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        with pytest.raises(ValueError, match="between 0 and 1"):
            SemanticSimilarityAssertion(expected_text="Hello", threshold=1.5)

    def test_from_config(self):
        """Test creating from config."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        config = {"expected_text": "Hello world", "threshold": 0.75}
        assertion = SemanticSimilarityAssertion.from_config(config)
        assert assertion.expected_text == "Hello world"
        assert assertion.threshold == 0.75

    def test_evaluate_high_similarity(self):
        """Test evaluation passes with high similarity."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        assertion = SemanticSimilarityAssertion(
            expected_text="The weather is nice today",
            threshold=0.7,
        )
        result = assertion.evaluate(
            output="Today has beautiful weather",
            expected=None,
            trace=None,
        )
        assert result.passed is True
        assert result.score is not None
        assert result.score > 0.7

    def test_evaluate_low_similarity(self):
        """Test evaluation fails with low similarity."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        assertion = SemanticSimilarityAssertion(
            expected_text="The weather is nice today",
            threshold=0.8,
        )
        result = assertion.evaluate(
            output="I like pizza",
            expected=None,
            trace=None,
        )
        assert result.passed is False
        assert result.score is not None
        assert result.score < 0.8

    def test_embedding_cache(self):
        """Test embedding cache is used."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        # Clear cache first
        SemanticSimilarityAssertion.clear_cache()
        initial_size = SemanticSimilarityAssertion.get_cache_size()

        assertion = SemanticSimilarityAssertion(
            expected_text="Hello world",
            threshold=0.8,
        )

        # First evaluation caches embeddings
        assertion.evaluate(output="Hello world", expected=None, trace=None)
        size_after_first = SemanticSimilarityAssertion.get_cache_size()
        assert size_after_first > initial_size

        # Second evaluation reuses cache
        assertion.evaluate(output="Hello world", expected=None, trace=None)
        size_after_second = SemanticSimilarityAssertion.get_cache_size()
        assert size_after_second == size_after_first

    def test_clear_cache(self):
        """Test cache clearing."""
        from prela.evals.assertions import SemanticSimilarityAssertion

        assertion = SemanticSimilarityAssertion(
            expected_text="Hello",
            threshold=0.8,
        )
        assertion.evaluate(output="Hi", expected=None, trace=None)

        assert SemanticSimilarityAssertion.get_cache_size() > 0
        SemanticSimilarityAssertion.clear_cache()
        assert SemanticSimilarityAssertion.get_cache_size() == 0
