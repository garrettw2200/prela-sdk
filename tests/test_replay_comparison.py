"""Tests for replay comparison functionality."""

from __future__ import annotations

import pytest

from prela.replay.comparison import ReplayComparator, compare_replays
from prela.replay.result import ReplayResult, ReplayedSpan


def _has_sentence_transformers() -> bool:
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestReplayComparator:
    """Test ReplayComparator class."""

    def test_init_default(self):
        """Comparator initializes with semantic similarity by default (when available)."""
        comparator = ReplayComparator()
        # Semantic similarity is enabled by default if sentence-transformers is available
        # Otherwise it falls back to False
        expected = _has_sentence_transformers()
        assert comparator.use_semantic_similarity is expected

    def test_init_no_semantic_similarity(self):
        """Comparator can disable semantic similarity."""
        comparator = ReplayComparator(use_semantic_similarity=False)
        assert comparator.use_semantic_similarity is False

    def test_init_missing_sentence_transformers(self, monkeypatch):
        """Comparator handles missing sentence-transformers gracefully."""
        # Mock import to fail
        import sys

        def mock_import(name, *args, **kwargs):
            if "sentence_transformers" in name:
                raise ImportError("sentence-transformers not installed")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        comparator = ReplayComparator()
        # Should disable semantic similarity gracefully
        assert comparator.use_semantic_similarity is False

    def test_compare_identical_results(self):
        """compare returns no differences for identical results."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={"prompt": "Hello"},
            output="Hi there!",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span1])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        assert len(comparison.differences) == 0
        assert comparison.identical_spans == 1
        assert comparison.changed_spans == 0

    def test_compare_different_outputs(self):
        """compare detects different outputs."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={"prompt": "Hello"},
            output="Hi there!",
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={"prompt": "Hello"},
            output="Hello world!",  # Different output
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        assert len(comparison.differences) == 1
        assert comparison.differences[0].field == "output"
        assert comparison.differences[0].original_value == "Hi there!"
        assert comparison.differences[0].modified_value == "Hello world!"

    def test_compare_different_inputs(self):
        """compare detects different inputs."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={"prompt": "Hello"},
            output="Hi!",
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={"prompt": "Goodbye"},  # Different input
            output="Hi!",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        assert len(comparison.differences) == 1
        assert comparison.differences[0].field == "input"

    def test_compare_duration_significant_change(self):
        """compare detects significant duration changes."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            duration_ms=100.0,
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            duration_ms=200.0,  # 100% increase
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        # Should detect duration change (>10% threshold)
        duration_diffs = [d for d in comparison.differences if d.field == "duration_ms"]
        assert len(duration_diffs) == 1

    def test_compare_duration_small_change(self):
        """compare ignores small duration changes."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            duration_ms=100.0,
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            duration_ms=105.0,  # Only 5% increase
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        # Should ignore small duration change (<10% threshold)
        duration_diffs = [d for d in comparison.differences if d.field == "duration_ms"]
        assert len(duration_diffs) == 0

    def test_compare_tokens_changed(self):
        """compare detects token usage changes."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            tokens_used=100,
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            tokens_used=150,  # Different tokens
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        token_diffs = [d for d in comparison.differences if d.field == "tokens_used"]
        assert len(token_diffs) == 1
        assert token_diffs[0].original_value == 100
        assert token_diffs[0].modified_value == 150

    def test_compare_cost_changed(self):
        """compare detects cost changes."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            cost_usd=0.01,
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            cost_usd=0.02,  # Different cost
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        cost_diffs = [d for d in comparison.differences if d.field == "cost_usd"]
        assert len(cost_diffs) == 1

    def test_compare_error_status(self):
        """compare detects error status changes."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            error=None,
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output=None,
            error="API rate limit exceeded",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        error_diffs = [d for d in comparison.differences if d.field == "error"]
        assert len(error_diffs) == 1
        assert error_diffs[0].modified_value == "API rate limit exceeded"

    def test_compare_missing_span_in_modified(self):
        """compare detects spans missing in modified result."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        existence_diffs = [d for d in comparison.differences if d.field == "existence"]
        assert len(existence_diffs) == 1
        assert existence_diffs[0].modified_value is None

    def test_compare_missing_span_in_original(self):
        """compare detects spans missing in original result."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[])
        result2 = ReplayResult(trace_id="trace-1", spans=[span1])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        existence_diffs = [d for d in comparison.differences if d.field == "existence"]
        assert len(existence_diffs) == 1
        assert existence_diffs[0].original_value is None

    def test_compare_generates_summary(self):
        """compare generates human-readable summary."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
            tokens_used=100,
            cost_usd=0.01,
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hello!",
            tokens_used=150,
            cost_usd=0.015,
        )

        result1 = ReplayResult(
            trace_id="trace-1", spans=[span1], total_tokens=100, total_cost_usd=0.01
        )
        result2 = ReplayResult(
            trace_id="trace-1", spans=[span2], total_tokens=150, total_cost_usd=0.015
        )

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        summary = comparison.generate_summary()
        assert "Replay Comparison Summary" in summary
        assert "Total Spans: 1" in summary
        assert "Cost:" in summary
        assert "Tokens:" in summary

    def test_compare_warns_different_trace_ids(self, caplog):
        """compare warns when comparing different trace IDs."""
        result1 = ReplayResult(trace_id="trace-1", spans=[])
        result2 = ReplayResult(trace_id="trace-2", spans=[])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparator.compare(result1, result2)

        assert "different traces" in caplog.text

    def test_compare_replays_convenience_function(self):
        """compare_replays convenience function works."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi!",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span1])

        comparison = compare_replays(result1, result2, use_semantic_similarity=False)

        assert len(comparison.differences) == 0

    @pytest.mark.skipif(
        not _has_sentence_transformers(),
        reason="sentence-transformers not installed",
    )
    def test_compute_semantic_similarity(self):
        """_compute_semantic_similarity calculates cosine similarity."""
        comparator = ReplayComparator(use_semantic_similarity=True)

        # Similar texts should have high similarity
        sim = comparator._compute_semantic_similarity(
            "Hello, how are you?", "Hi, how are you doing?"
        )
        assert 0.7 < sim <= 1.0

        # Different texts should have lower similarity
        sim = comparator._compute_semantic_similarity(
            "Hello, how are you?", "The weather is nice today."
        )
        assert 0.0 <= sim < 0.7

    @pytest.mark.skipif(
        not _has_sentence_transformers(),
        reason="sentence-transformers not installed",
    )
    def test_compare_with_semantic_similarity(self):
        """compare computes semantic similarity for text outputs."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hello, how are you?",
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hi, how are you doing?",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        comparator = ReplayComparator(use_semantic_similarity=True)
        comparison = comparator.compare(result1, result2)

        # Should detect difference but also compute similarity
        assert len(comparison.differences) == 1
        diff = comparison.differences[0]
        assert diff.semantic_similarity is not None
        assert 0.7 < diff.semantic_similarity <= 1.0  # Similar texts
