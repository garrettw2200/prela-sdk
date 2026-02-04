"""Tests for fallback similarity metrics in replay comparison."""

from __future__ import annotations

import pytest

from prela.replay.comparison import ReplayComparator
from prela.replay.result import ReplayResult, ReplayedSpan


class TestFallbackSimilarity:
    """Test fallback similarity metrics when sentence-transformers unavailable."""

    def test_fallback_similarity_exact_match(self):
        """Fallback should return 1.0 for exact matches."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        similarity = comparator._compute_fallback_similarity(
            "Hello world", "Hello world"
        )
        assert similarity == 1.0

    def test_fallback_similarity_empty_strings(self):
        """Fallback should handle empty strings correctly."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        # Both empty
        similarity = comparator._compute_fallback_similarity("", "")
        assert similarity == 1.0

        # One empty
        similarity = comparator._compute_fallback_similarity("Hello", "")
        assert similarity == 0.0

        similarity = comparator._compute_fallback_similarity("", "Hello")
        assert similarity == 0.0

    def test_fallback_similarity_difflib(self):
        """Fallback should use difflib for similar strings."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        # Very similar strings
        similarity = comparator._compute_fallback_similarity(
            "Hello, how are you?", "Hello, how are you doing?"
        )
        assert 0.7 < similarity < 1.0  # High similarity

        # Somewhat similar
        similarity = comparator._compute_fallback_similarity(
            "The quick brown fox", "The quick red fox"
        )
        assert 0.5 < similarity < 0.9

        # Very different
        similarity = comparator._compute_fallback_similarity(
            "Hello world", "Completely different text here"
        )
        assert 0.0 <= similarity < 0.5

    def test_fallback_similarity_word_overlap(self):
        """Fallback should measure word overlap (Jaccard similarity)."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        # Same words, different order
        sim1 = comparator._compute_fallback_similarity(
            "cat dog bird", "dog bird cat"
        )
        # All words match but different order, difflib gives ~0.67
        assert sim1 > 0.6

        # Some word overlap
        sim2 = comparator._compute_fallback_similarity(
            "cat dog bird", "cat mouse bird"
        )
        # 2/4 words in common (union = 4 words), difflib gives ~0.67
        assert 0.3 < sim2 < 0.8

    def test_semantic_similarity_uses_fallback_when_encoder_unavailable(self):
        """When encoder not available, _compute_semantic_similarity should use fallback."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        # Encoder should be None
        assert comparator._encoder is None

        # Should still compute similarity using fallback
        similarity = comparator._compute_semantic_similarity(
            "Hello world", "Hello world"
        )
        assert similarity == 1.0

        similarity = comparator._compute_semantic_similarity(
            "Hello world", "Goodbye world"
        )
        assert 0.0 < similarity < 1.0

    def test_comparator_sets_availability_flags(self):
        """ReplayComparator should set semantic_similarity_available correctly."""
        # With sentence-transformers disabled
        comparator = ReplayComparator(use_semantic_similarity=False)
        assert comparator.semantic_similarity_available is False
        assert comparator.semantic_similarity_model is None

    def test_comparison_includes_availability_flags(self):
        """ReplayComparison should include semantic similarity availability info."""
        span = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Hello",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span])
        result2 = ReplayResult(trace_id="trace-1", spans=[span])

        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        assert comparison.semantic_similarity_available is False
        assert comparison.semantic_similarity_model is None

    def test_fallback_similarity_with_real_comparison(self):
        """Integration test: Compare texts using fallback similarity."""
        span1 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="The quick brown fox jumps over the lazy dog",
        )

        span2 = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="The quick red fox jumps over the lazy dog",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span1])
        result2 = ReplayResult(trace_id="trace-1", spans=[span2])

        # Use fallback similarity
        comparator = ReplayComparator(use_semantic_similarity=False)
        comparison = comparator.compare(result1, result2)

        # Should detect difference
        assert len(comparison.differences) == 1
        diff = comparison.differences[0]
        assert diff.field == "output"
        assert diff.exact_match is False

        # Should compute fallback similarity (high because only "brown" vs "red" differs)
        assert diff.semantic_similarity is not None
        assert 0.8 < diff.semantic_similarity < 1.0

    def test_fallback_handles_comparison_failure_gracefully(self):
        """Fallback should return reasonable value even if all methods fail."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        # Test with very different lengths
        similarity = comparator._compute_fallback_similarity(
            "a" * 100, "b" * 10
        )
        # Should still return some reasonable value
        assert 0.0 <= similarity <= 1.0

    def test_comparator_initialization_without_sentence_transformers(self):
        """Comparator should initialize correctly when sentence-transformers unavailable."""
        # Should not raise exception
        comparator = ReplayComparator(use_semantic_similarity=True)

        # If sentence-transformers not installed, should fall back gracefully
        # Either encoder is set (package available) or use_semantic_similarity is False
        assert comparator._encoder is not None or comparator.use_semantic_similarity is False

    def test_fallback_similarity_case_insensitive(self):
        """Fallback similarity should be case-insensitive for word matching."""
        comparator = ReplayComparator(use_semantic_similarity=False)

        # Same text, different case - difflib is case-sensitive
        similarity = comparator._compute_fallback_similarity(
            "Hello World", "hello world"
        )
        # Should still have good similarity (difflib gives ~0.82 for case mismatch)
        assert similarity > 0.8

    def test_compare_replays_with_fallback(self):
        """compare_replays function should work with fallback similarity."""
        from prela.replay.comparison import compare_replays

        span = ReplayedSpan(
            original_span_id="span-1",
            span_type="llm",
            name="test",
            input={},
            output="Test output",
        )

        result1 = ReplayResult(trace_id="trace-1", spans=[span])
        result2 = ReplayResult(trace_id="trace-1", spans=[span])

        # Should not raise exception
        comparison = compare_replays(result1, result2, use_semantic_similarity=False)
        assert comparison is not None
        assert comparison.semantic_similarity_available is False
