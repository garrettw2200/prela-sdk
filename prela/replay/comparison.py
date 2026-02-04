"""Comparison utilities for replay results with semantic similarity."""

from __future__ import annotations

import logging
from typing import Any

from prela.replay.result import ReplayComparison, ReplayResult, SpanDifference

logger = logging.getLogger(__name__)


class ReplayComparator:
    """Compares replay results with semantic similarity analysis.

    Uses sentence-transformers for text similarity and deepdiff for
    structural comparison.
    """

    def __init__(self, use_semantic_similarity: bool = True, show_download_progress: bool = True) -> None:
        """Initialize comparator.

        Args:
            use_semantic_similarity: Whether to compute semantic similarity
                for text fields. Requires sentence-transformers package.
            show_download_progress: Whether to show progress for first-time model download
        """
        self.use_semantic_similarity = use_semantic_similarity
        self.semantic_similarity_available = False
        self.semantic_similarity_model = None
        self._encoder = None

        if use_semantic_similarity:
            try:
                from sentence_transformers import SentenceTransformer

                if show_download_progress:
                    logger.info("Loading semantic similarity model (one-time download ~90MB)...")

                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.semantic_similarity_available = True
                self.semantic_similarity_model = "all-MiniLM-L6-v2"

                if show_download_progress:
                    logger.info("✓ Semantic similarity model loaded successfully")

                logger.debug("Loaded sentence-transformers model for semantic similarity")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available. Using fallback similarity metrics. "
                    "For better comparison, install with: pip install sentence-transformers"
                )
                self.use_semantic_similarity = False
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers model: {e}. Using fallback metrics.")
                self.use_semantic_similarity = False

    def compare(
        self,
        original: ReplayResult,
        modified: ReplayResult,
    ) -> ReplayComparison:
        """Compare two replay results.

        Args:
            original: Original replay result
            modified: Modified replay result

        Returns:
            ReplayComparison with differences and summary
        """
        if original.trace_id != modified.trace_id:
            logger.warning(
                f"Comparing results from different traces: "
                f"{original.trace_id} vs {modified.trace_id}"
            )

        differences = []

        # Compare spans by matching on original_span_id
        original_map = {s.original_span_id: s for s in original.spans}
        modified_map = {s.original_span_id: s for s in modified.spans}

        # Find all span IDs (union of both sets)
        all_span_ids = set(original_map.keys()) | set(modified_map.keys())

        for span_id in all_span_ids:
            orig_span = original_map.get(span_id)
            mod_span = modified_map.get(span_id)

            # Handle missing spans
            if orig_span is None:
                differences.append(
                    SpanDifference(
                        span_name=mod_span.name if mod_span else "unknown",
                        span_type=mod_span.span_type if mod_span else "unknown",
                        field="existence",
                        original_value=None,
                        modified_value=mod_span.to_dict() if mod_span else None,
                        exact_match=False,
                    )
                )
                continue

            if mod_span is None:
                differences.append(
                    SpanDifference(
                        span_name=orig_span.name,
                        span_type=orig_span.span_type,
                        field="existence",
                        original_value=orig_span.to_dict(),
                        modified_value=None,
                        exact_match=False,
                    )
                )
                continue

            # Compare span fields
            span_diffs = self._compare_spans(orig_span, mod_span)
            differences.extend(span_diffs)

        # Create comparison object
        comparison = ReplayComparison(
            original=original,
            modified=modified,
            differences=differences,
            semantic_similarity_available=self.semantic_similarity_available,
            semantic_similarity_model=self.semantic_similarity_model,
        )

        # Generate summary
        comparison.generate_summary()

        return comparison

    def _compare_spans(self, original, modified) -> list[SpanDifference]:
        """Compare two replayed spans.

        Args:
            original: Original replayed span
            modified: Modified replayed span

        Returns:
            List of differences found
        """
        differences = []

        # Compare output (most important field)
        if original.output != modified.output:
            diff = self._compare_values(
                span_name=original.name,
                span_type=original.span_type,
                field="output",
                original_value=original.output,
                modified_value=modified.output,
            )
            if diff:
                differences.append(diff)

        # Compare input
        if original.input != modified.input:
            diff = self._compare_values(
                span_name=original.name,
                span_type=original.span_type,
                field="input",
                original_value=original.input,
                modified_value=modified.input,
            )
            if diff:
                differences.append(diff)

        # Compare duration (if significantly different)
        duration_change = abs(modified.duration_ms - original.duration_ms)
        duration_pct = (
            duration_change / original.duration_ms if original.duration_ms > 0 else 0
        )
        if duration_pct > 0.1:  # More than 10% change
            differences.append(
                SpanDifference(
                    span_name=original.name,
                    span_type=original.span_type,
                    field="duration_ms",
                    original_value=original.duration_ms,
                    modified_value=modified.duration_ms,
                    exact_match=False,
                )
            )

        # Compare tokens
        if original.tokens_used != modified.tokens_used:
            differences.append(
                SpanDifference(
                    span_name=original.name,
                    span_type=original.span_type,
                    field="tokens_used",
                    original_value=original.tokens_used,
                    modified_value=modified.tokens_used,
                    exact_match=False,
                )
            )

        # Compare cost
        if abs(original.cost_usd - modified.cost_usd) > 0.0001:  # $0.0001 threshold
            differences.append(
                SpanDifference(
                    span_name=original.name,
                    span_type=original.span_type,
                    field="cost_usd",
                    original_value=original.cost_usd,
                    modified_value=modified.cost_usd,
                    exact_match=False,
                )
            )

        # Compare error status
        if original.error != modified.error:
            differences.append(
                SpanDifference(
                    span_name=original.name,
                    span_type=original.span_type,
                    field="error",
                    original_value=original.error,
                    modified_value=modified.error,
                    exact_match=False,
                )
            )

        return differences

    def _compare_values(
        self,
        span_name: str,
        span_type: str,
        field: str,
        original_value: Any,
        modified_value: Any,
    ) -> SpanDifference | None:
        """Compare two values with semantic similarity if applicable.

        Args:
            span_name: Name of span being compared
            span_type: Type of span
            field: Field being compared
            original_value: Original value
            modified_value: Modified value

        Returns:
            SpanDifference if values differ, None if identical
        """
        # Check exact match first
        if original_value == modified_value:
            return None

        exact_match = False
        semantic_similarity = None

        # Compute semantic similarity for text fields
        # Always compute if both are strings (will use fallback if encoder unavailable)
        if isinstance(original_value, str) and isinstance(modified_value, str):
            semantic_similarity = self._compute_semantic_similarity(
                original_value, modified_value
            )

        return SpanDifference(
            span_name=span_name,
            span_type=span_type,
            field=field,
            original_value=original_value,
            modified_value=modified_value,
            semantic_similarity=semantic_similarity,
            exact_match=exact_match,
        )

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Uses sentence-transformers if available, falls back to simpler metrics otherwise.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Try sentence-transformers (best quality)
        if self._encoder:
            try:
                # Encode both texts
                embeddings = self._encoder.encode([text1, text2])

                # Compute cosine similarity
                from numpy import dot
                from numpy.linalg import norm

                similarity = dot(embeddings[0], embeddings[1]) / (
                    norm(embeddings[0]) * norm(embeddings[1])
                )

                # Convert to Python float and ensure in [0, 1] range
                return float(max(0.0, min(1.0, similarity)))

            except Exception as e:
                logger.warning(f"Failed to compute semantic similarity: {e}, using fallback")
                # Fall through to fallback methods

        # Fallback: Use difflib sequence matcher
        return self._compute_fallback_similarity(text1, text2)

    def _compute_fallback_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using fallback methods when sentence-transformers unavailable.

        Uses multiple heuristics to provide reasonable similarity estimation.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Exact match
        if text1 == text2:
            return 1.0

        # Both empty
        if not text1 and not text2:
            return 1.0

        # One empty, one not
        if not text1 or not text2:
            return 0.0

        # Method 1: difflib SequenceMatcher (best fallback)
        try:
            import difflib

            return difflib.SequenceMatcher(None, text1, text2).ratio()
        except Exception:
            pass

        # Method 2: Jaccard similarity on words
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0
        except Exception:
            pass

        # Method 3: Character-level Jaccard (last resort)
        try:
            chars1 = set(text1.lower())
            chars2 = set(text2.lower())

            intersection = len(chars1 & chars2)
            union = len(chars1 | chars2)

            return intersection / union if union > 0 else 0.0
        except Exception:
            pass

        # Complete fallback: 0.5 if comparable lengths, 0.0 otherwise
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        return 0.5 if len_ratio > 0.5 else 0.0


def compare_replays(
    original: ReplayResult,
    modified: ReplayResult,
    use_semantic_similarity: bool = True,
) -> ReplayComparison:
    """Convenience function to compare two replay results.

    Args:
        original: Original replay result
        modified: Modified replay result
        use_semantic_similarity: Whether to compute semantic similarity

    Returns:
        ReplayComparison with differences and summary
    """
    comparator = ReplayComparator(use_semantic_similarity=use_semantic_similarity)
    return comparator.compare(original, modified)
