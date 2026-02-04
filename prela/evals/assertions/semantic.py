"""
Semantic assertions using embedding-based similarity.

Requires sentence-transformers library:
    pip install sentence-transformers
"""

from __future__ import annotations

import hashlib
from typing import Any

# Check tier before allowing semantic assertions
from prela.license import check_tier

if not check_tier("Semantic assertions", "lunch-money", silent=False):
    raise ImportError(
        "Semantic assertions require 'lunch-money' subscription or higher. "
        "Upgrade at https://prela.dev/pricing"
    )

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    np = None

from prela.core.span import Span
from prela.evals.assertions.base import AssertionResult, BaseAssertion


class SemanticSimilarityAssertion(BaseAssertion):
    """Assert that output is semantically similar to expected text.

    Uses sentence embeddings to compare semantic meaning rather than exact
    text matching. Useful for evaluating LLM outputs where phrasing varies
    but meaning should be consistent.

    Example:
        >>> assertion = SemanticSimilarityAssertion(
        ...     expected_text="The weather is nice today",
        ...     threshold=0.8
        ... )
        >>> result = assertion.evaluate(
        ...     output="Today has beautiful weather",
        ...     expected=None,
        ...     trace=None
        ... )
        >>> assert result.passed  # High similarity despite different wording

    Requires:
        pip install sentence-transformers
    """

    # Class-level model cache (shared across instances)
    _model_cache: dict[str, Any] = {}

    # Embedding cache (to avoid recomputing for same text)
    _embedding_cache: dict[str, Any] = {}

    def __init__(
        self,
        expected_text: str,
        threshold: float = 0.8,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize semantic similarity assertion.

        Args:
            expected_text: Text to compare against
            threshold: Minimum cosine similarity score (0-1) to pass
            model_name: Sentence transformer model to use
                       (default: all-MiniLM-L6-v2, fast and accurate)

        Raises:
            ImportError: If sentence-transformers is not installed
            ValueError: If threshold is not between 0 and 1
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for semantic similarity. "
                "Install with: pip install sentence-transformers"
            )

        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        self.expected_text = expected_text
        self.threshold = threshold
        self.model_name = model_name

        # Load model (cached at class level)
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)

        self.model = self._model_cache[model_name]

    def _get_embedding(self, text: str) -> Any:
        """Get embedding for text, using cache if available."""
        # Create cache key from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"{self.model_name}:{text_hash}"

        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return self._embedding_cache[cache_key]

    def _cosine_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """Compute cosine similarity between two embeddings."""
        # Embeddings are already normalized, so dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if output is semantically similar to expected text."""
        output_str = str(output)

        # Get embeddings
        try:
            output_embedding = self._get_embedding(output_str)
            expected_embedding = self._get_embedding(self.expected_text)
        except Exception as e:
            return AssertionResult(
                passed=False,
                assertion_type="semantic_similarity",
                message=f"Failed to compute embeddings: {e}",
                expected=self.expected_text,
                actual=output_str[:100] + "..." if len(output_str) > 100 else output_str,
                details={"error": str(e)},
            )

        # Compute similarity
        similarity = self._cosine_similarity(output_embedding, expected_embedding)
        passed = similarity >= self.threshold

        # Determine score interpretation
        if similarity >= 0.9:
            interpretation = "very high"
        elif similarity >= 0.8:
            interpretation = "high"
        elif similarity >= 0.7:
            interpretation = "moderate"
        elif similarity >= 0.6:
            interpretation = "low"
        else:
            interpretation = "very low"

        if passed:
            message = (
                f"Output is semantically similar to expected "
                f"(similarity: {similarity:.3f}, {interpretation})"
            )
        else:
            message = (
                f"Output is not semantically similar enough "
                f"(similarity: {similarity:.3f} < threshold: {self.threshold}, {interpretation})"
            )

        return AssertionResult(
            passed=passed,
            assertion_type="semantic_similarity",
            message=message,
            score=similarity,
            expected=self.expected_text,
            actual=output_str[:100] + "..." if len(output_str) > 100 else output_str,
            details={
                "similarity": similarity,
                "threshold": self.threshold,
                "interpretation": interpretation,
                "model": self.model_name,
            },
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SemanticSimilarityAssertion:
        """Create from configuration.

        Config format:
            {
                "expected_text": "The expected output",
                "threshold": 0.8,  # optional, default: 0.8
                "model_name": "all-MiniLM-L6-v2"  # optional
            }
        """
        if "expected_text" not in config:
            raise ValueError("SemanticSimilarityAssertion requires 'expected_text' in config")

        return cls(
            expected_text=config["expected_text"],
            threshold=config.get("threshold", 0.8),
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
        )

    def __repr__(self) -> str:
        return (
            f"SemanticSimilarityAssertion("
            f"expected_text={self.expected_text[:30]!r}..., "
            f"threshold={self.threshold}, "
            f"model_name={self.model_name!r})"
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the embedding cache. Useful for testing or memory management."""
        cls._embedding_cache.clear()

    @classmethod
    def get_cache_size(cls) -> int:
        """Get the number of cached embeddings."""
        return len(cls._embedding_cache)
