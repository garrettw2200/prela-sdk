"""Result data structures for replay execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayedSpan:
    """Result of replaying a single span.

    Tracks what was executed, whether it was modified, and the output.
    """

    original_span_id: str
    span_type: str
    name: str
    input: Any
    output: Any
    was_modified: bool = False
    modification_details: str | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    retry_count: int = 0  # Number of retry attempts for API calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_span_id": self.original_span_id,
            "span_type": self.span_type,
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "was_modified": self.was_modified,
            "modification_details": self.modification_details,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class ReplayResult:
    """Complete result of replaying a trace.

    Contains all replayed spans, aggregated metrics, and final output.
    """

    trace_id: str
    spans: list[ReplayedSpan] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    final_output: Any = None
    errors: list[str] = field(default_factory=list)
    modifications_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if replay completed without errors."""
        return len(self.errors) == 0 and all(s.error is None for s in self.spans)

    @property
    def modified_span_count(self) -> int:
        """Count how many spans were modified."""
        return sum(1 for s in self.spans if s.was_modified)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "final_output": self.final_output,
            "errors": self.errors,
            "modifications_applied": self.modifications_applied,
            "success": self.success,
            "modified_span_count": self.modified_span_count,
        }


@dataclass
class SpanDifference:
    """Difference between two span executions.

    Captures what changed between original and modified replay.
    """

    span_name: str
    span_type: str
    field: str
    original_value: Any
    modified_value: Any
    semantic_similarity: float | None = None
    exact_match: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "span_name": self.span_name,
            "span_type": self.span_type,
            "field": self.field,
            "original_value": self.original_value,
            "modified_value": self.modified_value,
            "semantic_similarity": self.semantic_similarity,
            "exact_match": self.exact_match,
        }


@dataclass
class ReplayComparison:
    """Comparison between two replay results.

    Highlights differences and provides summary statistics.
    """

    original: ReplayResult
    modified: ReplayResult
    differences: list[SpanDifference] = field(default_factory=list)
    summary: str = ""
    semantic_similarity_available: bool = False  # Whether sentence-transformers is available
    semantic_similarity_model: str | None = None  # Model used for similarity (if available)

    @property
    def identical_spans(self) -> int:
        """Count spans with identical outputs."""
        return len(self.original.spans) - len(self.differences)

    @property
    def changed_spans(self) -> int:
        """Count spans with different outputs."""
        return len(set(d.span_name for d in self.differences))

    @property
    def total_cost_delta(self) -> float:
        """Calculate cost difference."""
        return self.modified.total_cost_usd - self.original.total_cost_usd

    @property
    def total_tokens_delta(self) -> int:
        """Calculate token usage difference."""
        return self.modified.total_tokens - self.original.total_tokens

    def generate_summary(self) -> str:
        """Generate human-readable summary of differences."""
        total_spans = len(self.original.spans)
        changed = self.changed_spans
        identical = self.identical_spans

        # Calculate percentages (handle zero total_spans)
        identical_pct = (identical / total_spans * 100) if total_spans > 0 else 0.0
        changed_pct = (changed / total_spans * 100) if total_spans > 0 else 0.0

        lines = [
            f"Replay Comparison Summary",
            f"=" * 50,
            f"Total Spans: {total_spans}",
            f"Identical: {identical} ({identical_pct:.1f}%)",
            f"Changed: {changed} ({changed_pct:.1f}%)",
            f"",
            f"Cost: ${self.original.total_cost_usd:.4f} → ${self.modified.total_cost_usd:.4f} "
            f"({'+' if self.total_cost_delta > 0 else ''}{self.total_cost_delta:.4f})",
            f"Tokens: {self.original.total_tokens} → {self.modified.total_tokens} "
            f"({'+' if self.total_tokens_delta > 0 else ''}{self.total_tokens_delta})",
        ]

        if self.differences:
            lines.append("")
            lines.append("Key Differences:")
            for diff in self.differences[:5]:  # Show top 5
                lines.append(f"  • {diff.span_name} ({diff.field})")
                if diff.semantic_similarity is not None:
                    lines.append(f"    Similarity: {diff.semantic_similarity:.2%}")

        self.summary = "\n".join(lines)
        return self.summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original": self.original.to_dict(),
            "modified": self.modified.to_dict(),
            "differences": [d.to_dict() for d in self.differences],
            "summary": self.summary or self.generate_summary(),
            "identical_spans": self.identical_spans,
            "changed_spans": self.changed_spans,
            "total_cost_delta": self.total_cost_delta,
            "total_tokens_delta": self.total_tokens_delta,
        }
