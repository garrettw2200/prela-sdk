"""Trace loading utilities for replay engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from prela.core.span import Span

logger = logging.getLogger(__name__)


class Trace:
    """Represents a complete trace with all spans.

    A trace is a collection of spans that form a complete execution tree.
    """

    def __init__(self, trace_id: str, spans: list[Span]) -> None:
        """Initialize a trace.

        Args:
            trace_id: Unique trace identifier
            spans: List of spans in this trace
        """
        self.trace_id = trace_id
        self.spans = spans
        self._build_tree()

    def _build_tree(self) -> None:
        """Build parent-child relationships between spans."""
        # Create span lookup
        self.span_map: dict[str, Span] = {s.span_id: s for s in self.spans}

        # Build children mapping
        self.children: dict[str, list[Span]] = {}
        self.root_spans: list[Span] = []

        for span in self.spans:
            if span.parent_span_id is None:
                self.root_spans.append(span)
            else:
                if span.parent_span_id not in self.children:
                    self.children[span.parent_span_id] = []
                self.children[span.parent_span_id].append(span)

        # Sort children by start time for deterministic execution order
        for children_list in self.children.values():
            children_list.sort(key=lambda s: s.started_at)

        self.root_spans.sort(key=lambda s: s.started_at)

    def get_children(self, span_id: str) -> list[Span]:
        """Get all child spans of a given span.

        Args:
            span_id: Parent span ID

        Returns:
            List of child spans (empty if no children)
        """
        return self.children.get(span_id, [])

    def walk_depth_first(self) -> list[Span]:
        """Walk the trace tree depth-first.

        Returns:
            List of spans in depth-first execution order
        """
        result = []

        def visit(span: Span) -> None:
            result.append(span)
            for child in self.get_children(span.span_id):
                visit(child)

        for root in self.root_spans:
            visit(root)

        return result

    def has_replay_data(self) -> bool:
        """Check if trace has replay snapshots.

        Returns:
            True if at least one span has replay data
        """
        return any(s.replay_snapshot is not None for s in self.spans)

    def validate_replay_completeness(self) -> tuple[bool, list[str]]:
        """Validate that trace has complete replay data.

        Returns:
            Tuple of (is_complete, list of missing span names)
        """
        missing = []
        for span in self.spans:
            if span.replay_snapshot is None:
                missing.append(f"{span.name} ({span.span_id})")

        return len(missing) == 0, missing


class TraceLoader:
    """Loads traces from various sources for replay."""

    @staticmethod
    def from_file(file_path: str | Path) -> Trace:
        """Load trace from a JSON file.

        Args:
            file_path: Path to trace JSON file

        Returns:
            Loaded trace

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {file_path}")

        with open(path) as f:
            data = json.load(f)

        return TraceLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Trace:
        """Load trace from dictionary.

        Args:
            data: Trace data dictionary

        Returns:
            Loaded trace

        Raises:
            ValueError: If data format is invalid
        """
        # Handle both single span and trace array formats
        if isinstance(data, dict) and "trace_id" in data and "spans" in data:
            # Trace format with metadata
            trace_id = data["trace_id"]
            spans_data = data["spans"]
        elif isinstance(data, dict) and "span_id" in data:
            # Single span format
            trace_id = data["trace_id"]
            spans_data = [data]
        elif isinstance(data, list):
            # Array of spans
            if not data:
                raise ValueError("Empty span list")
            trace_id = data[0]["trace_id"]
            spans_data = data
        else:
            raise ValueError("Invalid trace data format")

        # Deserialize spans
        spans = [Span.from_dict(span_data) for span_data in spans_data]

        return Trace(trace_id, spans)

    @staticmethod
    def from_jsonl(file_path: str | Path) -> list[Trace]:
        """Load multiple traces from JSONL file.

        Each line should be a complete trace or span.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of traces

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        # Group spans by trace_id
        traces_data: dict[str, list[dict[str, Any]]] = {}

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                span_data = json.loads(line)
                trace_id = span_data["trace_id"]

                if trace_id not in traces_data:
                    traces_data[trace_id] = []
                traces_data[trace_id].append(span_data)

        # Build traces
        traces = []
        for trace_id, spans_data in traces_data.items():
            spans = [Span.from_dict(span_data) for span_data in spans_data]
            traces.append(Trace(trace_id, spans))

        return traces

    @staticmethod
    def from_span_list(spans: list[Span]) -> Trace:
        """Create trace from list of spans.

        Args:
            spans: List of spans

        Returns:
            Trace

        Raises:
            ValueError: If spans don't share same trace_id
        """
        if not spans:
            raise ValueError("Empty span list")

        trace_id = spans[0].trace_id
        if not all(s.trace_id == trace_id for s in spans):
            raise ValueError("All spans must have the same trace_id")

        return Trace(trace_id, spans)
