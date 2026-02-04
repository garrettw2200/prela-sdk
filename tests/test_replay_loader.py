"""Tests for replay trace loader."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from prela.core.replay import ReplaySnapshot
from prela.core.span import Span, SpanStatus, SpanType
from prela.replay.loader import Trace, TraceLoader


class TestTrace:
    """Test Trace class."""

    def test_init_builds_tree(self):
        """Trace initialization builds span tree."""
        span1 = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="root",
            span_type=SpanType.AGENT,
        )
        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            name="child",
            span_type=SpanType.LLM,
        )

        trace = Trace("trace-1", [span1, span2])

        assert trace.trace_id == "trace-1"
        assert len(trace.spans) == 2
        assert len(trace.root_spans) == 1
        assert trace.root_spans[0] == span1

    def test_get_children(self):
        """get_children returns child spans."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="root", span_type=SpanType.AGENT
        )
        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            name="child1",
            span_type=SpanType.LLM,
        )
        span3 = Span(
            trace_id="trace-1",
            span_id="span-3",
            parent_span_id="span-1",
            name="child2",
            span_type=SpanType.TOOL,
        )

        trace = Trace("trace-1", [span1, span2, span3])

        children = trace.get_children("span-1")
        assert len(children) == 2
        assert span2 in children
        assert span3 in children

    def test_walk_depth_first(self):
        """walk_depth_first returns spans in execution order."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="root", span_type=SpanType.AGENT
        )
        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            name="child1",
            span_type=SpanType.LLM,
        )
        span3 = Span(
            trace_id="trace-1",
            span_id="span-3",
            parent_span_id="span-2",
            name="grandchild",
            span_type=SpanType.TOOL,
        )
        span4 = Span(
            trace_id="trace-1",
            span_id="span-4",
            parent_span_id="span-1",
            name="child2",
            span_type=SpanType.LLM,
        )

        trace = Trace("trace-1", [span1, span2, span3, span4])

        spans = trace.walk_depth_first()
        assert len(spans) == 4
        assert spans[0] == span1  # root
        assert spans[1] == span2  # child1
        assert spans[2] == span3  # grandchild
        assert spans[3] == span4  # child2

    def test_has_replay_data_true(self):
        """has_replay_data returns True if any span has replay_snapshot."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="root", span_type=SpanType.AGENT
        )
        span1.replay_snapshot = ReplaySnapshot(tool_name="test")

        trace = Trace("trace-1", [span1])
        assert trace.has_replay_data() is True

    def test_has_replay_data_false(self):
        """has_replay_data returns False if no spans have replay_snapshot."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="root", span_type=SpanType.AGENT
        )

        trace = Trace("trace-1", [span1])
        assert trace.has_replay_data() is False

    def test_validate_replay_completeness(self):
        """validate_replay_completeness checks all spans have replay data."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="root", span_type=SpanType.AGENT
        )
        span1.replay_snapshot = ReplaySnapshot(tool_name="test1")

        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            name="child",
            span_type=SpanType.LLM,
        )
        # span2 has no replay_snapshot

        trace = Trace("trace-1", [span1, span2])

        is_complete, missing = trace.validate_replay_completeness()
        assert is_complete is False
        assert len(missing) == 1
        assert "child (span-2)" in missing


class TestTraceLoader:
    """Test TraceLoader class."""

    def test_from_dict_trace_format(self):
        """from_dict loads trace with metadata."""
        data = {
            "trace_id": "trace-1",
            "spans": [
                {
                    "trace_id": "trace-1",
                    "span_id": "span-1",
                    "name": "test",
                    "span_type": "agent",
                    "started_at": "2024-01-01T00:00:00Z",
                }
            ],
        }

        trace = TraceLoader.from_dict(data)
        assert trace.trace_id == "trace-1"
        assert len(trace.spans) == 1

    def test_from_dict_single_span(self):
        """from_dict loads single span format."""
        data = {
            "trace_id": "trace-1",
            "span_id": "span-1",
            "name": "test",
            "span_type": "agent",
            "started_at": "2024-01-01T00:00:00Z",
        }

        trace = TraceLoader.from_dict(data)
        assert trace.trace_id == "trace-1"
        assert len(trace.spans) == 1

    def test_from_dict_array_format(self):
        """from_dict loads array of spans."""
        data = [
            {
                "trace_id": "trace-1",
                "span_id": "span-1",
                "name": "test1",
                "span_type": "agent",
                "started_at": "2024-01-01T00:00:00Z",
            },
            {
                "trace_id": "trace-1",
                "span_id": "span-2",
                "name": "test2",
                "span_type": "llm",
                "started_at": "2024-01-01T00:00:01Z",
            },
        ]

        trace = TraceLoader.from_dict(data)
        assert trace.trace_id == "trace-1"
        assert len(trace.spans) == 2

    def test_from_dict_invalid_format(self):
        """from_dict raises ValueError for invalid data."""
        with pytest.raises(ValueError, match="Invalid trace data format"):
            TraceLoader.from_dict({})

    def test_from_dict_empty_array(self):
        """from_dict raises ValueError for empty array."""
        with pytest.raises(ValueError, match="Empty span list"):
            TraceLoader.from_dict([])

    def test_from_file_json(self):
        """from_file loads trace from JSON file."""
        with TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "trace.json"

            data = {
                "trace_id": "trace-1",
                "spans": [
                    {
                        "trace_id": "trace-1",
                        "span_id": "span-1",
                        "name": "test",
                        "span_type": "agent",
                        "started_at": "2024-01-01T00:00:00Z",
                    }
                ],
            }

            with open(trace_file, "w") as f:
                json.dump(data, f)

            trace = TraceLoader.from_file(trace_file)
            assert trace.trace_id == "trace-1"
            assert len(trace.spans) == 1

    def test_from_file_not_found(self):
        """from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            TraceLoader.from_file("/nonexistent/file.json")

    def test_from_jsonl(self):
        """from_jsonl loads multiple traces from JSONL file."""
        with TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "traces.jsonl"

            with open(jsonl_file, "w") as f:
                # Write two traces
                f.write(
                    json.dumps(
                        {
                            "trace_id": "trace-1",
                            "span_id": "span-1",
                            "name": "test1",
                            "span_type": "agent",
                            "started_at": "2024-01-01T00:00:00Z",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "trace_id": "trace-2",
                            "span_id": "span-2",
                            "name": "test2",
                            "span_type": "agent",
                            "started_at": "2024-01-01T00:00:00Z",
                        }
                    )
                    + "\n"
                )

            traces = TraceLoader.from_jsonl(jsonl_file)
            assert len(traces) == 2
            assert traces[0].trace_id == "trace-1"
            assert traces[1].trace_id == "trace-2"

    def test_from_jsonl_grouped_spans(self):
        """from_jsonl groups spans by trace_id."""
        with TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "traces.jsonl"

            with open(jsonl_file, "w") as f:
                # Write two spans from same trace
                f.write(
                    json.dumps(
                        {
                            "trace_id": "trace-1",
                            "span_id": "span-1",
                            "name": "test1",
                            "span_type": "agent",
                            "started_at": "2024-01-01T00:00:00Z",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "trace_id": "trace-1",
                            "span_id": "span-2",
                            "parent_span_id": "span-1",
                            "name": "test2",
                            "span_type": "llm",
                            "started_at": "2024-01-01T00:00:01Z",
                        }
                    )
                    + "\n"
                )

            traces = TraceLoader.from_jsonl(jsonl_file)
            assert len(traces) == 1
            assert traces[0].trace_id == "trace-1"
            assert len(traces[0].spans) == 2

    def test_from_span_list(self):
        """from_span_list creates trace from span list."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="test", span_type=SpanType.AGENT
        )
        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            name="child",
            span_type=SpanType.LLM,
        )

        trace = TraceLoader.from_span_list([span1, span2])
        assert trace.trace_id == "trace-1"
        assert len(trace.spans) == 2

    def test_from_span_list_empty(self):
        """from_span_list raises ValueError for empty list."""
        with pytest.raises(ValueError, match="Empty span list"):
            TraceLoader.from_span_list([])

    def test_from_span_list_different_trace_ids(self):
        """from_span_list raises ValueError for mixed trace_ids."""
        span1 = Span(
            trace_id="trace-1", span_id="span-1", name="test", span_type=SpanType.AGENT
        )
        span2 = Span(
            trace_id="trace-2", span_id="span-2", name="test", span_type=SpanType.LLM
        )

        with pytest.raises(ValueError, match="same trace_id"):
            TraceLoader.from_span_list([span1, span2])
