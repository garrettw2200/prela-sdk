"""Console exporter for pretty-printing spans to stdout."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

from prela.core.span import Span, SpanStatus, SpanType
from prela.exporters.base import BaseExporter, ExportResult

# Try to import rich for colored output
try:
    from rich.console import Console
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConsoleExporter(BaseExporter):
    """
    Export spans to console with pretty-printed tree visualization.

    Features:
    - Tree structure showing parent-child relationships
    - Color-coded output (when rich library is available)
    - Multiple verbosity levels (minimal, normal, verbose)
    - Duration and status indicators
    - Key attribute display

    Example:
        ```python
        from prela.core.tracer import Tracer
        from prela.exporters.console import ConsoleExporter

        tracer = Tracer(
            service_name="my-agent",
            exporter=ConsoleExporter(
                verbosity="normal",
                color=True,
                show_timestamps=True
            )
        )

        with tracer.span("research_agent", span_type="agent") as span:
            with tracer.span("gpt-4", span_type="llm") as llm_span:
                llm_span.set_attribute("llm.model", "gpt-4")
                llm_span.set_attribute("llm.input_tokens", 150)
                llm_span.set_attribute("llm.output_tokens", 89)
        # Output:
        # ─ agent: research_agent (1.523s) ✓
        #   └─ llm: gpt-4 (823ms) ✓
        #      model: gpt-4 | tokens: 150 → 89
        ```
    """

    def __init__(
        self,
        verbosity: str = "normal",
        color: bool = True,
        show_timestamps: bool = True,
    ):
        """
        Initialize console exporter.

        Args:
            verbosity: Output verbosity level:
                - "minimal": name + duration + status only
                - "normal": + key attributes (model, tokens, query)
                - "verbose": + all attributes + events
            color: Enable colored output (requires rich library)
            show_timestamps: Show timestamps in output
        """
        if verbosity not in ("minimal", "normal", "verbose"):
            raise ValueError(
                f"Invalid verbosity: {verbosity}. "
                "Must be 'minimal', 'normal', or 'verbose'"
            )

        self.verbosity = verbosity
        self.color = color and RICH_AVAILABLE
        self.show_timestamps = show_timestamps

        if self.color:
            self.console = Console(file=sys.stdout)

    def export(self, spans: list[Span]) -> ExportResult:
        """
        Export spans to console with tree visualization.

        Args:
            spans: List of spans to export

        Returns:
            ExportResult.SUCCESS (console export never fails)
        """
        if not spans:
            return ExportResult.SUCCESS

        # Group spans by trace_id
        traces = defaultdict(list)
        for span in spans:
            traces[span.trace_id].append(span)

        # Print each trace
        for trace_id, trace_spans in traces.items():
            self._print_trace(trace_id, trace_spans)

        return ExportResult.SUCCESS

    def _print_trace(self, trace_id: str, spans: list[Span]) -> None:
        """
        Print a single trace as a tree structure.

        Args:
            trace_id: The trace ID
            spans: List of spans in this trace
        """
        # Build parent → children mapping
        children_map = defaultdict(list)
        span_map = {span.span_id: span for span in spans}

        for span in spans:
            parent_id = span.parent_span_id or "root"
            children_map[parent_id].append(span)

        # Find root spans
        root_spans = children_map["root"]

        if not root_spans:
            # No root spans found, treat first span as root
            root_spans = [spans[0]]

        # Print trace header
        if self.show_timestamps:
            timestamp = root_spans[0].started_at.strftime("%H:%M:%S.%f")[:-3]
            header = f"Trace {trace_id[:8]} @ {timestamp}"
        else:
            header = f"Trace {trace_id[:8]}"

        if self.color:
            self.console.print(f"\n[bold cyan]{header}[/bold cyan]")
        else:
            print(f"\n{header}")
            print("=" * len(header))

        # Print tree
        if self.color:
            for root_span in root_spans:
                tree = self._build_rich_tree(root_span, children_map)
                self.console.print(tree)
        else:
            for root_span in root_spans:
                self._print_plain_tree(root_span, children_map, prefix="", is_last=True)

    def _build_rich_tree(
        self, span: Span, children_map: dict[str, list[Span]]
    ) -> Tree:
        """
        Build a rich Tree node for a span.

        Args:
            span: The span to build a tree for
            children_map: Mapping of parent span IDs to children

        Returns:
            Rich Tree node
        """
        # Build label
        label = self._format_span_label(span, use_color=True)

        # Create tree node
        tree = Tree(label)

        # Add attributes
        if self.verbosity in ("normal", "verbose"):
            attr_str = self._format_attributes(span, use_color=True)
            if attr_str:
                tree.add(attr_str)

        # Add events (verbose only)
        if self.verbosity == "verbose" and span.events:
            events_tree = tree.add("[dim]Events:[/dim]")
            for event in span.events:
                event_time = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
                event_label = f"[dim]{event_time}[/dim] {event.name}"
                if event.attributes:
                    attr_list = [
                        f"{k}={v}" for k, v in sorted(event.attributes.items())
                    ]
                    event_label += f" [dim]({', '.join(attr_list)})[/dim]"
                events_tree.add(event_label)

        # Add children (exclude self to prevent infinite recursion)
        children = [
            c for c in children_map.get(span.span_id, []) if c.span_id != span.span_id
        ]
        for child in sorted(children, key=lambda s: s.started_at):
            child_tree = self._build_rich_tree(child, children_map)
            tree.add(child_tree)

        return tree

    def _print_plain_tree(
        self,
        span: Span,
        children_map: dict[str, list[Span]],
        prefix: str,
        is_last: bool,
    ) -> None:
        """
        Print span tree in plain text format.

        Args:
            span: The span to print
            children_map: Mapping of parent span IDs to children
            prefix: Prefix for tree indentation
            is_last: Whether this is the last child
        """
        # Build connector
        connector = "└─ " if is_last else "├─ "

        # Print span
        label = self._format_span_label(span, use_color=False)
        print(f"{prefix}{connector}{label}")

        # Print attributes
        if self.verbosity in ("normal", "verbose"):
            attr_str = self._format_attributes(span, use_color=False)
            if attr_str:
                child_prefix = prefix + ("   " if is_last else "│  ")
                print(f"{child_prefix}   {attr_str}")

        # Print events (verbose only)
        if self.verbosity == "verbose" and span.events:
            child_prefix = prefix + ("   " if is_last else "│  ")
            print(f"{child_prefix}   Events:")
            for event in span.events:
                event_time = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
                event_label = f"{event_time} {event.name}"
                if event.attributes:
                    attr_list = [
                        f"{k}={v}" for k, v in sorted(event.attributes.items())
                    ]
                    event_label += f" ({', '.join(attr_list)})"
                print(f"{child_prefix}     - {event_label}")

        # Print children (exclude self to prevent infinite recursion)
        children = [
            c for c in children_map.get(span.span_id, []) if c.span_id != span.span_id
        ]
        sorted_children = sorted(children, key=lambda s: s.started_at)
        for i, child in enumerate(sorted_children):
            child_is_last = i == len(sorted_children) - 1
            child_prefix = prefix + ("   " if is_last else "│  ")
            self._print_plain_tree(child, children_map, child_prefix, child_is_last)

    def _format_span_label(self, span: Span, use_color: bool) -> str:
        """
        Format a span label with type, name, duration, and status.

        Args:
            span: The span to format
            use_color: Whether to use color codes

        Returns:
            Formatted label string
        """
        # Span type
        span_type = span.span_type.value
        if use_color:
            type_color = self._get_type_color(span.span_type)
            type_str = f"[{type_color}]{span_type}[/{type_color}]"
        else:
            type_str = span_type

        # Duration
        duration = self._format_duration(span)

        # Status indicator
        if span.status == SpanStatus.SUCCESS:
            status_indicator = "[green]✓[/green]" if use_color else "✓"
        elif span.status == SpanStatus.ERROR:
            status_indicator = "[red]✗[/red]" if use_color else "✗"
        else:
            status_indicator = "[yellow]⋯[/yellow]" if use_color else "⋯"

        # Build label
        label = f"{type_str}: {span.name} ({duration}) {status_indicator}"

        return label

    def _format_attributes(self, span: Span, use_color: bool) -> str:
        """
        Format span attributes based on verbosity level.

        Args:
            span: The span to format attributes for
            use_color: Whether to use color codes

        Returns:
            Formatted attributes string
        """
        if self.verbosity == "verbose":
            # Show all attributes
            if not span.attributes:
                return ""
            attr_list = []
            for key, value in sorted(span.attributes.items()):
                if use_color:
                    attr_list.append(f"[dim]{key}[/dim] = [cyan]{value!r}[/cyan]")
                else:
                    attr_list.append(f"{key} = {value!r}")
            return "\n".join(attr_list)
        else:
            # Show key attributes only (including error messages)
            key_attrs = self._extract_key_attributes(span)
            if not key_attrs:
                return ""

            parts = []
            for key, value in key_attrs.items():
                if use_color:
                    parts.append(f"[dim]{key}:[/dim] [cyan]{value}[/cyan]")
                else:
                    parts.append(f"{key}: {value}")

            return " | ".join(parts)

    def _extract_key_attributes(self, span: Span) -> dict[str, Any]:
        """
        Extract key attributes to display based on span type.

        Args:
            span: The span to extract attributes from

        Returns:
            Dictionary of key attributes
        """
        attrs = span.attributes
        key_attrs = {}

        if span.span_type == SpanType.LLM:
            # LLM: model, tokens
            if "llm.model" in attrs:
                key_attrs["model"] = attrs["llm.model"]
            if "llm.input_tokens" in attrs and "llm.output_tokens" in attrs:
                key_attrs["tokens"] = (
                    f"{attrs['llm.input_tokens']} → {attrs['llm.output_tokens']}"
                )
            elif "llm.prompt_tokens" in attrs and "llm.completion_tokens" in attrs:
                key_attrs["tokens"] = (
                    f"{attrs['llm.prompt_tokens']} → {attrs['llm.completion_tokens']}"
                )

        elif span.span_type == SpanType.TOOL:
            # Tool: tool name, query/input
            if "tool.name" in attrs:
                key_attrs["tool"] = attrs["tool.name"]
            if "tool.input" in attrs:
                tool_input = str(attrs["tool.input"])
                if len(tool_input) > 50:
                    tool_input = tool_input[:50] + "..."
                key_attrs["input"] = tool_input

        elif span.span_type == SpanType.RETRIEVAL:
            # Retrieval: query, document count
            if "retriever.query" in attrs:
                query = str(attrs["retriever.query"])
                if len(query) > 50:
                    query = query[:50] + "..."
                key_attrs["query"] = query
            if "retriever.document_count" in attrs:
                key_attrs["docs"] = attrs["retriever.document_count"]

        elif span.span_type == SpanType.EMBEDDING:
            # Embedding: model, dimensions
            if "embedding.model" in attrs:
                key_attrs["model"] = attrs["embedding.model"]
            if "embedding.dimensions" in attrs:
                key_attrs["dims"] = attrs["embedding.dimensions"]

        # Show error message if present
        if span.status == SpanStatus.ERROR and span.status_message:
            error_msg = span.status_message
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            key_attrs["error"] = error_msg

        return key_attrs

    def _format_duration(self, span: Span) -> str:
        """
        Format span duration in human-readable format.

        Args:
            span: The span to format duration for

        Returns:
            Formatted duration string (e.g., "1.523s", "823ms")
        """
        if not span.ended_at:
            return "running"

        duration = (span.ended_at - span.started_at).total_seconds()

        if duration < 0.001:
            return f"{duration * 1_000_000:.0f}µs"
        elif duration < 1.0:
            return f"{duration * 1000:.0f}ms"
        else:
            return f"{duration:.3f}s"

    def _get_type_color(self, span_type: SpanType) -> str:
        """
        Get color for span type.

        Args:
            span_type: The span type

        Returns:
            Rich color name
        """
        color_map = {
            SpanType.AGENT: "yellow",
            SpanType.LLM: "magenta",
            SpanType.TOOL: "blue",
            SpanType.RETRIEVAL: "green",
            SpanType.EMBEDDING: "cyan",
            SpanType.CUSTOM: "white",
        }
        return color_map.get(span_type, "white")

    def shutdown(self) -> None:
        """
        Shutdown the exporter.

        No cleanup needed for console exporter.
        """
        pass
