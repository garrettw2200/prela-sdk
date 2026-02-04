"""Interactive trace explorer using Textual TUI framework.

This module provides an interactive terminal user interface for browsing
traces, navigating span hierarchies, and inspecting span details without
needing to copy/paste trace IDs.

Usage:
    $ prela explore
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

# Import from existing cli module
from prela.contrib.cli import (
    build_span_tree,
    find_root_span,
    group_spans_by_trace,
    load_config,
    load_traces_from_file,
)


class TraceExplorer(App):
    """Interactive trace explorer with keyboard navigation.

    Features:
    - Trace list view (default)
    - Trace detail view (expandable span tree)
    - Span detail view (attributes and events)

    Keyboard shortcuts:
    - ↑/k: Move up
    - ↓/j: Move down
    - Enter: Select/drill down
    - Esc: Go back
    - q: Quit
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #header {
        dock: top;
        height: 3;
        background: $boost;
        color: $text;
        content-align: center middle;
        text-style: bold;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    #content {
        height: 1fr;
    }

    DataTable {
        height: 100%;
    }

    Tree {
        height: 100%;
        background: $surface;
    }

    ScrollableContainer {
        height: 1fr;
    }

    VerticalScroll {
        height: 100%;
        background: $surface;
        padding: 1;
    }

    .span-detail-section {
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border: solid $primary;
    }

    .span-detail-label {
        text-style: bold;
        color: $accent;
    }

    .loading {
        padding: 2;
        text-align: center;
        color: $accent;
    }

    .error-message {
        padding: 2;
        background: $error;
        color: $text;
        border: heavy $error;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("up,k", "cursor_up", "Up"),
        ("down,j", "cursor_down", "Down"),
        ("enter", "select_item", "Select"),
        ("escape", "go_back", "Back"),
        ("q", "quit", "Quit"),
        ("?", "show_help", "Help"),
    ]

    def __init__(self, trace_dir: Path):
        """Initialize trace explorer.

        Args:
            trace_dir: Directory containing trace JSONL files
        """
        super().__init__()
        self.trace_dir = trace_dir
        self.traces_data: dict[str, list[dict[str, Any]]] = {}
        self.view_stack: list[str] = []  # Stack for back navigation
        self.current_trace_id: Optional[str] = None
        self.current_span: Optional[dict[str, Any]] = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Static("Loading traces...", id="header")
        yield Container(
            DataTable(id="trace-table", zebra_stripes=True),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Load traces when app starts."""
        header = self.query_one("#header", Static)
        header.update("Prela Trace Explorer")

        # Load traces from directory
        self._load_and_display_traces()

    def _load_and_display_traces(self) -> None:
        """Load traces from file system and display in table."""
        try:
            table = self.query_one("#trace-table", DataTable)

            # Clear existing data
            table.clear(columns=True)

            # Show loading state for large directories
            header = self.query_one("#header", Static)
            header.update("Loading traces...")

            # Load spans from file
            try:
                spans = load_traces_from_file(self.trace_dir)
            except Exception as e:
                header.update(f"Error loading traces: {str(e)[:50]}")
                return

            if not spans:
                # No traces found - show empty state
                header.update("No traces found - Try 'prela init' to configure")
                return

            # Group spans by trace_id
            self.traces_data = group_spans_by_trace(spans)

            # Set up table columns
            table.add_columns("Trace ID", "Name", "Status", "Duration", "Spans", "Time")

            # Add rows for each trace
            trace_summaries = []
            for trace_id, trace_spans in self.traces_data.items():
                root_span = find_root_span(trace_spans)
                if root_span:
                    trace_summaries.append(
                        {
                            "trace_id": trace_id,
                            "root_span": root_span.get("name", "unknown"),
                            "duration_ms": root_span.get("duration_ms", 0),
                            "status": root_span.get("status", "unknown"),
                            "started_at": root_span.get("started_at", ""),
                            "span_count": len(trace_spans),
                        }
                    )

            # Sort by time (most recent first)
            trace_summaries.sort(key=lambda x: x["started_at"], reverse=True)

            # Add rows to table
            for summary in trace_summaries:
                # Format duration
                duration_ms = summary["duration_ms"]
                if duration_ms > 1000:
                    duration_str = f"{duration_ms / 1000:.2f}s"
                else:
                    duration_str = f"{duration_ms:.0f}ms"

                # Format time
                try:
                    started_at = datetime.fromisoformat(summary["started_at"])
                    time_str = started_at.strftime("%H:%M:%S")
                except Exception:
                    time_str = summary["started_at"][:8] if summary["started_at"] else ""

                # Truncate trace ID for display
                trace_id_display = summary["trace_id"][:16]

                table.add_row(
                    trace_id_display,
                    summary["root_span"],
                    summary["status"],
                    duration_str,
                    str(summary["span_count"]),
                    time_str,
                    key=summary["trace_id"],  # Full ID as key for lookup
                )

        # Update header with count
            header = self.query_one("#header", Static)
            trace_count = len(trace_summaries)
            if trace_count == 0:
                header.update("No valid traces found")
            else:
                header.update(f"Traces ({trace_count} found)")
        except Exception as e:
            # Handle any unexpected errors gracefully
            header = self.query_one("#header", Static)
            header.update(f"Error: {str(e)[:60]}")
            import traceback
            traceback.print_exc()

    def action_select_item(self) -> None:
        """Handle Enter key - drill into trace or span."""
        # Check if we're in trace list view
        try:
            table = self.query_one("#trace-table", DataTable)
            # Get selected row key (which is the full trace_id)
            if table.cursor_row is None:
                return

            row_key = table.get_row_key_at(table.cursor_row)
            if row_key is None:
                return

            # Convert RowKey to trace_id string
            trace_id = str(row_key)

            # Show trace detail view
            self._show_trace_detail(trace_id)
        except Exception:
            # Not in trace list, check if we're in tree view
            try:
                tree = self.query_one("#span-tree", Tree)
                # Get selected node
                if tree.cursor_node is None:
                    return

                # Get span data from node
                span_data = tree.cursor_node.data
                if span_data:
                    self._show_span_detail(span_data)
            except Exception:
                # Not in tree view either, ignore
                pass

    def _show_trace_detail(self, trace_id: str) -> None:
        """Show detailed view of a single trace.

        Args:
            trace_id: Trace ID to display
        """
        # Update header
        header = self.query_one("#header", Static)
        header.update(f"Trace Detail: {trace_id[:16]} (Press Esc to go back)")

        # Store current view in stack
        self.view_stack.append("list")
        self.current_trace_id = trace_id

        # Get trace spans
        trace_spans = self.traces_data.get(trace_id, [])
        if not trace_spans:
            return

        # Remove the table and add tree
        container = self.query_one("#content", Container)
        container.remove_children()

        # Build span tree using existing cli function
        span_tree = build_span_tree(trace_spans)

        # Create Tree widget
        tree = Tree(f"Trace: {trace_id[:16]}", id="span-tree")
        tree.root.expand()

        # Populate tree with spans
        span_count = len(trace_spans)
        self._add_spans_to_tree(tree.root, span_tree, max_depth=0 if span_count > 50 else None)

        # Add tree to container
        container.mount(tree)

        # Update header with span count info
        if span_count > 50:
            header.update(
                f"Trace: {trace_id[:16]} ({span_count} spans - Use arrows to expand nodes)"
            )

    def _add_spans_to_tree(
        self,
        parent_node: TreeNode,
        spans: list[dict[str, Any]],
        max_depth: int | None = None,
        current_depth: int = 0,
    ) -> None:
        """Recursively add spans to tree widget.

        Args:
            parent_node: Parent tree node
            spans: List of span dicts with 'span' and 'children' keys
            max_depth: Maximum depth to auto-expand (None = unlimited)
            current_depth: Current depth in tree (for tracking)
        """
        for span_data in spans:
            span = span_data["span"]
            children = span_data.get("children", [])

            # Format span info
            name = span.get("name", "unknown")
            duration_ms = span.get("duration_ms", 0)
            status = span.get("status", "unknown")

            # Format duration
            if duration_ms > 1000:
                duration_str = f"{duration_ms / 1000:.2f}s"
            else:
                duration_str = f"{duration_ms:.0f}ms"

            # Create node label with status emoji
            status_icon = "✓" if status == "success" else "✗" if status == "error" else "○"
            label = f"{status_icon} {name} ({duration_str})"

            # Add node
            node = parent_node.add(label, data=span)

            # Recursively add children
            if children:
                self._add_spans_to_tree(
                    node, children, max_depth=max_depth, current_depth=current_depth + 1
                )

                # Collapse nodes beyond max_depth to improve performance
                if max_depth is not None and current_depth >= max_depth:
                    node.collapse()

    def _show_span_detail(self, span: dict[str, Any]) -> None:
        """Show detailed view of a single span.

        Args:
            span: Span dictionary with all attributes
        """
        # Update header
        span_name = span.get("name", "unknown")
        header = self.query_one("#header", Static)
        header.update(f"Span: {span_name} (Press Esc to go back)")

        # Store current view in stack
        self.view_stack.append("tree")
        self.current_span = span

        # Remove the tree and add span detail view
        container = self.query_one("#content", Container)
        container.remove_children()

        # Create scrollable container for span details
        scroll = VerticalScroll()

        # Add span information sections
        sections = []

        # Basic info section
        basic_info = f"""[span-detail-label]Span Information[/span-detail-label]
Name: {span.get('name', 'unknown')}
Type: {span.get('span_type', 'unknown')}
Status: {span.get('status', 'unknown')}
Duration: {self._format_duration(span.get('duration_ms', 0))}
Started: {span.get('started_at', 'unknown')}
Ended: {span.get('ended_at', 'unknown')}"""
        sections.append(Static(basic_info, classes="span-detail-section"))

        # Attributes section
        attributes = span.get("attributes", {})
        if attributes:
            attrs_json = json.dumps(attributes, indent=2, default=str)
            attrs_text = f"""[span-detail-label]Attributes[/span-detail-label]
{attrs_json}"""
            sections.append(Static(attrs_text, classes="span-detail-section"))

        # Events section
        events = span.get("events", [])
        if events:
            events_text = "[span-detail-label]Events[/span-detail-label]\n"
            for idx, event in enumerate(events, 1):
                event_name = event.get("name", "unknown")
                event_time = event.get("timestamp", "unknown")
                events_text += f"\n{idx}. {event_name} at {event_time}"
                event_attrs = event.get("attributes", {})
                if event_attrs:
                    events_text += f"\n   {json.dumps(event_attrs, indent=2, default=str)}"
            sections.append(Static(events_text, classes="span-detail-section"))

        # Add all sections to scroll container
        for section in sections:
            scroll.mount(section)

        # Add scroll container to main container
        container.mount(scroll)

    def _format_duration(self, duration_ms: float) -> str:
        """Format duration in ms or seconds.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Formatted duration string
        """
        if duration_ms > 1000:
            return f"{duration_ms / 1000:.2f}s"
        return f"{duration_ms:.0f}ms"

    def action_show_help(self) -> None:
        """Show help screen with keyboard shortcuts."""
        # Store current view
        help_text = """
[bold cyan]Prela Trace Explorer - Keyboard Shortcuts[/bold cyan]

[bold]Navigation:[/bold]
  ↑ / k       Move up
  ↓ / j       Move down
  Enter       Select item / Drill down
  Esc         Go back to previous view

[bold]Actions:[/bold]
  q           Quit application
  ?           Show this help

[bold]Views:[/bold]
  1. Trace List    Browse all traces
  2. Trace Detail  View span hierarchy (tree)
  3. Span Detail   View span attributes & events

[bold cyan]Press any key to return[/bold cyan]
        """

        # Update header
        header = self.query_one("#header", Static)
        header.update("Help - Press any key to return")

        # Replace content with help text
        container = self.query_one("#content", Container)
        container.remove_children()

        help_display = Static(help_text, classes="loading")
        container.mount(help_display)

        # Store in view stack so Esc will work
        self.view_stack.append("help")

    def action_go_back(self) -> None:
        """Handle Escape key - go back to previous view."""
        if not self.view_stack:
            # Already at top level - do nothing
            return

        # Pop previous view from stack
        previous_view = self.view_stack.pop()

        if previous_view == "help":
            # Go back from help to trace list
            container = self.query_one("#content", Container)
            container.remove_children()

            # Re-create table
            table = DataTable(id="trace-table", zebra_stripes=True)
            container.mount(table)

            # Reload trace list
            self._load_and_display_traces()

        elif previous_view == "list":
            # Go back to trace list
            # Remove current view and restore table
            container = self.query_one("#content", Container)
            container.remove_children()

            # Re-create table
            table = DataTable(id="trace-table", zebra_stripes=True)
            container.mount(table)

            # Reload trace list
            self._load_and_display_traces()
            self.current_trace_id = None

        elif previous_view == "tree":
            # Go back to trace tree view
            if self.current_trace_id:
                # Remove span detail and restore tree
                container = self.query_one("#content", Container)
                container.remove_children()

                # Re-show trace detail (which will recreate the tree)
                trace_spans = self.traces_data.get(self.current_trace_id, [])
                if trace_spans:
                    # Build span tree
                    span_tree = build_span_tree(trace_spans)

                    # Create Tree widget
                    tree = Tree(f"Trace: {self.current_trace_id[:16]}", id="span-tree")
                    tree.root.expand()

                    # Populate tree with spans
                    self._add_spans_to_tree(tree.root, span_tree)

                    # Add tree to container
                    container.mount(tree)

                    # Update header
                    header = self.query_one("#header", Static)
                    header.update(
                        f"Trace Detail: {self.current_trace_id[:16]} (Press Esc to go back)"
                    )

                self.current_span = None


def run_explorer(trace_dir: Optional[Path] = None) -> None:
    """Run the interactive trace explorer.

    Args:
        trace_dir: Directory containing traces (default: from config)
    """
    if trace_dir is None:
        config = load_config()
        trace_dir = Path(config.get("trace_dir", "./traces"))

    app = TraceExplorer(trace_dir)
    app.run()


if __name__ == "__main__":
    # For testing purposes
    run_explorer()
