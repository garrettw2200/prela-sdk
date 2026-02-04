"""Prela CLI - Command-line interface for AI agent observability.

This module provides a CLI for managing Prela configuration, viewing traces,
and running evaluations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import typer
    import yaml
    from rich.console import Console
    from rich.table import Table
    from rich.tree import Tree
except ImportError as e:
    print(
        f"CLI dependencies not installed: {e}\n"
        "Install with: pip install prela[cli]",
        file=sys.stderr,
    )
    sys.exit(1)

from prela.core.span import Span

app = typer.Typer(
    name="prela",
    help="Prela - AI Agent Observability Platform CLI",
    no_args_is_help=True,
)

console = Console()

# Default configuration
DEFAULT_CONFIG = {
    "service_name": "my-agent",
    "exporter": "file",
    "trace_dir": "./traces",
    "sample_rate": 1.0,
}

CONFIG_FILE = ".prela.yaml"


def load_config() -> dict[str, Any]:
    """Load configuration from .prela.yaml file."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return {**DEFAULT_CONFIG, **config}
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load {CONFIG_FILE}: {e}[/yellow]")
        return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to .prela.yaml file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]✓ Configuration saved to {CONFIG_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to save configuration: {e}[/red]")
        raise typer.Exit(1)


def parse_duration(duration: str) -> timedelta:
    """Parse duration string like '1h', '30m', '2d' into timedelta."""
    duration = duration.strip().lower()
    if not duration:
        raise ValueError("Duration cannot be empty")

    # Extract number and unit
    unit = duration[-1]
    try:
        value = int(duration[:-1])
    except ValueError:
        raise ValueError(f"Invalid duration format: {duration}")

    if unit == "s":
        return timedelta(seconds=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    else:
        raise ValueError(f"Unknown duration unit: {unit} (use s, m, h, or d)")


def load_traces_from_file(
    trace_dir: Path, since: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Load traces from JSONL file(s) in trace directory."""
    traces = []

    if not trace_dir.exists():
        return traces

    # Find all .jsonl files
    jsonl_files = sorted(trace_dir.glob("*.jsonl"))

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        span_data = json.loads(line)

                        # Filter by time if requested
                        if since:
                            started_at = datetime.fromisoformat(
                                span_data.get("started_at", "")
                            )
                            if started_at < since:
                                continue

                        traces.append(span_data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            console.print(
                f"[yellow]Warning: Failed to read {jsonl_file}: {e}[/yellow]"
            )

    return traces


def group_spans_by_trace(spans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group spans by trace_id."""
    traces: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        trace_id = span.get("trace_id", "unknown")
        if trace_id not in traces:
            traces[trace_id] = []
        traces[trace_id].append(span)
    return traces


def find_root_span(spans: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Find root span (parent_span_id is None) in a list of spans."""
    for span in spans:
        if span.get("parent_span_id") is None:
            return span
    return spans[0] if spans else None


def build_span_tree(
    spans: list[dict[str, Any]], parent_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """Build hierarchical tree of spans."""
    tree = []
    for span in spans:
        if span.get("parent_span_id") == parent_id:
            children = build_span_tree(spans, span.get("span_id"))
            tree.append({"span": span, "children": children})
    return tree


def render_span_tree(
    tree: list[dict[str, Any]], rich_tree: Optional[Tree] = None, is_root: bool = True
) -> Tree:
    """Render span tree using Rich Tree."""
    if rich_tree is None:
        # Create root tree
        if tree:
            root_span = tree[0]["span"]
            root_label = format_span_label(root_span)
            rich_tree = Tree(root_label)

            # Render children
            for child in tree[0].get("children", []):
                render_span_tree([child], rich_tree, is_root=False)

            # If there are more root spans, add them
            for node in tree[1:]:
                span = node["span"]
                branch = rich_tree.add(format_span_label(span))
                for child in node.get("children", []):
                    render_span_tree([child], branch, is_root=False)
        else:
            rich_tree = Tree("[dim]No spans[/dim]")
    else:
        # Add to existing tree
        for node in tree:
            span = node["span"]
            branch = rich_tree.add(format_span_label(span))
            for child in node.get("children", []):
                render_span_tree([child], branch, is_root=False)

    return rich_tree


def format_span_label(span: dict[str, Any]) -> str:
    """Format span as a label for tree display."""
    name = span.get("name", "unknown")
    span_type = span.get("span_type", "unknown")
    status = span.get("status", "unknown")
    duration_ms = span.get("duration_ms", 0)

    # Status color
    status_color = "green" if status == "success" else "red" if status == "error" else "yellow"

    # Format duration
    if duration_ms > 1000:
        duration_str = f"{duration_ms / 1000:.2f}s"
    else:
        duration_str = f"{duration_ms:.0f}ms"

    return f"[bold]{name}[/bold] [dim]({span_type})[/dim] [{status_color}]{status}[/{status_color}] [dim]{duration_str}[/dim]"


@app.command()
def init() -> None:
    """Initialize Prela configuration with interactive prompts."""
    console.print("[bold blue]Prela Configuration Setup[/bold blue]\n")

    # Load existing config if available
    existing_config = load_config()

    # Prompt for service name
    service_name = typer.prompt(
        "Service name", default=existing_config.get("service_name", "my-agent")
    )

    # Prompt for exporter
    exporter = typer.prompt(
        "Exporter (console/file)", default=existing_config.get("exporter", "file")
    )

    # Prompt for trace directory (if file exporter)
    if exporter == "file":
        trace_dir = typer.prompt(
            "Trace directory", default=existing_config.get("trace_dir", "./traces")
        )
    else:
        trace_dir = existing_config.get("trace_dir", "./traces")

    # Prompt for sample rate
    sample_rate_str = typer.prompt(
        "Sample rate (0.0-1.0)", default=str(existing_config.get("sample_rate", 1.0))
    )

    try:
        sample_rate = float(sample_rate_str)
        if not 0.0 <= sample_rate <= 1.0:
            console.print("[red]Sample rate must be between 0.0 and 1.0[/red]")
            raise typer.Exit(1)
    except ValueError:
        console.print("[red]Invalid sample rate[/red]")
        raise typer.Exit(1)

    # Build config
    config = {
        "service_name": service_name,
        "exporter": exporter,
        "trace_dir": trace_dir,
        "sample_rate": sample_rate,
    }

    # Save config
    save_config(config)

    # Create trace directory if needed
    if exporter == "file":
        trace_path = Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓ Created trace directory: {trace_dir}[/green]")


@app.command(name="list")
def list_traces(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of traces to show"),
    since: Optional[str] = typer.Option(
        None, "--since", "-s", help="Show traces since duration (e.g., '1h', '30m', '2d')"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enable interactive selection (numbered list)"
    ),
) -> None:
    """List recent traces from file exporter.

    With --interactive, displays a numbered list and prompts for selection,
    then automatically shows the selected trace details.
    """
    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    # Parse since duration
    since_dt = None
    if since:
        try:
            duration = parse_duration(since)
            since_dt = datetime.now(timezone.utc) - duration
        except ValueError as e:
            console.print(f"[red]Invalid duration: {e}[/red]")
            raise typer.Exit(1)

    # Load traces
    spans = load_traces_from_file(trace_dir, since=since_dt)

    if not spans:
        console.print("[yellow]No traces found[/yellow]")
        return

    # Group by trace_id
    traces = group_spans_by_trace(spans)

    # Get root spans for each trace
    trace_summaries = []
    for trace_id, trace_spans in traces.items():
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

    # Limit results
    trace_summaries = trace_summaries[:limit]

    # Display table
    if interactive:
        # Interactive mode: numbered list
        table = Table(title=f"Recent Traces ({len(trace_summaries)} of {len(traces)}) - Select by number")
        table.add_column("#", style="bold yellow", justify="right", width=4)
    else:
        # Normal mode: standard table
        table = Table(title=f"Recent Traces ({len(trace_summaries)} of {len(traces)})")

    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Root Span", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Spans", justify="right")
    table.add_column("Time", style="dim")

    for idx, summary in enumerate(trace_summaries, start=1):
        # Format duration
        duration_ms = summary["duration_ms"]
        if duration_ms > 1000:
            duration_str = f"{duration_ms / 1000:.2f}s"
        else:
            duration_str = f"{duration_ms:.0f}ms"

        # Status color
        status = summary["status"]
        status_color = "green" if status == "success" else "red" if status == "error" else "yellow"

        # Format time
        try:
            started_at = datetime.fromisoformat(summary["started_at"])
            time_str = started_at.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = summary["started_at"][:19]

        if interactive:
            # Add row number in interactive mode
            table.add_row(
                str(idx),
                summary["trace_id"][:16],
                summary["root_span"],
                duration_str,
                f"[{status_color}]{status}[/{status_color}]",
                str(summary["span_count"]),
                time_str,
            )
        else:
            # Normal row without number
            table.add_row(
                summary["trace_id"][:16],
                summary["root_span"],
                duration_str,
                f"[{status_color}]{status}[/{status_color}]",
                str(summary["span_count"]),
                time_str,
            )

    console.print(table)

    # Interactive selection
    if interactive:
        console.print()  # Blank line
        try:
            selection = typer.prompt(
                f"Select trace (1-{len(trace_summaries)}), or 'q' to quit",
                default="q",
            )

            # Handle quit
            if selection.lower() == "q":
                return

            # Validate selection
            try:
                selected_idx = int(selection)
                if not 1 <= selected_idx <= len(trace_summaries):
                    console.print(f"[red]Invalid selection. Must be between 1 and {len(trace_summaries)}[/red]")
                    raise typer.Exit(1)
            except ValueError:
                console.print("[red]Invalid input. Please enter a number or 'q'[/red]")
                raise typer.Exit(1)

            # Get selected trace
            selected_trace = trace_summaries[selected_idx - 1]
            selected_trace_id = selected_trace["trace_id"]

            # Automatically show the selected trace
            console.print(f"\n[bold blue]→ Showing trace {selected_idx}: {selected_trace_id[:16]}...[/bold blue]\n")
            show_trace(selected_trace_id)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Selection cancelled[/yellow]")
            return


@app.command(name="show")
def show_trace(
    trace_id: str,
    compact: bool = typer.Option(
        False, "--compact", "-c", help="Show tree only (no detailed attributes)"
    ),
) -> None:
    """Display full trace tree with all spans, attributes, and events.

    Use --compact to show only the tree structure without detailed span information.
    """
    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    # Load all traces
    spans = load_traces_from_file(trace_dir)

    # Filter by trace_id (support partial match)
    matching_spans = [s for s in spans if s.get("trace_id", "").startswith(trace_id)]

    if not matching_spans:
        console.print(f"[red]No trace found with ID: {trace_id}[/red]")
        raise typer.Exit(1)

    # Get full trace_id
    full_trace_id = matching_spans[0].get("trace_id")

    # Get all spans for this trace
    trace_spans = [s for s in spans if s.get("trace_id") == full_trace_id]

    console.print(f"\n[bold blue]Trace:[/bold blue] {full_trace_id}\n")

    # Build and render tree
    span_tree = build_span_tree(trace_spans, parent_id=None)
    tree = render_span_tree(span_tree)
    console.print(tree)

    # Show detailed attributes for each span (unless compact mode)
    if not compact:
        console.print("\n[bold blue]Span Details:[/bold blue]\n")

        for span in sorted(trace_spans, key=lambda s: s.get("started_at", "")):
            console.print(f"[bold cyan]{span.get('name', 'unknown')}[/bold cyan]")
            console.print(f"  Span ID: {span.get('span_id', 'unknown')}")
            console.print(f"  Type: {span.get('span_type', 'unknown')}")
            console.print(f"  Status: {span.get('status', 'unknown')}")

            # Attributes
            attributes = span.get("attributes", {})
            if attributes:
                console.print("  Attributes:")
                for key, value in sorted(attributes.items()):
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    console.print(f"    {key}: {value_str}")

            # Events
            events = span.get("events", [])
            if events:
                console.print(f"  Events ({len(events)}):")
                for event in events:
                    event_name = event.get("name", "unknown")
                    timestamp = event.get("timestamp", "")
                    console.print(f"    - {event_name} @ {timestamp}")

            console.print()
    else:
        # In compact mode, show helpful tip
        console.print("\n[dim]💡 Tip: Run without --compact to see full span details[/dim]\n")


@app.command(name="search")
def search_traces(query: str) -> None:
    """Search span names and attributes for matching traces."""
    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    # Load all traces
    spans = load_traces_from_file(trace_dir)

    if not spans:
        console.print("[yellow]No traces found[/yellow]")
        return

    # Search spans
    query_lower = query.lower()
    matching_spans = []

    for span in spans:
        # Search in span name
        if query_lower in span.get("name", "").lower():
            matching_spans.append(span)
            continue

        # Search in attributes
        attributes = span.get("attributes", {})
        for key, value in attributes.items():
            if query_lower in key.lower() or query_lower in str(value).lower():
                matching_spans.append(span)
                break

    if not matching_spans:
        console.print(f"[yellow]No traces found matching: {query}[/yellow]")
        return

    # Group by trace
    traces = group_spans_by_trace(matching_spans)

    console.print(f"\n[bold green]Found {len(traces)} traces matching '{query}'[/bold green]\n")

    # Display table
    table = Table(title=f"Search Results")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Root Span", style="bold")
    table.add_column("Matching Spans", justify="right")
    table.add_column("Status", justify="center")

    for trace_id, trace_spans in traces.items():
        root_span = find_root_span(trace_spans)
        if root_span:
            status = root_span.get("status", "unknown")
            status_color = (
                "green" if status == "success" else "red" if status == "error" else "yellow"
            )

            table.add_row(
                trace_id[:16],
                root_span.get("name", "unknown"),
                str(len(trace_spans)),
                f"[{status_color}]{status}[/{status_color}]",
            )

    console.print(table)


@app.command(name="replay")
def replay_trace(
    trace_file: str = typer.Argument(..., help="Path to trace file (JSON or JSONL)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Override temperature"
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt", "-s", help="Override system prompt"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Override max_tokens"
    ),
    compare: bool = typer.Option(
        False, "--compare", "-c", help="Compare replay with original"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save replay result to file"
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Use streaming API and show real-time output"
    ),
) -> None:
    """Replay a captured trace with optional modifications.

    Examples:
        prela replay trace.json
        prela replay trace.json --model gpt-4o --compare
        prela replay trace.json --temperature 0.7 --output result.json
        prela replay trace.json --model claude-sonnet-4 --stream
    """
    from prela.replay import ReplayEngine, compare_replays
    from prela.replay.loader import TraceLoader

    console.print(f"[cyan]Loading trace from {trace_file}...[/cyan]")

    # Load trace
    try:
        trace_path = Path(trace_file)
        if not trace_path.exists():
            console.print(f"[red]✗ File not found: {trace_file}[/red]")
            raise typer.Exit(1)

        if trace_path.suffix == ".jsonl":
            traces = TraceLoader.from_jsonl(trace_path)
            if not traces:
                console.print("[red]✗ No traces found in JSONL file[/red]")
                raise typer.Exit(1)
            trace = traces[0]  # Use first trace
            if len(traces) > 1:
                console.print(
                    f"[yellow]Note: Found {len(traces)} traces, using first one[/yellow]"
                )
        else:
            trace = TraceLoader.from_file(trace_path)

        console.print(
            f"[green]✓ Loaded trace {trace.trace_id} with {len(trace.spans)} spans[/green]"
        )

    except Exception as e:
        console.print(f"[red]✗ Failed to load trace: {e}[/red]")
        raise typer.Exit(1)

    # Create replay engine
    try:
        engine = ReplayEngine(trace)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)

    # Execute replay
    console.print("\n[cyan]Executing replay...[/cyan]")

    # Create streaming callback if requested
    stream_callback = None
    if stream:
        console.print("[dim]Streaming enabled - showing real-time output:[/dim]\n")

        def streaming_callback(chunk_text: str) -> None:
            """Print streaming chunks in real-time."""
            console.print(chunk_text, end="", highlight=False)

        stream_callback = streaming_callback

    try:
        # Check if modifications requested
        has_modifications = any([model, temperature, system_prompt, max_tokens])

        if has_modifications:
            # Modified replay
            result = engine.replay_with_modifications(
                model=model,
                temperature=temperature,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                stream=stream,
                stream_callback=stream_callback,
            )
            if stream:
                console.print("\n")  # New line after streaming output
            console.print(
                f"[green]✓ Modified replay completed ({result.modified_span_count} spans modified)[/green]"
            )
        else:
            # Exact replay
            result = engine.replay_exact()
            console.print("[green]✓ Exact replay completed[/green]")

    except NotImplementedError as e:
        console.print(
            f"[yellow]Warning: {e}[/yellow]\n"
            f"[dim]Currently only exact replay is supported. "
            f"Real API calls for modified replay coming soon.[/dim]"
        )
        # Fall back to exact replay
        result = engine.replay_exact()

    except Exception as e:
        console.print(f"[red]✗ Replay failed: {e}[/red]")
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold]Replay Results:[/bold]")
    console.print(f"  Trace ID: {result.trace_id}")
    console.print(f"  Total Spans: {len(result.spans)}")
    console.print(f"  Duration: {result.total_duration_ms:.1f}ms")
    console.print(f"  Tokens: {result.total_tokens}")
    console.print(f"  Cost: ${result.total_cost_usd:.4f}")
    console.print(f"  Success: {'✓' if result.success else '✗'}")

    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for error in result.errors[:5]:  # Show first 5
            console.print(f"  • {error}")

    if result.final_output is not None:
        console.print(f"\n[bold]Final Output:[/bold]")
        if isinstance(result.final_output, str):
            output_preview = result.final_output[:200]
            if len(result.final_output) > 200:
                output_preview += "..."
            console.print(f"  {output_preview}")
        else:
            console.print(f"  {result.final_output}")

    # Compare with original if requested
    if compare and has_modifications:
        console.print("\n[cyan]Comparing with original execution...[/cyan]")

        try:
            # Run exact replay for comparison
            original_result = engine.replay_exact()

            # Compare
            comparison = compare_replays(original_result, result)

            # Display comparison
            console.print("\n" + comparison.generate_summary())

            # Show top differences with semantic similarity
            if comparison.differences:
                console.print("\n[bold]Top Differences:[/bold]")
                for diff in comparison.differences[:10]:  # Show first 10
                    sim_text = ""
                    if diff.semantic_similarity is not None:
                        sim_text = f" (similarity: {diff.semantic_similarity:.1%})"

                    console.print(
                        f"\n  • {diff.span_name} - {diff.field}{sim_text}"
                    )
                    console.print(f"    Original: {str(diff.original_value)[:100]}")
                    console.print(f"    Modified: {str(diff.modified_value)[:100]}")

        except Exception as e:
            console.print(f"[yellow]Warning: Comparison failed: {e}[/yellow]")

    # Save output if requested
    if output:
        try:
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            console.print(f"\n[green]✓ Results saved to {output}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save output: {e}[/yellow]")


@app.command()
def explore() -> None:
    """Launch interactive trace explorer (TUI).

    Opens an interactive terminal interface for browsing traces, navigating
    span hierarchies, and inspecting details without copy/pasting trace IDs.

    Keyboard shortcuts:
        ↑/k: Move up
        ↓/j: Move down
        Enter: Select/drill down
        Esc: Go back
        q: Quit

    Example:
        $ prela explore
    """
    try:
        from prela.contrib.explorer import run_explorer
    except ImportError:
        console.print(
            "[red]Textual dependency not installed[/red]\n"
            "Install with: pip install prela[cli]"
        )
        raise typer.Exit(1)

    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    if not trace_dir.exists():
        console.print(
            f"[yellow]Trace directory not found: {trace_dir}[/yellow]\n"
            f"Run 'prela init' to configure or use 'prela list' to see traces."
        )
        raise typer.Exit(1)

    try:
        run_explorer(trace_dir)
    except Exception as e:
        console.print(f"[red]Explorer failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="last")
def last_trace(
    compact: bool = typer.Option(
        False, "--compact", "-c", help="Show tree only (no detailed attributes)"
    ),
) -> None:
    """Show the most recent trace.

    Shortcut for viewing the latest trace without copy/pasting IDs.
    """
    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    # Load traces
    spans = load_traces_from_file(trace_dir)

    if not spans:
        console.print("[yellow]No traces found[/yellow]")
        return

    # Group by trace_id
    traces = group_spans_by_trace(spans)

    # Get root spans for each trace
    trace_summaries = []
    for trace_id, trace_spans in traces.items():
        root_span = find_root_span(trace_spans)
        if root_span:
            trace_summaries.append(
                {
                    "trace_id": trace_id,
                    "started_at": root_span.get("started_at", ""),
                }
            )

    # Sort by time (most recent first)
    trace_summaries.sort(key=lambda x: x["started_at"], reverse=True)

    if not trace_summaries:
        console.print("[yellow]No valid traces found[/yellow]")
        return

    # Get most recent trace
    most_recent = trace_summaries[0]
    trace_id = most_recent["trace_id"]

    console.print(f"[dim]Showing most recent trace ({trace_id[:16]}...)[/dim]\n")
    show_trace(trace_id, compact=compact)


@app.command(name="errors")
def error_traces(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of traces to show"),
) -> None:
    """Show only failed traces.

    Shortcut for filtering error traces without manual searching.
    """
    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    # Load traces
    spans = load_traces_from_file(trace_dir)

    if not spans:
        console.print("[yellow]No traces found[/yellow]")
        return

    # Group by trace_id
    traces = group_spans_by_trace(spans)

    # Get root spans for error traces
    error_summaries = []
    for trace_id, trace_spans in traces.items():
        root_span = find_root_span(trace_spans)
        if root_span and root_span.get("status") == "error":
            error_summaries.append(
                {
                    "trace_id": trace_id,
                    "root_span": root_span.get("name", "unknown"),
                    "duration_ms": root_span.get("duration_ms", 0),
                    "started_at": root_span.get("started_at", ""),
                    "span_count": len(trace_spans),
                }
            )

    # Sort by time (most recent first)
    error_summaries.sort(key=lambda x: x["started_at"], reverse=True)

    # Limit results
    error_summaries = error_summaries[:limit]

    if not error_summaries:
        console.print("[green]✓ No failed traces found - all systems nominal![/green]")
        return

    # Display table
    table = Table(title=f"Failed Traces ({len(error_summaries)} errors)")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Root Span", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Spans", justify="right")
    table.add_column("Time", style="dim")

    for summary in error_summaries:
        # Format duration
        duration_ms = summary["duration_ms"]
        if duration_ms > 1000:
            duration_str = f"{duration_ms / 1000:.2f}s"
        else:
            duration_str = f"{duration_ms:.0f}ms"

        # Format time
        try:
            started_at = datetime.fromisoformat(summary["started_at"])
            time_str = started_at.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = summary["started_at"][:19]

        table.add_row(
            summary["trace_id"][:16],
            summary["root_span"],
            duration_str,
            str(summary["span_count"]),
            time_str,
        )

    console.print(table)
    console.print(f"\n[dim]💡 Tip: Use 'prela show <trace-id>' to inspect a specific error[/dim]\n")


@app.command(name="tail")
def tail_traces(
    interval: int = typer.Option(2, "--interval", "-i", help="Polling interval in seconds"),
    compact: bool = typer.Option(
        False, "--compact", "-c", help="Show compact output (no details)"
    ),
) -> None:
    """Follow new traces in real-time.

    Simple polling mode that shows new traces as they arrive.
    Press Ctrl+C to stop.
    """
    import time

    config = load_config()
    trace_dir = Path(config.get("trace_dir", "./traces"))

    console.print(f"[cyan]Following traces in {trace_dir} (polling every {interval}s)[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Track seen trace IDs
    seen_trace_ids: set[str] = set()

    # Initial load
    spans = load_traces_from_file(trace_dir)
    traces = group_spans_by_trace(spans)
    seen_trace_ids.update(traces.keys())

    try:
        while True:
            time.sleep(interval)

            # Load traces
            spans = load_traces_from_file(trace_dir)
            traces = group_spans_by_trace(spans)

            # Find new traces
            new_trace_ids = set(traces.keys()) - seen_trace_ids

            if new_trace_ids:
                for trace_id in sorted(new_trace_ids):
                    trace_spans = traces[trace_id]
                    root_span = find_root_span(trace_spans)

                    if root_span:
                        # Format timestamp
                        now = datetime.now()
                        timestamp = now.strftime("%H:%M:%S")

                        # Status color and icon
                        status = root_span.get("status", "unknown")
                        if status == "success":
                            status_icon = "✓"
                            status_color = "green"
                        elif status == "error":
                            status_icon = "✗"
                            status_color = "red"
                        else:
                            status_icon = "○"
                            status_color = "yellow"

                        # Format duration
                        duration_ms = root_span.get("duration_ms", 0)
                        if duration_ms > 1000:
                            duration_str = f"{duration_ms / 1000:.2f}s"
                        else:
                            duration_str = f"{duration_ms:.0f}ms"

                        # Print compact or detailed
                        if compact:
                            console.print(
                                f"[dim]{timestamp}[/dim] [{status_color}]{status_icon}[/{status_color}] "
                                f"[cyan]{trace_id[:12]}[/cyan] "
                                f"[bold]{root_span.get('name', 'unknown')}[/bold] "
                                f"[dim]{duration_str}[/dim]"
                            )
                        else:
                            console.print(
                                f"[dim][{timestamp}][/dim] "
                                f"[{status_color}]{status_icon} {status.upper()}[/{status_color}] | "
                                f"Trace: [cyan]{trace_id[:16]}[/cyan] | "
                                f"Name: [bold]{root_span.get('name', 'unknown')}[/bold] | "
                                f"Duration: {duration_str}"
                            )

                        # Update seen set
                        seen_trace_ids.add(trace_id)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped following traces[/yellow]")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run server on"),
) -> None:
    """Start local web dashboard (placeholder - not implemented)."""
    console.print(
        f"[yellow]Web dashboard not yet implemented[/yellow]\n"
        f"Planned: Start server on http://localhost:{port}\n"
        f"Will serve API endpoints + static frontend"
    )
    console.print("\n[dim]This feature is planned for Phase 1 (Months 4-8)[/dim]")


@app.command()
def eval(
    suite_path: str = typer.Argument(..., help="Path to eval suite file"),
) -> None:
    """Run evaluation suite (placeholder - not implemented)."""
    console.print(
        f"[yellow]Eval runner not yet implemented[/yellow]\n"
        f"Planned: Run eval suite from {suite_path}\n"
        f"Will output results with pass/fail metrics"
    )
    console.print("\n[dim]This feature is planned for Phase 1 (Months 4-8)[/dim]")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
