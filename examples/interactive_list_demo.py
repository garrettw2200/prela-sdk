#!/usr/bin/env python3
"""Demonstration of prela list --interactive feature.

This script shows how the interactive list command works.
To actually use it, run:

    prela list --interactive
    # or
    prela list -i

Then select a trace by number to view its details automatically.
"""

from rich.console import Console

console = Console()

console.print("\n[bold blue]prela list --interactive Demo[/bold blue]\n")

console.print("[dim]# Run the interactive list command:[/dim]")
console.print("$ prela list --interactive\n")

console.print("[dim]# You'll see a numbered table:[/dim]")
console.print("""
Recent Traces (2 of 2) - Select by number
┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ # ┃ Trace ID       ┃ Root Span     ┃ Duration ┃ Status  ┃ Spans ┃ Time           ┃
┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 1 │ test-trace-002 │ multi-agent…  │    3.13s │  error  │     4 │ 2026-01-29     │
│   │                │               │          │         │       │ 20:39:52       │
│ 2 │ test-trace-001 │ agent.run     │    1.25s │ success │     3 │ 2026-01-29     │
│   │                │               │          │         │       │ 20:39:12       │
└───┴────────────────┴───────────────┴──────────┴─────────┴───────┴────────────────┘
""")

console.print("[dim]# Enter a number to view that trace:[/dim]")
console.print("Select trace (1-2), or 'q' to quit [q]: [cyan]1[/cyan]\n")

console.print("[bold blue]→ Showing trace 1: test-trace-002...[/bold blue]\n")

console.print("[dim]# The trace detail view is shown automatically![/dim]")
console.print("""
Trace: test-trace-002-3e8f-4b9a-9c5d-1a2b3c4d5e6f

multi-agent.workflow (llm) error 3.13s
├── agent.task (agent) success 1.52s
│   └── llm.completion (llm) success 1.23s
├── agent.task (agent) error 1.61s
│   └── llm.completion (llm) error 1.45s

[Span details follow...]
""")

console.print("\n[bold green]✓ No copy/paste needed![/bold green]")
console.print("[dim]Just type the number, and the trace details appear instantly.[/dim]\n")

# Show comparison
console.print("\n[bold yellow]Before (without --interactive):[/bold yellow]")
console.print("1. Run: prela list")
console.print("2. Look at trace IDs")
console.print("3. Copy a trace ID (or type it manually)")
console.print("4. Run: prela show <trace-id>")

console.print("\n[bold green]After (with --interactive):[/bold green]")
console.print("1. Run: prela list --interactive")
console.print("2. Type the number")
console.print("3. Done! ✨")

console.print("\n[dim]Saves you from copy/pasting long trace IDs![/dim]\n")
