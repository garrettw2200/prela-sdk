#!/usr/bin/env python3
"""Demonstration of prela convenience shortcuts (Week 3).

This script shows examples of the three new convenience commands:
- prela last: Show most recent trace
- prela errors: Show only failed traces
- prela tail: Follow new traces in real-time

To actually use these commands, run them from the terminal:

    prela last
    prela errors
    prela tail

This demo shows what the commands do and when to use them.
"""

from rich.console import Console

console = Console()

console.print("\n[bold blue]prela Convenience Shortcuts Demo (Week 3)[/bold blue]\n")

# ===========================
# 1. prela last
# ===========================

console.print("[bold yellow]1. prela last[/bold yellow] [dim]- Show most recent trace[/dim]\n")

console.print("[dim]Before (4 steps):[/dim]")
console.print("  1. $ prela list")
console.print("  2. Look at trace IDs and timestamps")
console.print("  3. Find the most recent one")
console.print("  4. $ prela show <trace-id>")

console.print("\n[dim]After (1 step):[/dim]")
console.print("  $ prela last")
console.print("  [dim]Showing most recent trace (abc-123...)...[/dim]")
console.print("  [dim]→ Automatically shows the latest trace[/dim]")

console.print("\n[bold green]Use cases:[/bold green]")
console.print("  • Quick check after running your agent")
console.print("  • See what just happened without copy/paste")
console.print("  • Fast feedback loop during development")

console.print("\n[bold cyan]Variations:[/bold cyan]")
console.print("  $ prela last           # Full details")
console.print("  $ prela last --compact # Tree only")

console.print("\n" + "─" * 70 + "\n")

# ===========================
# 2. prela errors
# ===========================

console.print("[bold yellow]2. prela errors[/bold yellow] [dim]- Show only failed traces[/dim]\n")

console.print("[dim]Before (manual filtering):[/dim]")
console.print("  $ prela list")
console.print("  [dim]→ Scan table for red 'error' status[/dim]")
console.print("  [dim]→ Mentally filter out successful traces[/dim]")
console.print("  [dim]→ Copy/paste error trace IDs one by one[/dim]")

console.print("\n[dim]After (automatic):[/dim]")
console.print("  $ prela errors")
console.print()
console.print(
    """  Failed Traces (3 errors)
  ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
  ┃ Trace ID       ┃ Root Span     ┃ Duration ┃ Spans ┃ Time           ┃
  ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
  │ test-trace-002 │ api-failure   │    0.52s │     2 │ 2026-01-29     │
  │                │               │          │       │ 20:45:33       │
  │ test-trace-005 │ timeout       │    5.01s │     3 │ 2026-01-29     │
  │                │               │          │       │ 20:44:12       │
  │ test-trace-008 │ parse-error   │    0.12s │     1 │ 2026-01-29     │
  │                │               │          │       │ 20:43:01       │
  └────────────────┴───────────────┴──────────┴───────┴────────────────┘

  💡 Tip: Use 'prela show <trace-id>' to inspect a specific error
"""
)

console.print("[bold green]Use cases:[/bold green]")
console.print("  • Debugging production failures")
console.print("  • Post-deployment health check")
console.print("  • Finding patterns in errors")

console.print("\n[bold cyan]Variations:[/bold cyan]")
console.print("  $ prela errors               # Default: 20 errors")
console.print("  $ prela errors --limit 50    # Show 50 errors")

console.print("\n[bold green]No errors case:[/bold green]")
console.print("  $ prela errors")
console.print("  ✓ No failed traces found - all systems nominal!")

console.print("\n" + "─" * 70 + "\n")

# ===========================
# 3. prela tail
# ===========================

console.print(
    "[bold yellow]3. prela tail[/bold yellow] [dim]- Follow new traces in real-time[/dim]\n"
)

console.print("[dim]Usage:[/dim]")
console.print("  $ prela tail")
console.print("  Following traces in ./traces (polling every 2s)")
console.print("  Press Ctrl+C to stop")
console.print()

console.print("[dim]Live output (compact mode):[/dim]")
console.print("  20:45:12 ✓ abc-123... multi-step-agent 2.3s")
console.print("  20:45:15 ✓ def-456... api-call 0.5s")
console.print("  20:45:18 ✗ ghi-789... data-pipeline ERROR 1.2s")
console.print("  20:45:21 ✓ jkl-012... workflow 5.1s")
console.print()

console.print("[dim]Live output (detailed mode):[/dim]")
console.print(
    "  [20:45:12] ✓ SUCCESS | Trace: abc-123... | Name: multi-step-agent | Duration: 2.3s"
)
console.print("  [20:45:15] ✓ SUCCESS | Trace: def-456... | Name: api-call | Duration: 0.5s")
console.print(
    "  [20:45:18] ✗ ERROR | Trace: ghi-789... | Name: data-pipeline | Duration: 1.2s"
)
console.print("  [20:45:21] ✓ SUCCESS | Trace: jkl-012... | Name: workflow | Duration: 5.1s")

console.print("\n[bold green]Use cases:[/bold green]")
console.print("  • Monitor live agent executions")
console.print("  • See traces as they complete")
console.print("  • Real-time debugging feedback")
console.print("  • Production health monitoring")

console.print("\n[bold cyan]Variations:[/bold cyan]")
console.print("  $ prela tail                     # Default: 2s interval, detailed")
console.print("  $ prela tail --compact           # Compact one-line output")
console.print("  $ prela tail --interval 5        # Poll every 5 seconds")
console.print("  $ prela tail --compact -i 1      # Fast compact monitoring")

console.print("\n" + "─" * 70 + "\n")

# ===========================
# Workflow Comparison
# ===========================

console.print("[bold blue]Complete Workflow Comparison[/bold blue]\n")

console.print("[bold]Old workflow (copy/paste heavy):[/bold]")
console.print("  1. $ prela list")
console.print("  2. Scan table for issues")
console.print("  3. Copy trace ID")
console.print("  4. $ prela show <trace-id>")
console.print("  5. Repeat for each trace")

console.print("\n[bold green]New workflow (shortcuts):[/bold green]")
console.print("  $ prela last       # See latest execution immediately")
console.print("  $ prela errors     # Check for failures")
console.print("  $ prela tail       # Monitor live (if needed)")

console.print("\n[bold]Result:[/bold]")
console.print("  • ~75% fewer commands")
console.print("  • Zero copy/paste needed")
console.print("  • Faster feedback loop")
console.print("  • More intuitive workflow")

console.print("\n[dim]💡 All these commands compose with Week 2 features:[/dim]")
console.print("  $ prela last --compact          # Quick overview")
console.print("  $ prela list --interactive      # Numbered selection")
console.print("  $ prela show <id> --compact     # Tree only")
console.print("  $ prela tail --compact          # Live monitoring")

console.print()
