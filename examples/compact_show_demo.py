#!/usr/bin/env python3
"""Demonstration of prela show --compact feature.

This script shows the difference between compact and full modes.
"""

from rich.console import Console
from rich.panel import Panel

console = Console()

console.print("\n[bold blue]prela show --compact Demo[/bold blue]\n")

# Show compact mode
console.print("[bold yellow]COMPACT MODE[/bold yellow] [dim](quick overview)[/dim]\n")
console.print("[dim]$ prela show test-trace-001 --compact[/dim]\n")

console.print("""Trace: test-trace-001

agent.run (agent) success 1.25s
├── llm.openai.chat (llm) success 850ms
└── tool.web_search (tool) success 321ms

💡 Tip: Run without --compact to see full span details
""")

console.print("\n[dim]" + "─" * 70 + "[/dim]\n")

# Show full mode
console.print("[bold yellow]FULL MODE[/bold yellow] [dim](detailed debugging)[/dim]\n")
console.print("[dim]$ prela show test-trace-001[/dim]\n")

console.print("""Trace: test-trace-001

agent.run (agent) success 1.25s
├── llm.openai.chat (llm) success 850ms
└── tool.web_search (tool) success 321ms

Span Details:

agent.run
  Span ID: span-1
  Type: agent
  Status: success
  Attributes:
    agent.goal: Research AI trends
    agent.name: ResearchAgent
  Events (1):
    - agent.started @ 2026-01-29T20:39:12.692514

llm.openai.chat
  Span ID: span-2
  Type: llm
  Status: success
  Attributes:
    llm.completion_tokens: 89
    llm.model: gpt-4
    llm.prompt_tokens: 150
    llm.temperature: 0.7

tool.web_search
  Span ID: span-3
  Type: tool
  Status: success
  Attributes:
    tool.input: AI trends 2026
    tool.name: web_search
    tool.output: Found 5 articles about AI trends...
""")

console.print("\n[bold green]Use Cases:[/bold green]\n")

console.print(Panel(
    "[bold]--compact[/bold] (Quick Overview)\n\n"
    "✓ When you just want to see the flow\n"
    "✓ When debugging execution order\n"
    "✓ When comparing trace structures\n"
    "✓ When viewing many traces\n"
    "✓ When terminal output is limited",
    title="Compact Mode",
    border_style="yellow"
))

console.print()

console.print(Panel(
    "[bold]Full Mode[/bold] (Deep Debugging)\n\n"
    "✓ When investigating errors\n"
    "✓ When examining LLM prompts/responses\n"
    "✓ When reviewing tool inputs/outputs\n"
    "✓ When analyzing token usage\n"
    "✓ When checking exact timestamps",
    title="Full Mode",
    border_style="green"
))

console.print("\n[dim]💡 You can switch between modes anytime by adding/removing --compact[/dim]\n")
