"""Console reporter for evaluation results with rich terminal output.

This module provides a reporter that prints evaluation results to the console
with beautiful formatting, colors, and tree structures for easy debugging.
"""

from __future__ import annotations

from prela.evals.runner import EvalRunResult

# Try to import rich for colored output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ConsoleReporter:
    """Reporter that pretty-prints evaluation results to the console.

    Uses rich library for colored output if available, falls back to
    plain text formatting otherwise. Provides:
    - Summary statistics (pass rate, duration)
    - List of all test cases with pass/fail status
    - Detailed failure information for failed cases
    - Color coding (green=pass, red=fail, yellow=warning)

    Example:
        >>> from prela.evals import EvalRunner
        >>> from prela.evals.reporters import ConsoleReporter
        >>>
        >>> runner = EvalRunner(suite, agent)
        >>> result = runner.run()
        >>>
        >>> reporter = ConsoleReporter(verbose=True, use_colors=True)
        >>> reporter.report(result)
        ✓ Geography QA Suite
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total: 10 | Passed: 9 (90.0%) | Failed: 1
        Duration: 2.5s
        ...
    """

    def __init__(self, verbose: bool = True, use_colors: bool = True):
        """Initialize the console reporter.

        Args:
            verbose: If True, show detailed failure information. If False,
                     only show summary statistics and failed case names.
            use_colors: If True and rich is available, use colored output.
                        If False or rich unavailable, use plain text.
        """
        self.verbose = verbose
        self.use_colors = use_colors and HAS_RICH
        if self.use_colors:
            self.console = Console()

    def report(self, result: EvalRunResult) -> None:
        """Print the evaluation results to the console.

        Args:
            result: The evaluation run result to report.
        """
        if self.use_colors:
            self._report_rich(result)
        else:
            self._report_plain(result)

    def _report_rich(self, result: EvalRunResult) -> None:
        """Print results using rich library (colored output)."""
        # Create title with status symbol
        title = Text()
        if result.pass_rate == 1.0:
            title.append("✓ ", style="bold green")
        elif result.pass_rate == 0.0:
            title.append("✗ ", style="bold red")
        else:
            title.append("⚠ ", style="bold yellow")
        title.append(result.suite_name, style="bold")

        # Create summary statistics
        duration = (result.completed_at - result.started_at).total_seconds()
        summary = (
            f"Total: {result.total_cases} | "
            f"[green]Passed: {result.passed_cases}[/green] "
            f"([cyan]{result.pass_rate * 100:.1f}%[/cyan]) | "
            f"[red]Failed: {result.failed_cases}[/red]\n"
            f"Duration: {duration:.2f}s"
        )

        # Print panel with summary
        panel = Panel(
            summary,
            title=title,
            border_style="blue" if result.pass_rate == 1.0 else "yellow",
        )
        self.console.print(panel)
        self.console.print()

        # Create table of test cases
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Status", width=8)
        table.add_column("Test Case")
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Assertions", justify="center", width=12)

        for case_result in result.case_results:
            # Status column with color
            if case_result.passed:
                status = Text("✓ PASS", style="bold green")
            else:
                status = Text("✗ FAIL", style="bold red")

            # Duration formatting
            duration_str = f"{case_result.duration_ms:.1f}ms"

            # Assertion counts
            total_assertions = len(case_result.assertion_results)
            passed_assertions = sum(
                1 for a in case_result.assertion_results if a.passed
            )
            assertion_str = f"{passed_assertions}/{total_assertions}"

            table.add_row(
                status,
                case_result.case_name,
                duration_str,
                assertion_str,
            )

        self.console.print(table)

        # Show detailed failure information if verbose
        if self.verbose and result.failed_cases > 0:
            self.console.print()
            self.console.print("[bold red]Failed Test Details:[/bold red]")
            self.console.print()

            for case_result in result.case_results:
                if not case_result.passed:
                    self.console.print(
                        f"[bold red]✗ {case_result.case_name}[/bold red]"
                    )

                    # Show error if present
                    if case_result.error:
                        self.console.print(
                            f"  [red]Error:[/red] {case_result.error}"
                        )

                    # Show failed assertions
                    for assertion in case_result.assertion_results:
                        if not assertion.passed:
                            self.console.print(
                                f"  [red]✗[/red] {assertion.message}"
                            )
                            if assertion.expected is not None:
                                self.console.print(
                                    f"    [dim]Expected:[/dim] {self._truncate(assertion.expected)}"
                                )
                            if assertion.actual is not None:
                                self.console.print(
                                    f"    [dim]Actual:[/dim] {self._truncate(assertion.actual)}"
                                )

                    self.console.print()

    def _report_plain(self, result: EvalRunResult) -> None:
        """Print results using plain text (no colors)."""
        # Print header
        if result.pass_rate == 1.0:
            status_symbol = "✓"
        elif result.pass_rate == 0.0:
            status_symbol = "✗"
        else:
            status_symbol = "⚠"

        print(f"{status_symbol} {result.suite_name}")
        print("=" * 60)

        # Print summary
        duration = (result.completed_at - result.started_at).total_seconds()
        print(f"Total: {result.total_cases} | ", end="")
        print(f"Passed: {result.passed_cases} ({result.pass_rate * 100:.1f}%) | ", end="")
        print(f"Failed: {result.failed_cases}")
        print(f"Duration: {duration:.2f}s")
        print()

        # Print test cases
        print("Test Cases:")
        print("-" * 60)
        for case_result in result.case_results:
            status = "✓ PASS" if case_result.passed else "✗ FAIL"
            duration_str = f"{case_result.duration_ms:.1f}ms"
            total_assertions = len(case_result.assertion_results)
            passed_assertions = sum(
                1 for a in case_result.assertion_results if a.passed
            )
            assertion_str = f"{passed_assertions}/{total_assertions}"

            print(
                f"{status:8} {case_result.case_name:35} "
                f"{duration_str:>10} {assertion_str:>12}"
            )

        # Show detailed failure information if verbose
        if self.verbose and result.failed_cases > 0:
            print()
            print("Failed Test Details:")
            print("=" * 60)

            for case_result in result.case_results:
                if not case_result.passed:
                    print(f"\n✗ {case_result.case_name}")

                    # Show error if present
                    if case_result.error:
                        print(f"  Error: {case_result.error}")

                    # Show failed assertions
                    for assertion in case_result.assertion_results:
                        if not assertion.passed:
                            print(f"  ✗ {assertion.message}")
                            if assertion.expected is not None:
                                print(
                                    f"    Expected: {self._truncate(assertion.expected)}"
                                )
                            if assertion.actual is not None:
                                print(
                                    f"    Actual: {self._truncate(assertion.actual)}"
                                )

    def _truncate(self, value: any, max_length: int = 100) -> str:
        """Truncate long strings for display.

        Args:
            value: The value to truncate (will be converted to string).
            max_length: Maximum length before truncation.

        Returns:
            Truncated string with "..." suffix if needed.
        """
        value_str = str(value)
        if len(value_str) > max_length:
            return value_str[: max_length - 3] + "..."
        return value_str
