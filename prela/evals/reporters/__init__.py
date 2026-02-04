"""Reporter implementations for evaluation results.

This module provides reporters for outputting evaluation results in various formats:
- ConsoleReporter: Pretty-printed terminal output with colors
- JSONReporter: JSON file output for programmatic access
- JUnitReporter: JUnit XML for CI/CD integration

Example:
    >>> from prela.evals import EvalRunner
    >>> from prela.evals.reporters import ConsoleReporter, JSONReporter
    >>>
    >>> runner = EvalRunner(suite, agent)
    >>> result = runner.run()
    >>>
    >>> # Print to console
    >>> console = ConsoleReporter()
    >>> console.report(result)
    >>>
    >>> # Save to JSON
    >>> json_reporter = JSONReporter("results.json")
    >>> json_reporter.report(result)
"""

from prela.evals.reporters.console import ConsoleReporter
from prela.evals.reporters.json import JSONReporter
from prela.evals.reporters.junit import JUnitReporter

__all__ = [
    "ConsoleReporter",
    "JSONReporter",
    "JUnitReporter",
]
