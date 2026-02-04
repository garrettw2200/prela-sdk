"""Evaluation framework for AI agents.

This module provides a comprehensive evaluation framework for testing AI agents:
- Test case definition (EvalInput, EvalExpected, EvalCase)
- Test suite management (EvalSuite)
- YAML-based configuration
- Assertion framework
- Result tracking
- n8n workflow evaluation

Example:
    >>> from prela.evals import EvalSuite, EvalCase, EvalInput, EvalExpected
    >>>
    >>> # Define a test case
    >>> case = EvalCase(
    ...     id="test_qa",
    ...     name="Basic QA test",
    ...     input=EvalInput(query="What is the capital of France?"),
    ...     expected=EvalExpected(contains=["Paris"]),
    ...     assertions=[
    ...         {"type": "contains", "value": "Paris"},
    ...         {"type": "semantic_similarity", "threshold": 0.8}
    ...     ]
    ... )
    >>>
    >>> # Create a suite
    >>> suite = EvalSuite(
    ...     name="Geography QA Suite",
    ...     description="Tests for geography knowledge",
    ...     cases=[case]
    ... )
    >>>
    >>> # Save to YAML
    >>> suite.to_yaml("geography_qa.yaml")
    >>>
    >>> # Load from YAML
    >>> loaded_suite = EvalSuite.from_yaml("geography_qa.yaml")
"""

from prela.evals.case import EvalCase, EvalExpected, EvalInput

# n8n evaluation framework
from prela.evals.n8n import (
    N8nEvalCase,
    N8nWorkflowEvalConfig,
    N8nWorkflowEvalRunner,
    eval_n8n_workflow,
)

# n8n assertions (convenience re-export)
from prela.evals.n8n.assertions import (
    duration_under,
    node_completed,
    node_output,
    tokens_under,
    workflow_completed,
    workflow_status,
)
from prela.evals.reporters import ConsoleReporter, JSONReporter, JUnitReporter
from prela.evals.runner import (
    CaseResult,
    EvalRunResult,
    EvalRunner,
    create_assertion,
)
from prela.evals.suite import EvalSuite

__all__ = [
    # Core eval framework
    "EvalCase",
    "EvalExpected",
    "EvalInput",
    "EvalSuite",
    "CaseResult",
    "EvalRunResult",
    "EvalRunner",
    "create_assertion",
    # Reporters
    "ConsoleReporter",
    "JSONReporter",
    "JUnitReporter",
    # n8n workflow evaluation
    "N8nEvalCase",
    "N8nWorkflowEvalConfig",
    "N8nWorkflowEvalRunner",
    "eval_n8n_workflow",
    # n8n assertions (convenience)
    "node_completed",
    "node_output",
    "duration_under",
    "tokens_under",
    "workflow_completed",
    "workflow_status",
]
