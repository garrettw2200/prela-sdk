"""n8n workflow evaluation framework.

This module provides specialized evaluation tools for testing n8n workflows:
- N8nEvalCase: Test case for n8n workflows with trigger data
- N8nWorkflowEvalConfig: Configuration for n8n workflow evaluation
- N8nWorkflowEvalRunner: Runner for executing n8n workflow tests
- eval_n8n_workflow: Convenience function for quick testing
- n8n-specific assertions: node_completed, node_output, duration_under, etc.

Example:
    >>> from prela.evals.n8n import eval_n8n_workflow, N8nEvalCase
    >>> from prela.evals.n8n.assertions import node_completed, duration_under
    >>>
    >>> results = await eval_n8n_workflow(
    ...     workflow_id="abc123",
    ...     test_cases=[
    ...         N8nEvalCase(
    ...             id="test_1",
    ...             name="High-intent lead",
    ...             trigger_data={"email": "I want to buy..."},
    ...             workflow_assertions=[
    ...                 node_completed("Classify Intent"),
    ...                 duration_under(5.0)
    ...             ]
    ...         )
    ...     ]
    ... )
"""

from prela.evals.n8n.assertions import (
    N8nAINodeTokens,
    N8nNodeCompleted,
    N8nNodeOutput,
    N8nWorkflowDuration,
    N8nWorkflowStatus,
    duration_under,
    node_completed,
    node_output,
    tokens_under,
    workflow_completed,
    workflow_status,
)
from prela.evals.n8n.runner import (
    N8nEvalCase,
    N8nWorkflowEvalConfig,
    N8nWorkflowEvalRunner,
    eval_n8n_workflow,
)

__all__ = [
    # Runner components
    "N8nEvalCase",
    "N8nWorkflowEvalConfig",
    "N8nWorkflowEvalRunner",
    "eval_n8n_workflow",
    # Assertion classes
    "N8nNodeCompleted",
    "N8nNodeOutput",
    "N8nWorkflowDuration",
    "N8nAINodeTokens",
    "N8nWorkflowStatus",
    # Convenience functions
    "node_completed",
    "node_output",
    "duration_under",
    "tokens_under",
    "workflow_completed",
    "workflow_status",
]
