"""
n8n workflow automation platform instrumentation.

This module provides automatic tracing for n8n workflows, capturing:
- Workflow executions with status and timing
- Individual node executions (AI and non-AI)
- LLM calls with token usage and costs
- Tool invocations and retrievals
- Error tracking and retries

Usage:
    import prela
    from prela.instrumentation.n8n import N8nInstrumentor

    # Initialize Prela
    tracer = prela.init(service_name="n8n-workflows")

    # Instrument n8n
    N8nInstrumentor().instrument(tracer)

    # All n8n workflow executions are now automatically traced
"""

from __future__ import annotations

# Check tier on module import
from prela.license import check_tier

if not check_tier("n8n instrumentation", "lunch-money", silent=False):
    raise ImportError(
        "n8n instrumentation requires 'lunch-money' subscription or higher. "
        "Upgrade at https://prela.dev/pricing"
    )

from prela.instrumentation.n8n.code_node import (
    PrelaN8nContext,
    prela_n8n_traced,
    trace_n8n_code,
)
from prela.instrumentation.n8n.models import (
    N8nAINodeExecution,
    N8nNodeExecution,
    N8nSpanType,
    N8nWorkflowExecution,
)
from prela.instrumentation.n8n.webhook import (
    N8N_AI_NODE_TYPES,
    N8nWebhookHandler,
    N8nWebhookPayload,
    parse_n8n_webhook,
)

__all__ = [
    # Models
    "N8nWorkflowExecution",
    "N8nNodeExecution",
    "N8nAINodeExecution",
    "N8nSpanType",
    # Webhook
    "N8nWebhookHandler",
    "N8nWebhookPayload",
    "parse_n8n_webhook",
    "N8N_AI_NODE_TYPES",
    # Code Node Helpers
    "PrelaN8nContext",
    "trace_n8n_code",
    "prela_n8n_traced",
]
