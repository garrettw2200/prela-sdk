"""
Assertions for evaluating AI agent outputs.

This module provides various assertion types for testing AI agent behavior:
- Structural: Text matching, regex, length, JSON validation
- Tool: Tool call verification and sequence checking
- Semantic: Embedding-based similarity comparison
"""

from __future__ import annotations

from prela.evals.assertions.base import AssertionResult, BaseAssertion
from prela.evals.assertions.structural import (
    ContainsAssertion,
    JSONValidAssertion,
    LengthAssertion,
    NotContainsAssertion,
    RegexAssertion,
)
from prela.evals.assertions.tool import (
    ToolArgsAssertion,
    ToolCalledAssertion,
    ToolSequenceAssertion,
)
from prela.evals.assertions.multi_agent import (
    AgentCollaborationAssertion,
    AgentUsedAssertion,
    ConversationTurnsAssertion,
    DelegationOccurredAssertion,
    HandoffOccurredAssertion,
    NoCircularDelegationAssertion,
    TaskCompletedAssertion,
)

# Semantic assertions are optional (require sentence-transformers)
try:
    from prela.evals.assertions.semantic import SemanticSimilarityAssertion

    __all__ = [
        "AssertionResult",
        "BaseAssertion",
        "ContainsAssertion",
        "NotContainsAssertion",
        "RegexAssertion",
        "LengthAssertion",
        "JSONValidAssertion",
        "ToolCalledAssertion",
        "ToolArgsAssertion",
        "ToolSequenceAssertion",
        "SemanticSimilarityAssertion",
        "AgentUsedAssertion",
        "TaskCompletedAssertion",
        "DelegationOccurredAssertion",
        "HandoffOccurredAssertion",
        "AgentCollaborationAssertion",
        "ConversationTurnsAssertion",
        "NoCircularDelegationAssertion",
    ]
except ImportError:
    __all__ = [
        "AssertionResult",
        "BaseAssertion",
        "ContainsAssertion",
        "NotContainsAssertion",
        "RegexAssertion",
        "LengthAssertion",
        "JSONValidAssertion",
        "ToolCalledAssertion",
        "ToolArgsAssertion",
        "ToolSequenceAssertion",
        "AgentUsedAssertion",
        "TaskCompletedAssertion",
        "DelegationOccurredAssertion",
        "HandoffOccurredAssertion",
        "AgentCollaborationAssertion",
        "ConversationTurnsAssertion",
        "NoCircularDelegationAssertion",
    ]
