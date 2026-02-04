"""Multi-agent framework instrumentation for Prela.

This module provides instrumentation for multi-agent frameworks including:
- CrewAI: Task-based multi-agent orchestration
- AutoGen: Conversational multi-agent framework
- LangGraph: Graph-based multi-agent workflows
- Swarm: OpenAI's experimental multi-agent framework

Each instrumentor captures agent definitions, inter-agent messages, task assignments,
and conversation turns to provide complete observability for multi-agent systems.
"""

from prela.instrumentation.multi_agent.models import (
    AgentDefinition,
    AgentMessage,
    AgentRole,
    ConversationTurn,
    CrewExecution,
    MessageType,
    TaskAssignment,
    extract_agent_graph,
    generate_agent_id,
)

# Instrumentors
from prela.instrumentation.multi_agent.autogen import AutoGenInstrumentor
from prela.instrumentation.multi_agent.crewai import CrewAIInstrumentor
from prela.instrumentation.multi_agent.langgraph import LangGraphInstrumentor
from prela.instrumentation.multi_agent.swarm import SwarmInstrumentor

__all__ = [
    # Data models
    "AgentDefinition",
    "AgentMessage",
    "TaskAssignment",
    "CrewExecution",
    "ConversationTurn",
    "AgentRole",
    "MessageType",
    # Helper functions
    "generate_agent_id",
    "extract_agent_graph",
    # Instrumentors
    "AutoGenInstrumentor",
    "CrewAIInstrumentor",
    "LangGraphInstrumentor",
    "SwarmInstrumentor",
]
