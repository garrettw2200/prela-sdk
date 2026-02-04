"""Shared data models for multi-agent instrumentation.

This module provides common data structures for tracing multi-agent systems
across different frameworks (CrewAI, AutoGen, LangGraph, Swarm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
import hashlib


class AgentRole(str, Enum):
    """Standard agent roles across frameworks."""

    MANAGER = "manager"
    WORKER = "worker"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    USER_PROXY = "user_proxy"
    ASSISTANT = "assistant"
    CUSTOM = "custom"


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    QUESTION = "question"
    ANSWER = "answer"
    FEEDBACK = "feedback"
    DELEGATION = "delegation"
    HANDOFF = "handoff"
    SYSTEM = "system"


@dataclass
class AgentDefinition:
    """Represents an agent in a multi-agent system."""

    agent_id: str
    name: str
    role: AgentRole
    framework: str  # "crewai", "autogen", "langgraph", "swarm"

    # Agent configuration
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: list[str] = field(default_factory=list)

    # Framework-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_span_attributes(self) -> dict[str, Any]:
        """Convert to span attributes for tracing."""
        return {
            "agent.id": self.agent_id,
            "agent.name": self.name,
            "agent.role": self.role.value,
            "agent.framework": self.framework,
            "agent.model": self.model,
            "agent.tools": self.tools,
        }


@dataclass
class AgentMessage:
    """A message between agents."""

    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: str
    timestamp: datetime

    # Optional structured data
    tool_calls: Optional[list[dict]] = None
    tool_results: Optional[list[dict]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_span_event(self) -> dict[str, Any]:
        """Convert to span event for tracing."""
        return {
            "name": f"agent.message.{self.message_type.value}",
            "attributes": {
                "message.id": self.message_id,
                "message.sender": self.sender_id,
                "message.receiver": self.receiver_id,
                "message.type": self.message_type.value,
                "message.content_length": len(self.content),
            },
        }


@dataclass
class TaskAssignment:
    """A task assigned from one agent to another."""

    task_id: str
    assigner_id: str
    assignee_id: str
    description: str
    expected_output: Optional[str] = None

    # Execution tracking
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None

    # Token/cost tracking
    total_tokens: int = 0
    total_cost_usd: float = 0.0


@dataclass
class CrewExecution:
    """Top-level execution of a multi-agent crew/team."""

    execution_id: str
    framework: str

    # Crew configuration
    agents: list[AgentDefinition] = field(default_factory=list)
    tasks: list[TaskAssignment] = field(default_factory=list)

    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: Literal["running", "completed", "failed"] = "running"

    # Aggregated metrics
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0
    total_agent_messages: int = 0


@dataclass
class ConversationTurn:
    """A single turn in a multi-agent conversation (for AutoGen style)."""

    turn_id: str
    turn_number: int
    speaker_id: str
    content: str
    timestamp: datetime

    # If this turn triggered an LLM call
    llm_call_id: Optional[str] = None
    tokens_used: int = 0

    # If this turn triggered tool use
    tool_calls: list[dict] = field(default_factory=list)


# Helper functions


def generate_agent_id(framework: str, name: str) -> str:
    """Generate a consistent agent ID.

    Args:
        framework: The multi-agent framework (e.g., "crewai", "autogen")
        name: The agent name

    Returns:
        A deterministic 12-character hash based on framework and name
    """
    return hashlib.sha256(f"{framework}:{name}".encode()).hexdigest()[:12]


def extract_agent_graph(
    agents: list[AgentDefinition], messages: list[AgentMessage]
) -> dict:
    """Extract agent communication graph for visualization.

    Args:
        agents: List of agent definitions
        messages: List of messages exchanged between agents

    Returns:
        A dictionary with "nodes" and "edges" for graph visualization
    """
    nodes = {a.agent_id: {"name": a.name, "role": a.role.value} for a in agents}
    edges = []
    for msg in messages:
        if msg.receiver_id:
            edges.append(
                {
                    "from": msg.sender_id,
                    "to": msg.receiver_id,
                    "type": msg.message_type.value,
                }
            )
    return {"nodes": nodes, "edges": edges}
