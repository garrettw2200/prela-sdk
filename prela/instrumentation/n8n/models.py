"""
Data models for n8n workflow telemetry.

These models represent n8n workflow executions, node executions, and AI-specific
node executions with comprehensive telemetry data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class N8nSpanType(str, Enum):
    """Span types specific to n8n workflow executions."""

    WORKFLOW = "n8n.workflow"
    NODE = "n8n.node"
    AI_AGENT = "n8n.ai_agent"
    LLM = "n8n.llm"
    TOOL = "n8n.tool"
    RETRIEVAL = "n8n.retrieval"
    MEMORY = "n8n.memory"


class N8nWorkflowExecution(BaseModel):
    """
    Represents a complete n8n workflow execution.

    Captures high-level metadata about a workflow run, including timing,
    status, node counts, and aggregate token/cost metrics.
    """

    workflow_id: str = Field(..., description="Unique identifier for the workflow")
    workflow_name: str = Field(..., description="Human-readable workflow name")
    execution_id: str = Field(
        ..., description="Unique identifier for this execution instance"
    )
    trigger_type: str = Field(
        ...,
        description="How the workflow was triggered (webhook, cron, manual, etc.)",
    )
    started_at: datetime = Field(..., description="When the workflow execution began")
    completed_at: Optional[datetime] = Field(
        None, description="When the workflow execution completed"
    )
    status: Literal["running", "success", "error", "waiting"] = Field(
        ..., description="Current execution status"
    )
    node_count: int = Field(
        ..., ge=0, description="Total number of nodes executed"
    )
    ai_node_count: int = Field(
        0, ge=0, description="Number of AI-related nodes executed"
    )
    total_tokens: int = Field(
        0, ge=0, description="Total tokens consumed across all AI nodes"
    )
    total_cost_usd: float = Field(
        0.0, ge=0.0, description="Total estimated cost in USD"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if status is 'error'"
    )

    @field_validator("node_count", "ai_node_count", "total_tokens")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Ensure counts are non-negative."""
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v

    @field_validator("total_cost_usd")
    @classmethod
    def validate_cost(cls, v: float) -> float:
        """Ensure cost is non-negative."""
        if v < 0.0:
            raise ValueError("Cost must be non-negative")
        return v

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(
        cls, v: Optional[datetime], info
    ) -> Optional[datetime]:
        """Ensure completed_at is after started_at if both exist."""
        if v is not None and "started_at" in info.data:
            started_at = info.data["started_at"]
            if v < started_at:
                raise ValueError("completed_at must be after started_at")
        return v

    def duration_ms(self) -> Optional[float]:
        """Calculate execution duration in milliseconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    def to_span_attributes(self) -> dict:
        """Convert to Prela span attributes."""
        attrs = {
            "n8n.workflow_id": self.workflow_id,
            "n8n.workflow_name": self.workflow_name,
            "n8n.execution_id": self.execution_id,
            "n8n.trigger_type": self.trigger_type,
            "n8n.status": self.status,
            "n8n.node_count": self.node_count,
            "n8n.ai_node_count": self.ai_node_count,
            "n8n.total_tokens": self.total_tokens,
            "n8n.total_cost_usd": self.total_cost_usd,
        }
        if self.error_message:
            attrs["n8n.error_message"] = self.error_message
        duration = self.duration_ms()
        if duration is not None:
            attrs["n8n.duration_ms"] = duration
        return attrs


class N8nNodeExecution(BaseModel):
    """
    Represents execution of a single node within a workflow.

    Captures node-level telemetry including input/output data, timing,
    status, and error information.
    """

    node_id: str = Field(..., description="Unique identifier for the node")
    node_name: str = Field(..., description="Human-readable node name")
    node_type: str = Field(
        ...,
        description="Node type identifier (e.g., 'n8n-nodes-langchain.agent')",
    )
    execution_index: int = Field(
        ..., ge=0, description="Order of execution within workflow"
    )
    started_at: datetime = Field(..., description="When node execution began")
    completed_at: Optional[datetime] = Field(
        None, description="When node execution completed"
    )
    status: Literal["success", "error"] = Field(
        ..., description="Node execution status"
    )
    input_data: Optional[dict] = Field(
        None, description="JSON representation of input items"
    )
    output_data: Optional[dict] = Field(
        None, description="JSON representation of output items"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if status is 'error'"
    )
    retry_count: int = Field(0, ge=0, description="Number of retry attempts")

    @field_validator("execution_index", "retry_count")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Ensure counts are non-negative."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(
        cls, v: Optional[datetime], info
    ) -> Optional[datetime]:
        """Ensure completed_at is after started_at if both exist."""
        if v is not None and "started_at" in info.data:
            started_at = info.data["started_at"]
            if v < started_at:
                raise ValueError("completed_at must be after started_at")
        return v

    def duration_ms(self) -> Optional[float]:
        """Calculate node execution duration in milliseconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    def to_span_attributes(self) -> dict:
        """Convert to Prela span attributes."""
        attrs = {
            "n8n.node_id": self.node_id,
            "n8n.node_name": self.node_name,
            "n8n.node_type": self.node_type,
            "n8n.execution_index": self.execution_index,
            "n8n.status": self.status,
            "n8n.retry_count": self.retry_count,
        }
        if self.error_message:
            attrs["n8n.error_message"] = self.error_message
        if self.input_data:
            attrs["n8n.input_data"] = str(self.input_data)[:500]  # Truncate
        if self.output_data:
            attrs["n8n.output_data"] = str(self.output_data)[:500]  # Truncate
        duration = self.duration_ms()
        if duration is not None:
            attrs["n8n.duration_ms"] = duration
        return attrs


class N8nAINodeExecution(N8nNodeExecution):
    """
    Represents execution of an AI-specific node (LLM, agent, vector store, etc.).

    Extends N8nNodeExecution with AI-specific telemetry including model info,
    token usage, costs, prompts, and retrieval data.
    """

    model: Optional[str] = Field(
        None, description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')"
    )
    provider: Optional[str] = Field(
        None, description="AI provider (openai, anthropic, ollama, etc.)"
    )
    prompt_tokens: int = Field(0, ge=0, description="Tokens in the prompt")
    completion_tokens: int = Field(0, ge=0, description="Tokens in the completion")
    total_tokens: int = Field(0, ge=0, description="Total tokens consumed")
    cost_usd: float = Field(0.0, ge=0.0, description="Estimated cost in USD")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Temperature parameter"
    )
    system_prompt: Optional[str] = Field(None, description="System prompt text")
    user_prompt: Optional[str] = Field(None, description="User prompt text")
    response_content: Optional[str] = Field(None, description="LLM response content")
    tool_calls: Optional[list[dict]] = Field(
        None, description="List of tool calls made by the LLM"
    )
    retrieval_query: Optional[str] = Field(
        None, description="Query used for vector store retrieval"
    )
    retrieved_documents: Optional[list[dict]] = Field(
        None, description="Documents retrieved from vector store"
    )

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens")
    @classmethod
    def validate_non_negative_tokens(cls, v: int) -> int:
        """Ensure token counts are non-negative."""
        if v < 0:
            raise ValueError("Token count must be non-negative")
        return v

    @field_validator("cost_usd")
    @classmethod
    def validate_cost(cls, v: float) -> float:
        """Ensure cost is non-negative."""
        if v < 0.0:
            raise ValueError("Cost must be non-negative")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        """Ensure temperature is in valid range."""
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    def to_span_attributes(self) -> dict:
        """Convert to Prela span attributes with AI-specific fields."""
        # Start with base node attributes
        attrs = super().to_span_attributes()

        # Add AI-specific attributes
        if self.model:
            attrs["llm.model"] = self.model
        if self.provider:
            attrs["llm.provider"] = self.provider
        if self.prompt_tokens:
            attrs["llm.prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            attrs["llm.completion_tokens"] = self.completion_tokens
        if self.total_tokens:
            attrs["llm.total_tokens"] = self.total_tokens
        if self.cost_usd:
            attrs["llm.cost_usd"] = self.cost_usd
        if self.temperature is not None:
            attrs["llm.temperature"] = self.temperature
        if self.system_prompt:
            attrs["llm.system_prompt"] = self.system_prompt[:500]  # Truncate
        if self.user_prompt:
            attrs["llm.user_prompt"] = self.user_prompt[:500]  # Truncate
        if self.response_content:
            attrs["llm.response_content"] = self.response_content[:500]  # Truncate
        if self.tool_calls:
            attrs["llm.tool_calls_count"] = len(self.tool_calls)
            attrs["llm.tool_calls"] = str(self.tool_calls)[:500]  # Truncate
        if self.retrieval_query:
            attrs["retrieval.query"] = self.retrieval_query[:200]  # Truncate
        if self.retrieved_documents:
            attrs["retrieval.document_count"] = len(self.retrieved_documents)
            attrs["retrieval.documents"] = str(self.retrieved_documents)[
                :500
            ]  # Truncate

        return attrs

    def infer_span_type(self) -> N8nSpanType:
        """
        Infer the appropriate span type based on node characteristics.

        Returns:
            N8nSpanType appropriate for this node's function
        """
        node_type_lower = self.node_type.lower()

        # Check for specific node types
        if "agent" in node_type_lower:
            return N8nSpanType.AI_AGENT
        elif any(
            x in node_type_lower
            for x in ["chat", "llm", "openai", "anthropic", "ollama"]
        ):
            return N8nSpanType.LLM
        elif any(
            x in node_type_lower for x in ["tool", "function", "code"]
        ):
            return N8nSpanType.TOOL
        elif any(
            x in node_type_lower
            for x in ["vector", "retrieval", "search", "pinecone", "qdrant"]
        ):
            return N8nSpanType.RETRIEVAL
        elif any(x in node_type_lower for x in ["memory", "buffer", "history"]):
            return N8nSpanType.MEMORY
        else:
            # Default to generic node type
            return N8nSpanType.NODE
