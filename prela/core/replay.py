"""Replay capture for deterministic re-execution of AI agent workflows."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Storage size warning threshold (100 KB)
REPLAY_SIZE_WARNING_THRESHOLD = 100 * 1024


class ReplaySnapshot:
    """Complete replay-enabling data for a span.

    This class holds all information needed to deterministically replay
    an agent execution, including full request/response data, tool I/O,
    retrieval results, and agent state.

    Different span types populate different fields:
    - LLM spans: llm_request, llm_response, llm_streaming_chunks, model_info
    - Tool spans: tool_name, tool_input, tool_output, has_side_effects
    - Retrieval spans: retrieval_query, retrieved_documents, retrieval_metadata
    - Agent spans: system_prompt, available_tools, agent_memory, agent_config

    Memory efficiency: Uses __slots__ to minimize per-instance overhead.
    """

    __slots__ = (
        # LLM fields
        "llm_request",
        "llm_response",
        "llm_streaming_chunks",
        "model_info",
        "request_timestamp",
        # Tool fields
        "tool_name",
        "tool_description",
        "tool_input",
        "tool_output",
        "has_side_effects",
        # Retrieval fields
        "retrieval_query",
        "retrieved_documents",
        "retrieval_scores",
        "retrieval_metadata",
        # Agent fields
        "system_prompt",
        "available_tools",
        "agent_memory",
        "agent_config",
    )

    def __init__(
        self,
        # LLM fields
        llm_request: dict[str, Any] | None = None,
        llm_response: dict[str, Any] | None = None,
        llm_streaming_chunks: list[dict[str, Any]] | None = None,
        model_info: dict[str, Any] | None = None,
        request_timestamp: datetime | None = None,
        # Tool fields
        tool_name: str | None = None,
        tool_description: str | None = None,
        tool_input: dict[str, Any] | str | None = None,
        tool_output: Any = None,
        has_side_effects: bool = True,
        # Retrieval fields
        retrieval_query: str | None = None,
        retrieved_documents: list[dict[str, Any]] | None = None,
        retrieval_scores: list[float] | None = None,
        retrieval_metadata: dict[str, Any] | None = None,
        # Agent fields
        system_prompt: str | None = None,
        available_tools: list[dict[str, Any]] | None = None,
        agent_memory: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize replay snapshot with optional fields."""
        self.llm_request = llm_request
        self.llm_response = llm_response
        self.llm_streaming_chunks = llm_streaming_chunks
        self.model_info = model_info
        self.request_timestamp = request_timestamp
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_input = tool_input
        self.tool_output = tool_output
        self.has_side_effects = has_side_effects
        self.retrieval_query = retrieval_query
        self.retrieved_documents = retrieved_documents
        self.retrieval_scores = retrieval_scores
        self.retrieval_metadata = retrieval_metadata
        self.system_prompt = system_prompt
        self.available_tools = available_tools
        self.agent_memory = agent_memory
        self.agent_config = agent_config

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Returns:
            Dictionary containing all non-None fields
        """
        result = {}

        for field_name in self.__slots__:
            value = getattr(self, field_name)
            if value is not None:
                # Handle datetime serialization
                if isinstance(value, datetime):
                    result[field_name] = value.isoformat()
                else:
                    result[field_name] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplaySnapshot:
        """Deserialize from dict.

        Args:
            data: Dictionary from to_dict()

        Returns:
            ReplaySnapshot instance
        """
        # Convert ISO timestamp back to datetime
        if "request_timestamp" in data and isinstance(data["request_timestamp"], str):
            data["request_timestamp"] = datetime.fromisoformat(data["request_timestamp"])

        return cls(**data)

    def estimate_size_bytes(self) -> int:
        """Estimate storage size in bytes.

        This is an approximation based on JSON serialization size.
        Useful for monitoring storage costs.

        Logs a warning if size exceeds 100 KB threshold.

        Returns:
            Estimated size in bytes
        """
        serialized = json.dumps(self.to_dict())
        size_bytes = len(serialized.encode("utf-8"))

        # Warn if exceeds threshold
        if size_bytes > REPLAY_SIZE_WARNING_THRESHOLD:
            logger.warning(
                f"Replay snapshot size ({size_bytes / 1024:.1f} KB) exceeds "
                f"recommended threshold ({REPLAY_SIZE_WARNING_THRESHOLD / 1024:.0f} KB). "
                f"Consider reducing captured data or increasing storage budget."
            )

        return size_bytes


class ReplayCapture:
    """Helper for building ReplaySnapshot during span execution.

    This class provides a builder-style API for incrementally capturing
    replay data as a span executes.

    Example:
        ```python
        capture = ReplayCapture()
        capture.set_llm_request(model="gpt-4", messages=[...])
        capture.set_llm_response(text="...", tokens=100)
        snapshot = capture.build()
        ```
    """

    def __init__(self) -> None:
        """Initialize empty capture."""
        self._snapshot = ReplaySnapshot()

    # LLM capture methods
    def set_llm_request(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture LLM request details.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-sonnet-4")
            messages: Chat messages (OpenAI/Anthropic format)
            prompt: Single prompt string (legacy completions)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        request: dict[str, Any] = {"model": model}

        if messages is not None:
            request["messages"] = messages
        if prompt is not None:
            request["prompt"] = prompt
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        # Capture all other kwargs (top_p, frequency_penalty, etc.)
        request.update(kwargs)

        self._snapshot.llm_request = request
        self._snapshot.request_timestamp = datetime.now(timezone.utc)

    def set_llm_response(
        self,
        text: str,
        finish_reason: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture LLM response details.

        Args:
            text: Complete response text
            finish_reason: Why generation stopped (stop, length, tool_calls)
            model: Actual model used (may differ from requested)
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            **kwargs: Additional response metadata
        """
        response: dict[str, Any] = {"text": text}

        if finish_reason is not None:
            response["finish_reason"] = finish_reason
        if model is not None:
            response["model"] = model
        if prompt_tokens is not None:
            response["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            response["completion_tokens"] = completion_tokens

        response.update(kwargs)
        self._snapshot.llm_response = response

    def add_streaming_chunk(
        self,
        chunk: dict[str, Any],
    ) -> None:
        """Add a streaming chunk to the replay data.

        For streaming LLM responses, each delta/chunk is captured separately
        to enable exact replay of streaming behavior.

        Args:
            chunk: Chunk data (provider-specific format)
        """
        if self._snapshot.llm_streaming_chunks is None:
            self._snapshot.llm_streaming_chunks = []

        self._snapshot.llm_streaming_chunks.append(chunk)

    def set_model_info(self, **info: Any) -> None:
        """Capture model version/endpoint info.

        Args:
            **info: Model metadata (version, endpoint, created timestamp, etc.)
        """
        self._snapshot.model_info = info

    # Tool capture methods
    def set_tool_call(
        self,
        name: str,
        description: str | None = None,
        input_args: dict[str, Any] | str | None = None,
        output: Any = None,
        has_side_effects: bool = True,  # SAFE DEFAULT
    ) -> None:
        """Capture tool call details.

        Args:
            name: Tool name
            description: Tool description
            input_args: Input arguments (dict or JSON string)
            output: Tool output/return value
            has_side_effects: Whether tool modifies external state (default: True)
        """
        self._snapshot.tool_name = name
        self._snapshot.tool_description = description
        self._snapshot.tool_input = input_args
        self._snapshot.tool_output = output
        self._snapshot.has_side_effects = has_side_effects

    # Retrieval capture methods
    def set_retrieval(
        self,
        query: str,
        documents: list[dict[str, Any]],
        scores: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Capture retrieval operation details.

        Args:
            query: Query text
            documents: Retrieved documents (full content)
            scores: Similarity scores for each document
            metadata: Retrieval metadata (index name, collection, etc.)
        """
        self._snapshot.retrieval_query = query
        self._snapshot.retrieved_documents = documents
        self._snapshot.retrieval_scores = scores
        self._snapshot.retrieval_metadata = metadata

    # Agent capture methods
    def set_agent_context(
        self,
        system_prompt: str | None = None,
        available_tools: list[dict[str, Any]] | None = None,
        memory: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Capture agent context and configuration.

        Args:
            system_prompt: System/instruction prompt
            available_tools: List of tools with schemas
            memory: Agent memory/context state
            config: Agent configuration
        """
        if system_prompt is not None:
            self._snapshot.system_prompt = system_prompt
        if available_tools is not None:
            self._snapshot.available_tools = available_tools
        if memory is not None:
            self._snapshot.agent_memory = memory
        if config is not None:
            self._snapshot.agent_config = config

    def build(self) -> ReplaySnapshot:
        """Return the completed snapshot.

        Returns:
            ReplaySnapshot with all captured data
        """
        return self._snapshot


def estimate_replay_storage(
    span: Any,  # Span type (avoid circular import)
    replay_snapshot: ReplaySnapshot | None = None,
) -> int:
    """Estimate total storage size for span with replay data.

    Args:
        span: The span to estimate (must have .to_dict() method)
        replay_snapshot: Optional replay snapshot (if not attached to span)

    Returns:
        Estimated size in bytes
    """
    # Base span size
    span_dict = span.to_dict()
    base_size = len(json.dumps(span_dict).encode("utf-8"))

    # Replay data size
    replay_size = 0
    if replay_snapshot is not None:
        replay_size = replay_snapshot.estimate_size_bytes()
    elif hasattr(span, "replay_snapshot") and span.replay_snapshot is not None:
        replay_size = span.replay_snapshot.estimate_size_bytes()

    return base_size + replay_size


def serialize_replay_data(value: Any) -> Any:
    """Serialize arbitrary Python values for replay storage.

    Handles common types that may appear in tool I/O or agent state.

    Args:
        value: Value to serialize

    Returns:
        JSON-compatible value
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (list, tuple)):
        return [serialize_replay_data(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_replay_data(v) for k, v in value.items()}
    else:
        # Fallback: convert to string representation
        return str(value)
