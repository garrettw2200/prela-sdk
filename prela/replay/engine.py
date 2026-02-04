"""Replay execution engine for deterministic trace re-execution."""

from __future__ import annotations

import logging
import threading
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from prela.core.span import Span, SpanType
from prela.replay.loader import Trace
from prela.replay.result import ReplayResult, ReplayedSpan

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Thread-local storage for retry counts
_retry_context = threading.local()


def _is_retryable_error(error: Exception) -> bool:
    """Check if API error is retryable.

    Args:
        error: Exception from API call

    Returns:
        True if error is likely transient and should be retried
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Retryable error patterns
    retryable_patterns = [
        "rate limit",
        "429",  # HTTP 429 Too Many Requests
        "503",  # HTTP 503 Service Unavailable
        "502",  # HTTP 502 Bad Gateway
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "service unavailable",
        "try again",
        "overloaded",
    ]

    # Check error message
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True

    # Check error type
    retryable_types = [
        "timeout",
        "connectionerror",
        "httpstatuserror",  # httpx
    ]

    for error_type_pattern in retryable_types:
        if error_type_pattern in error_type:
            return True

    return False


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for API calls with exponential backoff retry logic.

    Stores retry count in thread-local storage accessible via _get_last_retry_count().

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries)
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries (cap for exponential backoff)
        exponential_base: Base for exponential backoff (2.0 = double each retry)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            retry_count = 0

            # Reset retry count in thread-local storage
            _retry_context.retry_count = 0

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Store final retry count
                    _retry_context.retry_count = retry_count
                    return result
                except Exception as e:
                    last_exception = e

                    # Check if retryable
                    if not _is_retryable_error(e):
                        logger.debug(f"Non-retryable error, not retrying: {e}")
                        _retry_context.retry_count = 0
                        raise

                    # Last attempt, don't retry
                    if attempt == max_retries:
                        logger.error(
                            f"API call failed after {max_retries + 1} attempts: {e}"
                        )
                        _retry_context.retry_count = retry_count
                        raise

                    # Increment retry count
                    retry_count += 1

                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base**attempt),
                        max_delay,
                    )

                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

            # Should never reach here, but for type safety
            _retry_context.retry_count = retry_count
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic failed unexpectedly")

        return wrapper

    return decorator


def _get_last_retry_count() -> int:
    """Get retry count from last API call in current thread.

    Returns:
        Number of retries (0 if no retries or not set)
    """
    return getattr(_retry_context, "retry_count", 0)


class ReplayEngine:
    """Engine for replaying traces with exact or modified parameters.

    Supports:
    - Exact replay: Use captured data, no API calls
    - Modified replay: Change parameters, make real API calls for modified spans
    """

    def __init__(
        self,
        trace: Trace,
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        retry_exponential_base: float = 2.0,
    ) -> None:
        """Initialize replay engine with a trace.

        Args:
            trace: Trace to replay
            max_retries: Maximum retry attempts for API calls (default: 3)
            retry_initial_delay: Initial delay before first retry in seconds (default: 1.0)
            retry_max_delay: Maximum delay between retries in seconds (default: 60.0)
            retry_exponential_base: Base for exponential backoff (default: 2.0)

        Raises:
            ValueError: If trace lacks replay data
        """
        self.trace = trace
        self.max_retries = max_retries
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self.retry_exponential_base = retry_exponential_base

        # Validate trace has replay data
        if not trace.has_replay_data():
            raise ValueError(
                "Trace does not contain replay data. "
                "Enable capture_for_replay=True when creating traces."
            )

        is_complete, missing = trace.validate_replay_completeness()
        if not is_complete:
            logger.warning(
                f"Trace has incomplete replay data. Missing snapshots for: {', '.join(missing[:5])}"
            )

    def replay_exact(self) -> ReplayResult:
        """Replay trace using all captured data without making API calls.

        Returns:
            ReplayResult with all spans executed using captured data
        """
        result = ReplayResult(trace_id=self.trace.trace_id)

        # Walk trace in execution order
        for span in self.trace.walk_depth_first():
            replayed_span = self._replay_span_exact(span)
            result.spans.append(replayed_span)

            # Aggregate metrics
            result.total_duration_ms += replayed_span.duration_ms
            result.total_tokens += replayed_span.tokens_used
            result.total_cost_usd += replayed_span.cost_usd

            if replayed_span.error:
                result.errors.append(f"{span.name}: {replayed_span.error}")

        # Extract final output from last root span
        if self.trace.root_spans:
            last_root = self.trace.root_spans[-1]
            result.final_output = self._extract_output(last_root)

        return result

    def replay_with_modifications(
        self,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        mock_tool_responses: dict[str, Any] | None = None,
        mock_retrieval_results: list[dict[str, Any]] | None = None,
        enable_tool_execution: bool = False,
        tool_execution_allowlist: list[str] | None = None,
        tool_execution_blocklist: list[str] | None = None,
        tool_registry: dict[str, Any] | None = None,
        enable_retrieval_execution: bool = False,
        retrieval_client: Any | None = None,
        retrieval_query_override: str | None = None,
        stream: bool = False,
        stream_callback: Any = None,
    ) -> ReplayResult:
        """Replay trace with specified modifications.

        For modified LLM spans, makes real API calls.
        For unmodified spans, uses captured data.

        Args:
            model: Override model for LLM spans
            temperature: Override temperature for LLM spans
            system_prompt: Override system prompt for LLM spans
            max_tokens: Override max_tokens for LLM spans
            mock_tool_responses: Override tool outputs by tool name
            mock_retrieval_results: Override retrieval results
            enable_tool_execution: If True, re-execute tools instead of using cached data
            tool_execution_allowlist: Only execute tools in this list (if provided)
            tool_execution_blocklist: Never execute tools in this list
            tool_registry: Dictionary mapping tool names to callable functions
            enable_retrieval_execution: If True, re-query vector database
            retrieval_client: Vector database client (ChromaDB, Pinecone, etc.)
            retrieval_query_override: Override query for retrieval spans
            stream: If True, use streaming API for modified LLM spans
            stream_callback: Optional callback for streaming chunks (chunk_text: str) -> None

        Returns:
            ReplayResult with modifications applied
        """
        result = ReplayResult(
            trace_id=self.trace.trace_id,
            modifications_applied={
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "mock_tool_responses": list(mock_tool_responses.keys())
                if mock_tool_responses
                else [],
                "mock_retrieval_results": len(mock_retrieval_results)
                if mock_retrieval_results
                else 0,
                "enable_tool_execution": enable_tool_execution,
                "tool_execution_allowlist": tool_execution_allowlist,
                "tool_execution_blocklist": tool_execution_blocklist,
                "enable_retrieval_execution": enable_retrieval_execution,
                "retrieval_query_override": retrieval_query_override,
            },
        )

        # Walk trace in execution order
        for span in self.trace.walk_depth_first():
            # Determine if this span needs modification
            needs_modification = self._span_needs_modification(
                span,
                model,
                temperature,
                system_prompt,
                max_tokens,
                mock_tool_responses,
                mock_retrieval_results,
                enable_tool_execution,
                enable_retrieval_execution,
                retrieval_query_override,
            )

            if needs_modification:
                replayed_span = self._replay_span_modified(
                    span,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    mock_tool_responses=mock_tool_responses,
                    mock_retrieval_results=mock_retrieval_results,
                    enable_tool_execution=enable_tool_execution,
                    tool_execution_allowlist=tool_execution_allowlist,
                    tool_execution_blocklist=tool_execution_blocklist,
                    tool_registry=tool_registry,
                    enable_retrieval_execution=enable_retrieval_execution,
                    retrieval_client=retrieval_client,
                    retrieval_query_override=retrieval_query_override,
                    stream=stream,
                    stream_callback=stream_callback,
                )
            else:
                replayed_span = self._replay_span_exact(span)

            result.spans.append(replayed_span)

            # Aggregate metrics
            result.total_duration_ms += replayed_span.duration_ms
            result.total_tokens += replayed_span.tokens_used
            result.total_cost_usd += replayed_span.cost_usd

            if replayed_span.error:
                result.errors.append(f"{span.name}: {replayed_span.error}")

        # Extract final output
        if self.trace.root_spans:
            last_root = self.trace.root_spans[-1]
            result.final_output = self._extract_output(last_root)

        return result

    def _replay_span_exact(self, span: Span) -> ReplayedSpan:
        """Replay span using captured data.

        Args:
            span: Span to replay

        Returns:
            ReplayedSpan with captured output
        """
        if span.replay_snapshot is None:
            return ReplayedSpan(
                original_span_id=span.span_id,
                span_type=span.span_type.value,
                name=span.name,
                input=None,
                output=None,
                error="No replay data available",
            )

        snapshot = span.replay_snapshot

        # Extract input and output based on span type
        if span.span_type == SpanType.LLM:
            input_data = snapshot.llm_request
            output_data = snapshot.llm_response.get("text") if snapshot.llm_response else None
            tokens = (
                snapshot.llm_response.get("prompt_tokens", 0)
                + snapshot.llm_response.get("completion_tokens", 0)
                if snapshot.llm_response
                else 0
            )
            cost = self._estimate_cost(
                snapshot.llm_request.get("model") if snapshot.llm_request else None, tokens
            )
        elif span.span_type == SpanType.TOOL:
            input_data = snapshot.tool_input
            output_data = snapshot.tool_output
            tokens = 0
            cost = 0.0
        elif span.span_type == SpanType.RETRIEVAL:
            input_data = snapshot.retrieval_query
            output_data = snapshot.retrieved_documents
            tokens = 0
            cost = 0.0
        elif span.span_type == SpanType.AGENT:
            input_data = snapshot.agent_memory
            output_data = snapshot.agent_config
            tokens = 0
            cost = 0.0
        else:
            input_data = None
            output_data = None
            tokens = 0
            cost = 0.0

        duration_ms = span.duration_ms if span.ended_at else 0.0

        return ReplayedSpan(
            original_span_id=span.span_id,
            span_type=span.span_type.value,
            name=span.name,
            input=input_data,
            output=output_data,
            was_modified=False,
            duration_ms=duration_ms,
            tokens_used=tokens,
            cost_usd=cost,
        )

    def _replay_span_modified(
        self,
        span: Span,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        mock_tool_responses: dict[str, Any] | None = None,
        mock_retrieval_results: list[dict[str, Any]] | None = None,
        enable_tool_execution: bool = False,
        tool_execution_allowlist: list[str] | None = None,
        tool_execution_blocklist: list[str] | None = None,
        tool_registry: dict[str, Any] | None = None,
        enable_retrieval_execution: bool = False,
        retrieval_client: Any | None = None,
        retrieval_query_override: str | None = None,
        stream: bool = False,
        stream_callback: Any = None,
    ) -> ReplayedSpan:
        """Replay span with modifications.

        For LLM spans with param changes, makes real API call.
        For tool/retrieval with mocks, uses mock data.
        For tool/retrieval with execution enabled, re-executes.

        Args:
            span: Span to replay
            model: Override model
            temperature: Override temperature
            system_prompt: Override system prompt
            max_tokens: Override max_tokens
            mock_tool_responses: Mock tool responses by name
            mock_retrieval_results: Mock retrieval documents
            enable_tool_execution: If True, re-execute tools
            tool_execution_allowlist: Only execute tools in this list
            tool_execution_blocklist: Never execute tools in this list
            tool_registry: Dictionary mapping tool names to callables
            enable_retrieval_execution: If True, re-query vector database
            retrieval_client: Vector database client
            retrieval_query_override: Override query for retrieval

        Returns:
            ReplayedSpan with modified output
        """
        if span.replay_snapshot is None:
            return ReplayedSpan(
                original_span_id=span.span_id,
                span_type=span.span_type.value,
                name=span.name,
                input=None,
                output=None,
                error="No replay data available",
            )

        snapshot = span.replay_snapshot
        modifications = []

        # Handle LLM spans
        if span.span_type == SpanType.LLM:
            # Build modified request
            modified_request = dict(snapshot.llm_request) if snapshot.llm_request else {}

            if model is not None:
                modified_request["model"] = model
                modifications.append(f"model={model}")
            if temperature is not None:
                modified_request["temperature"] = temperature
                modifications.append(f"temperature={temperature}")
            if max_tokens is not None:
                modified_request["max_tokens"] = max_tokens
                modifications.append(f"max_tokens={max_tokens}")
            if system_prompt is not None:
                # Update system prompt in messages
                if "messages" in modified_request:
                    messages = modified_request["messages"]
                    # Add or update system message
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                modifications.append("system_prompt=<modified>")

            # Make real API call
            try:
                output_data, tokens, cost = self._call_llm_api(
                    modified_request, stream=stream, stream_callback=stream_callback
                )
                duration_ms = 0.0  # TODO: Measure actual duration
                error = None
                retry_count = _get_last_retry_count()
            except Exception as e:
                logger.error(f"API call failed for {span.name}: {e}")
                output_data = None
                tokens = 0
                cost = 0.0
                duration_ms = 0.0
                error = str(e)
                retry_count = _get_last_retry_count()

            return ReplayedSpan(
                original_span_id=span.span_id,
                span_type=span.span_type.value,
                name=span.name,
                input=modified_request,
                output=output_data,
                was_modified=True,
                modification_details=", ".join(modifications),
                duration_ms=duration_ms,
                tokens_used=tokens,
                cost_usd=cost,
                error=error,
                retry_count=retry_count,
            )

        # Handle tool spans with mocks or execution
        elif span.span_type == SpanType.TOOL:
            # Priority 1: Mock responses
            if mock_tool_responses and snapshot.tool_name in mock_tool_responses:
                output_data = mock_tool_responses[snapshot.tool_name]
                modifications.append(f"mocked_output")
                error = None
            # Priority 2: Tool execution
            elif enable_tool_execution:
                try:
                    output_data = self._execute_tool_safely(
                        snapshot.tool_name,
                        snapshot.tool_input,
                        tool_execution_allowlist,
                        tool_execution_blocklist,
                        tool_registry,
                    )
                    modifications.append("tool_executed")
                    error = None
                except Exception as e:
                    logger.error(f"Tool execution failed for {snapshot.tool_name}: {e}")
                    output_data = None
                    error = str(e)
            # Priority 3: Cached data
            else:
                output_data = snapshot.tool_output
                error = None

            return ReplayedSpan(
                original_span_id=span.span_id,
                span_type=span.span_type.value,
                name=span.name,
                input=snapshot.tool_input,
                output=output_data,
                was_modified=len(modifications) > 0,
                modification_details=", ".join(modifications) if modifications else None,
                duration_ms=span.duration_ms if span.ended_at else 0.0,
                error=error,
            )

        # Handle retrieval spans with mocks or execution
        elif span.span_type == SpanType.RETRIEVAL:
            # Priority 1: Mock results
            if mock_retrieval_results is not None:
                output_data = mock_retrieval_results
                modifications.append("mocked_documents")
                error = None
            # Priority 2: Retrieval execution
            elif enable_retrieval_execution and retrieval_client is not None:
                try:
                    query = retrieval_query_override if retrieval_query_override else snapshot.retrieval_query
                    output_data = self._execute_retrieval(
                        query,
                        retrieval_client,
                        snapshot,
                    )
                    modifications.append("retrieval_executed")
                    if retrieval_query_override:
                        modifications.append(f"query_overridden")
                    error = None
                except Exception as e:
                    logger.error(f"Retrieval execution failed: {e}")
                    output_data = None
                    error = str(e)
            # Priority 3: Cached data
            else:
                output_data = snapshot.retrieved_documents
                error = None

            return ReplayedSpan(
                original_span_id=span.span_id,
                span_type=span.span_type.value,
                name=span.name,
                input=retrieval_query_override if retrieval_query_override else snapshot.retrieval_query,
                output=output_data,
                was_modified=len(modifications) > 0,
                modification_details=", ".join(modifications) if modifications else None,
                duration_ms=span.duration_ms if span.ended_at else 0.0,
                error=error,
            )

        # Default: use exact replay
        return self._replay_span_exact(span)

    def _span_needs_modification(
        self,
        span: Span,
        model: str | None,
        temperature: float | None,
        system_prompt: str | None,
        max_tokens: int | None,
        mock_tool_responses: dict[str, Any] | None,
        mock_retrieval_results: list[dict[str, Any]] | None,
        enable_tool_execution: bool,
        enable_retrieval_execution: bool,
        retrieval_query_override: str | None,
    ) -> bool:
        """Check if span needs modification.

        Args:
            span: Span to check
            model: Model override
            temperature: Temperature override
            system_prompt: System prompt override
            max_tokens: Max tokens override
            mock_tool_responses: Tool response mocks
            mock_retrieval_results: Retrieval result mocks
            enable_tool_execution: If True, tools should be re-executed
            enable_retrieval_execution: If True, retrieval should be re-executed
            retrieval_query_override: If set, retrieval query should be overridden

        Returns:
            True if span should be modified
        """
        if span.span_type == SpanType.LLM:
            return any([model, temperature, system_prompt, max_tokens])
        elif span.span_type == SpanType.TOOL:
            if mock_tool_responses and span.replay_snapshot:
                return span.replay_snapshot.tool_name in mock_tool_responses
            return enable_tool_execution
        elif span.span_type == SpanType.RETRIEVAL:
            return any([mock_retrieval_results is not None, enable_retrieval_execution, retrieval_query_override])

        return False

    def _execute_tool_safely(
        self,
        tool_name: str,
        tool_input: Any,
        allowlist: list[str] | None,
        blocklist: list[str] | None,
        tool_registry: dict[str, Any] | None,
    ) -> Any:
        """Execute tool with safety checks.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input data for tool
            allowlist: Only execute tools in this list (if provided)
            blocklist: Never execute tools in this list
            tool_registry: Dictionary mapping tool names to callable functions

        Returns:
            Tool output

        Raises:
            ValueError: If tool is blocked, not in allowlist, or not found in registry
            Exception: If tool execution fails
        """
        # Check blocklist
        if blocklist and tool_name in blocklist:
            raise ValueError(f"Tool '{tool_name}' is blocked from execution")

        # Check allowlist
        if allowlist and tool_name not in allowlist:
            raise ValueError(f"Tool '{tool_name}' not in allowlist")

        # Find tool function from registry
        if not tool_registry:
            raise ValueError("tool_registry is required for tool execution")

        tool_func = tool_registry.get(tool_name)
        if not tool_func:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        # Execute tool
        logger.debug(f"Executing tool '{tool_name}' with input: {tool_input}")
        return tool_func(tool_input)

    def _execute_retrieval(
        self,
        query: str,
        client: Any,
        snapshot: Any,
    ) -> list[dict[str, Any]]:
        """Execute retrieval query against vector database.

        Args:
            query: Search query
            client: Vector database client (ChromaDB, Pinecone, Qdrant, Weaviate)
            snapshot: Original span snapshot (for metadata like top_k)

        Returns:
            List of retrieved documents with text and scores

        Raises:
            ValueError: If client type is not supported
            Exception: If retrieval fails
        """
        # Detect client type
        client_type = self._detect_retrieval_client(client)

        # Extract metadata from snapshot
        top_k = getattr(snapshot, "similarity_top_k", 5)

        logger.debug(f"Executing retrieval with query: {query}, client: {client_type}, top_k: {top_k}")

        if client_type == "chromadb":
            return self._query_chromadb(client, query, top_k)
        elif client_type == "pinecone":
            return self._query_pinecone(client, query, top_k)
        elif client_type == "qdrant":
            return self._query_qdrant(client, query, top_k)
        elif client_type == "weaviate":
            return self._query_weaviate(client, query, top_k)
        else:
            raise ValueError(
                f"Unsupported vector DB client type: {client_type}. "
                f"Supported types: chromadb, pinecone, qdrant, weaviate"
            )

    def _detect_retrieval_client(self, client: Any) -> str:
        """Detect vector database client type.

        Args:
            client: Vector database client object

        Returns:
            Client type string ("chromadb", "pinecone", "qdrant", "weaviate")
        """
        client_class = client.__class__.__name__

        if "chroma" in client_class.lower():
            return "chromadb"
        elif "pinecone" in client_class.lower():
            return "pinecone"
        elif "qdrant" in client_class.lower():
            return "qdrant"
        elif "weaviate" in client_class.lower():
            return "weaviate"
        else:
            # Try module name as fallback
            module_name = client.__class__.__module__
            if "chroma" in module_name.lower():
                return "chromadb"
            elif "pinecone" in module_name.lower():
                return "pinecone"
            elif "qdrant" in module_name.lower():
                return "qdrant"
            elif "weaviate" in module_name.lower():
                return "weaviate"

        return "unknown"

    def _query_chromadb(
        self,
        client: Any,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Query ChromaDB client.

        Args:
            client: ChromaDB client or collection
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with text and scores
        """
        # ChromaDB query API
        results = client.query(query_texts=[query], n_results=top_k)

        documents = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]  # First query
            distances = results.get("distances", [[]])[0]

            for i, doc in enumerate(docs):
                documents.append({
                    "text": doc,
                    "score": 1.0 - distances[i] if i < len(distances) else 0.0,
                })

        return documents

    def _query_pinecone(
        self,
        client: Any,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Query Pinecone index.

        Args:
            client: Pinecone index
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with text and scores
        """
        # Pinecone requires embedding the query first
        # For now, return empty list (user should provide embedding model)
        logger.warning("Pinecone retrieval requires embedding model - returning empty results")
        return []

    def _query_qdrant(
        self,
        client: Any,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Query Qdrant client.

        Args:
            client: Qdrant client
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with text and scores
        """
        # Qdrant requires embedding the query first
        # For now, return empty list (user should provide embedding model)
        logger.warning("Qdrant retrieval requires embedding model - returning empty results")
        return []

    def _query_weaviate(
        self,
        client: Any,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Query Weaviate client.

        Args:
            client: Weaviate client
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with text and scores
        """
        # Weaviate requires class name and schema
        # For now, return empty list (user should provide class name)
        logger.warning("Weaviate retrieval requires class name - returning empty results")
        return []

    def _call_llm_api(
        self,
        request: dict[str, Any],
        stream: bool = False,
        stream_callback: Any = None,
    ) -> tuple[str, int, float]:
        """Make real LLM API call with modified parameters.

        Args:
            request: LLM request with model, messages, etc.
            stream: If True, use streaming API
            stream_callback: Optional callback for streaming chunks (chunk_text: str) -> None

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)

        Raises:
            ValueError: If model is missing or vendor cannot be detected
            ImportError: If required SDK is not installed
            Exception: If API call fails
        """
        model = request.get("model")
        if not model:
            raise ValueError("Model is required for LLM API calls")

        # Detect vendor from model name
        vendor = self._detect_vendor(model)

        if vendor == "openai":
            return self._call_openai_api(request, stream=stream, stream_callback=stream_callback)
        elif vendor == "anthropic":
            return self._call_anthropic_api(request, stream=stream, stream_callback=stream_callback)
        else:
            raise ValueError(
                f"Unsupported model vendor for '{model}'. "
                f"Supported vendors: openai, anthropic"
            )

    def _detect_vendor(self, model: str) -> str:
        """Detect LLM vendor from model name.

        Args:
            model: Model name

        Returns:
            Vendor name (openai, anthropic)

        Raises:
            ValueError: If vendor cannot be detected
        """
        model_lower = model.lower()

        # OpenAI models
        if any(prefix in model_lower for prefix in ["gpt-", "o1-", "text-embedding"]):
            return "openai"

        # Anthropic models
        if any(prefix in model_lower for prefix in ["claude-", "claude"]):
            return "anthropic"

        raise ValueError(f"Cannot detect vendor from model name: {model}")

    def _call_openai_api(
        self,
        request: dict[str, Any],
        stream: bool = False,
        stream_callback: Any = None,
    ) -> tuple[str, int, float]:
        """Call OpenAI API with optional streaming support and retry logic.

        Automatically retries on transient errors (rate limits, timeouts, connection issues).

        Args:
            request: Request with model, messages, temperature, etc.
            stream: If True, use streaming API
            stream_callback: Optional callback for streaming chunks (chunk_text: str) -> None

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)

        Raises:
            ImportError: If openai package is not installed
            Exception: If API call fails after all retries
        """
        # Apply retry decorator dynamically
        @with_retry(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            max_delay=self.retry_max_delay,
            exponential_base=self.retry_exponential_base,
        )
        def _make_call() -> tuple[str, int, float]:
            return self._call_openai_api_impl(request, stream, stream_callback)

        return _make_call()

    def _call_openai_api_impl(
        self,
        request: dict[str, Any],
        stream: bool = False,
        stream_callback: Any = None,
    ) -> tuple[str, int, float]:
        """OpenAI API implementation (without retry logic).

        Args:
            request: Request with model, messages, temperature, etc.
            stream: If True, use streaming API
            stream_callback: Optional callback for streaming chunks

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI API calls. "
                "Install it with: pip install openai"
            )

        # Extract parameters
        model = request.get("model")
        messages = request.get("messages", [])
        temperature = request.get("temperature")
        max_tokens = request.get("max_tokens")

        # Build kwargs
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Make API call
        client = openai.OpenAI()

        if stream:
            # Streaming mode
            response_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            stream_response = client.chat.completions.create(**kwargs)
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    response_text += chunk_text

                    # Call callback if provided
                    if stream_callback:
                        stream_callback(chunk_text)

                # Extract usage from final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

            total_tokens = prompt_tokens + completion_tokens
        else:
            # Non-streaming mode
            response = client.chat.completions.create(**kwargs)

            # Extract response
            response_text = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        cost = self._estimate_cost(model, total_tokens)

        return response_text, total_tokens, cost

    def _call_anthropic_api(
        self,
        request: dict[str, Any],
        stream: bool = False,
        stream_callback: Any = None,
    ) -> tuple[str, int, float]:
        """Call Anthropic API with optional streaming support and retry logic.

        Automatically retries on transient errors (rate limits, timeouts, connection issues).

        Args:
            request: Request with model, messages, temperature, etc.
            stream: If True, use streaming API
            stream_callback: Optional callback for streaming chunks (chunk_text: str) -> None

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)

        Raises:
            ImportError: If anthropic package is not installed
            Exception: If API call fails after all retries
        """
        # Apply retry decorator dynamically
        @with_retry(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            max_delay=self.retry_max_delay,
            exponential_base=self.retry_exponential_base,
        )
        def _make_call() -> tuple[str, int, float]:
            return self._call_anthropic_api_impl(request, stream, stream_callback)

        return _make_call()

    def _call_anthropic_api_impl(
        self,
        request: dict[str, Any],
        stream: bool = False,
        stream_callback: Any = None,
    ) -> tuple[str, int, float]:
        """Anthropic API implementation (without retry logic).

        Args:
            request: Request with model, messages, temperature, etc.
            stream: If True, use streaming API
            stream_callback: Optional callback for streaming chunks

        Returns:
            Tuple of (response_text, tokens_used, cost_usd)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic API calls. "
                "Install it with: pip install anthropic"
            )

        # Extract parameters
        model = request.get("model")
        messages = request.get("messages", [])
        temperature = request.get("temperature")
        max_tokens = request.get("max_tokens", 1024)  # Anthropic requires max_tokens

        # Separate system message from messages
        system_message = None
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            else:
                user_messages.append(msg)

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
        }
        if system_message:
            kwargs["system"] = system_message
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Make API call
        client = anthropic.Anthropic()

        if stream:
            # Streaming mode
            response_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    response_text += text

                    # Call callback if provided
                    if stream_callback:
                        stream_callback(text)

                # Get final message with usage stats
                final_message = stream.get_final_message()
                if final_message.usage:
                    prompt_tokens = final_message.usage.input_tokens
                    completion_tokens = final_message.usage.output_tokens

            total_tokens = prompt_tokens + completion_tokens
        else:
            # Non-streaming mode
            response = client.messages.create(**kwargs)

            # Extract response text from content blocks
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            # Extract token usage
            prompt_tokens = response.usage.input_tokens if response.usage else 0
            completion_tokens = response.usage.output_tokens if response.usage else 0
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        cost = self._estimate_cost(model, total_tokens)

        return response_text, total_tokens, cost

    def _estimate_cost(self, model: str | None, tokens: int) -> float:
        """Estimate cost for API call.

        Args:
            model: Model name
            tokens: Total tokens used

        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates (per 1M tokens)
        cost_per_1m = {
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "gpt-3.5-turbo": 1.5,
            "claude-3-opus": 15.0,
            "claude-3-sonnet": 3.0,
            "claude-3-haiku": 0.8,
        }

        if not model:
            return 0.0

        # Find matching model
        for model_prefix, cost in cost_per_1m.items():
            if model.startswith(model_prefix):
                return (tokens / 1_000_000) * cost

        # Default rough estimate
        return (tokens / 1_000_000) * 5.0

    def _extract_output(self, span: Span) -> Any:
        """Extract final output from span.

        Args:
            span: Span to extract output from

        Returns:
            Output value
        """
        if span.replay_snapshot is None:
            return None

        snapshot = span.replay_snapshot

        if span.span_type == SpanType.LLM:
            return snapshot.llm_response.get("text") if snapshot.llm_response else None
        elif span.span_type == SpanType.TOOL:
            return snapshot.tool_output
        elif span.span_type == SpanType.RETRIEVAL:
            return snapshot.retrieved_documents
        elif span.span_type == SpanType.AGENT:
            return snapshot.agent_config

        return None
