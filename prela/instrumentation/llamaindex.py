"""Instrumentation for LlamaIndex (llama-index-core>=0.10.0).

This module provides automatic tracing for LlamaIndex operations via callbacks:
- LLM calls (OpenAI, Anthropic, etc. through LlamaIndex)
- Embeddings generation
- Query engine operations
- Retrieval operations with node scores
- Synthesis operations

The instrumentation works by injecting a PrelaHandler into LlamaIndex's
callback manager, which automatically captures all executions.

Example:
    ```python
    from prela.instrumentation.llamaindex import LlamaIndexInstrumentor
    from prela.core.tracer import Tracer
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    # Setup instrumentation
    tracer = Tracer()
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer)

    # Now all LlamaIndex operations are automatically traced
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the main topic?")
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from prela.core.clock import now
from prela.core.span import Span, SpanStatus, SpanType
from prela.instrumentation.base import Instrumentor

if TYPE_CHECKING:
    from prela.core.tracer import Tracer

logger = logging.getLogger(__name__)

# Maximum content length for truncation
_MAX_CONTENT_LEN = 500
_MAX_NODE_TEXT_LEN = 200
_MAX_ITEMS = 5


class PrelaHandler:
    """LlamaIndex callback handler that creates Prela spans.

    This handler implements LlamaIndex's BaseCallbackHandler interface and
    creates spans for all major LlamaIndex operations. It maintains a mapping
    from event_id to span to properly handle concurrent executions and nested
    operations.

    The handler tracks:
    - LLM calls: Model invocations with prompts and responses
    - Embeddings: Vector generation operations
    - Retrieval: Document retrieval with similarity scores
    - Query: Query engine operations
    - Synthesis: Response synthesis from retrieved documents
    """

    def __init__(self, tracer: Tracer) -> None:
        """Initialize the callback handler.

        Args:
            tracer: The tracer to use for creating spans
        """
        self._tracer = tracer
        # Map event_id -> span for tracking concurrent operations
        self._spans: dict[str, Span] = {}
        # Map event_id -> context manager for proper cleanup
        self._contexts: dict[str, Any] = {}
        # Map event_id -> ReplayCapture for replay data
        self._replay_captures: dict[str, Any] = {}

        # Required attributes for LlamaIndex callback interface
        self.event_starts_to_ignore: list[str] = []
        self.event_ends_to_ignore: list[str] = []

    def on_event_start(
        self,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when an event starts.

        Args:
            event_type: Type of event (LLM, EMBEDDING, RETRIEVE, etc.)
            payload: Event-specific data
            event_id: Unique identifier for this event
            parent_id: ID of parent event (if nested)
            **kwargs: Additional arguments

        Returns:
            The event_id for tracking
        """
        try:
            # Import here to avoid requiring llama-index at module level
            try:
                from llama_index.core.callbacks.schema import CBEventType
            except ImportError:
                logger.debug("llama_index.core not available, skipping event")
                return event_id

            # Map event type to span type
            span_type = self._map_event_to_span_type(event_type)
            span_name = self._get_span_name(event_type, payload)

            # Start the span
            ctx = self._tracer.span(
                name=span_name,
                span_type=span_type,
            )
            span = ctx.__enter__()

            # Store span and context for later retrieval
            self._spans[event_id] = span
            self._contexts[event_id] = ctx

            # Capture event-specific attributes
            self._capture_start_attributes(span, event_type, payload)

        except Exception as e:
            # Never break user code due to instrumentation errors
            logger.debug(f"Error in on_event_start: {e}")

        return event_id

    def on_event_end(
        self,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event ends.

        Args:
            event_type: Type of event (LLM, EMBEDDING, RETRIEVE, etc.)
            payload: Event-specific response data
            event_id: Unique identifier for this event
            **kwargs: Additional arguments
        """
        try:
            # Retrieve the span for this event
            span = self._spans.get(event_id)
            ctx = self._contexts.get(event_id)

            if not span or not ctx:
                logger.debug(f"No span found for event_id: {event_id}")
                return

            # Capture end attributes
            self._capture_end_attributes(span, event_type, payload)

            # End the span
            ctx.__exit__(None, None, None)

            # Clean up tracking dictionaries
            self._spans.pop(event_id, None)
            self._contexts.pop(event_id, None)

            # Clean up any remaining replay captures for this span
            span_id = str(id(span))
            if span_id in self._replay_captures:
                del self._replay_captures[span_id]

        except Exception as e:
            # Never break user code due to instrumentation errors
            logger.debug(f"Error in on_event_end: {e}")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Called when a trace starts.

        Args:
            trace_id: Optional trace identifier
        """
        # LlamaIndex specific - not used for our purposes
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Called when a trace ends.

        Args:
            trace_id: Optional trace identifier
            trace_map: Optional trace mapping
        """
        # LlamaIndex specific - not used for our purposes
        pass

    def _map_event_to_span_type(self, event_type: str) -> SpanType:
        """Map LlamaIndex event type to Prela span type.

        Args:
            event_type: LlamaIndex CBEventType string

        Returns:
            Corresponding SpanType
        """
        # Import here to avoid circular dependency
        try:
            from llama_index.core.callbacks.schema import CBEventType

            event_type_map = {
                CBEventType.LLM: SpanType.LLM,
                CBEventType.EMBEDDING: SpanType.EMBEDDING,
                CBEventType.RETRIEVE: SpanType.RETRIEVAL,
                CBEventType.QUERY: SpanType.AGENT,
                CBEventType.SYNTHESIZE: SpanType.AGENT,
                CBEventType.TREE: SpanType.AGENT,
                CBEventType.SUB_QUESTION: SpanType.AGENT,
                CBEventType.CHUNKING: SpanType.CUSTOM,
                CBEventType.NODE_PARSING: SpanType.CUSTOM,
                CBEventType.TEMPLATING: SpanType.CUSTOM,
            }

            return event_type_map.get(event_type, SpanType.CUSTOM)

        except (ImportError, AttributeError):
            # If CBEventType not available, default to CUSTOM
            return SpanType.CUSTOM

    def _get_span_name(
        self, event_type: str, payload: Optional[dict[str, Any]]
    ) -> str:
        """Generate a descriptive span name.

        Args:
            event_type: LlamaIndex event type
            payload: Event payload

        Returns:
            Human-readable span name
        """
        base_name = f"llamaindex.{event_type.lower()}"

        # Add more specific info if available
        if payload:
            if event_type == "LLM" and "serialized" in payload:
                model = payload.get("serialized", {}).get("model", "")
                if model:
                    return f"{base_name}.{model}"

            if event_type == "RETRIEVE" and "retriever_type" in payload:
                retriever = payload.get("retriever_type", "")
                if retriever:
                    return f"{base_name}.{retriever}"

        return base_name

    def _capture_start_attributes(
        self, span: Span, event_type: str, payload: Optional[dict[str, Any]]
    ) -> None:
        """Capture event-specific attributes at start.

        Args:
            span: The span to add attributes to
            event_type: LlamaIndex event type
            payload: Event payload
        """
        if not payload:
            return

        try:
            # Common attributes
            span.set_attribute("llamaindex.event_type", event_type)

            # LLM-specific attributes
            if event_type == "LLM":
                self._capture_llm_start(span, payload)

            # Embedding-specific attributes
            elif event_type == "EMBEDDING":
                self._capture_embedding_start(span, payload)

            # Retrieval-specific attributes
            elif event_type == "RETRIEVE":
                self._capture_retrieve_start(span, payload)

            # Query-specific attributes
            elif event_type == "QUERY":
                self._capture_query_start(span, payload)

            # Synthesis-specific attributes
            elif event_type == "SYNTHESIZE":
                self._capture_synthesize_start(span, payload)

        except Exception as e:
            logger.debug(f"Error capturing start attributes: {e}")

    def _capture_end_attributes(
        self, span: Span, event_type: str, payload: Optional[dict[str, Any]]
    ) -> None:
        """Capture event-specific attributes at end.

        Args:
            span: The span to add attributes to
            event_type: LlamaIndex event type
            payload: Event payload
        """
        if not payload:
            return

        try:
            # LLM-specific response
            if event_type == "LLM":
                self._capture_llm_end(span, payload)

            # Embedding-specific response
            elif event_type == "EMBEDDING":
                self._capture_embedding_end(span, payload)

            # Retrieval-specific response
            elif event_type == "RETRIEVE":
                self._capture_retrieve_end(span, payload)

            # Query-specific response
            elif event_type == "QUERY":
                self._capture_query_end(span, payload)

            # Synthesis-specific response
            elif event_type == "SYNTHESIZE":
                self._capture_synthesize_end(span, payload)

        except Exception as e:
            logger.debug(f"Error capturing end attributes: {e}")

    def _capture_llm_start(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture LLM start attributes."""
        # Model info
        model = None
        temperature = None
        max_tokens = None

        if "serialized" in payload:
            serialized = payload["serialized"]
            if "model" in serialized:
                model = serialized["model"]
                span.set_attribute("llm.model", model)
            if "temperature" in serialized:
                temperature = serialized["temperature"]
                span.set_attribute("llm.temperature", temperature)
            if "max_tokens" in serialized:
                max_tokens = serialized["max_tokens"]
                span.set_attribute("llm.max_tokens", max_tokens)

        # Prompts
        prompt = None
        if "messages" in payload:
            messages = payload["messages"]
            if messages and len(messages) > 0:
                # Truncate for display
                msg_str = str(messages[0])[:_MAX_CONTENT_LEN]
                prompt = msg_str
                span.set_attribute("llm.prompt", msg_str)
                span.set_attribute("llm.prompt_count", len(messages))

        if "formatted_prompt" in payload:
            prompt = payload["formatted_prompt"][:_MAX_CONTENT_LEN]
            span.set_attribute("llm.formatted_prompt", prompt)

        # NEW: Initialize replay capture if enabled
        if self._tracer.capture_for_replay:
            try:
                from prela.core.replay import ReplayCapture

                replay_capture = ReplayCapture()
                replay_capture.set_llm_request(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Store using span's id as key
                event_id = id(span)
                self._replay_captures[str(event_id)] = replay_capture
            except Exception as e:
                logger.debug(f"Failed to initialize LLM replay capture: {e}")

    def _capture_llm_end(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture LLM end attributes."""
        # Response text
        response_text = None
        prompt_tokens = None
        completion_tokens = None

        if "response" in payload:
            response = payload["response"]
            if hasattr(response, "text"):
                text = response.text[:_MAX_CONTENT_LEN]
                response_text = text
                span.set_attribute("llm.response", text)

            # Token usage
            if hasattr(response, "raw") and response.raw:
                raw = response.raw
                if hasattr(raw, "usage"):
                    usage = raw.usage
                    if hasattr(usage, "prompt_tokens"):
                        prompt_tokens = usage.prompt_tokens
                        span.set_attribute("llm.input_tokens", usage.prompt_tokens)
                    if hasattr(usage, "completion_tokens"):
                        completion_tokens = usage.completion_tokens
                        span.set_attribute("llm.output_tokens", usage.completion_tokens)
                    if hasattr(usage, "total_tokens"):
                        span.set_attribute("llm.total_tokens", usage.total_tokens)

        # NEW: Complete replay capture if enabled
        event_id = str(id(span))
        if self._tracer.capture_for_replay and event_id in self._replay_captures:
            try:
                replay_capture = self._replay_captures[event_id]
                replay_capture.set_llm_response(
                    text=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Attach to span
                object.__setattr__(span, "replay_snapshot", replay_capture.build())

                # Clean up
                del self._replay_captures[event_id]
            except Exception as e:
                logger.debug(f"Failed to capture LLM replay data: {e}")

    def _capture_embedding_start(
        self, span: Span, payload: dict[str, Any]
    ) -> None:
        """Capture embedding start attributes."""
        # Model info
        if "serialized" in payload:
            serialized = payload["serialized"]
            if "model_name" in serialized:
                span.set_attribute("embedding.model", serialized["model_name"])

        # Input chunks
        if "chunks" in payload:
            chunks = payload["chunks"]
            span.set_attribute("embedding.input_count", len(chunks))
            # Show first chunk as sample
            if chunks:
                sample = str(chunks[0])[:_MAX_CONTENT_LEN]
                span.set_attribute("embedding.input_sample", sample)

    def _capture_embedding_end(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture embedding end attributes."""
        # Embeddings
        if "chunks" in payload:
            chunks = payload["chunks"]
            span.set_attribute("embedding.output_count", len(chunks))

            # Capture dimensions from first embedding
            if chunks and len(chunks[0]) > 0:
                span.set_attribute("embedding.dimensions", len(chunks[0]))

    def _capture_retrieve_start(
        self, span: Span, payload: dict[str, Any]
    ) -> None:
        """Capture retrieval start attributes."""
        # Query string
        query = None
        if "query_str" in payload:
            query = payload["query_str"][:_MAX_CONTENT_LEN]
            span.set_attribute("retrieval.query", query)

        # Retriever configuration
        retriever_type = None
        similarity_top_k = None
        if "retriever_type" in payload:
            retriever_type = payload["retriever_type"]
            span.set_attribute("retrieval.type", retriever_type)

        if "similarity_top_k" in payload:
            similarity_top_k = payload["similarity_top_k"]
            span.set_attribute("retrieval.top_k", similarity_top_k)

        # NEW: Initialize replay capture if enabled
        if self._tracer.capture_for_replay:
            try:
                from prela.core.replay import ReplayCapture

                replay_capture = ReplayCapture()

                # Store for completion in _capture_retrieve_end
                event_id = id(span)
                self._replay_captures[str(event_id)] = {
                    "capture": replay_capture,
                    "query": query,
                    "retriever_type": retriever_type,
                    "metadata": {"similarity_top_k": similarity_top_k} if similarity_top_k else {},
                }
            except Exception as e:
                logger.debug(f"Failed to initialize retrieval replay capture: {e}")

    def _capture_retrieve_end(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture retrieval end attributes."""
        # Retrieved nodes
        if "nodes" in payload:
            nodes = payload["nodes"]
            span.set_attribute("retrieval.node_count", len(nodes))

            # Capture node details (limited)
            for i, node in enumerate(nodes[:_MAX_ITEMS]):
                prefix = f"retrieval.node.{i}"

                # Node score
                if hasattr(node, "score") and node.score is not None:
                    span.set_attribute(f"{prefix}.score", node.score)

                # Node text (truncated)
                if hasattr(node, "node") and hasattr(node.node, "text"):
                    text = node.node.text[:_MAX_NODE_TEXT_LEN]
                    span.set_attribute(f"{prefix}.text", text)

                # Node metadata
                if hasattr(node, "node") and hasattr(node.node, "metadata"):
                    metadata = node.node.metadata
                    if metadata:
                        # Capture a few key metadata fields
                        for key in ["file_name", "file_path", "page_label"]:
                            if key in metadata:
                                span.set_attribute(f"{prefix}.{key}", metadata[key])

        # NEW: Complete retrieval replay capture if enabled
        event_id = str(id(span))
        if self._tracer.capture_for_replay and event_id in self._replay_captures:
            try:
                replay_data = self._replay_captures[event_id]
                replay_capture = replay_data["capture"]

                # Extract document data from nodes
                docs = []
                scores = []
                if "nodes" in payload:
                    for node in payload["nodes"][:_MAX_ITEMS]:
                        doc_dict = {}
                        if hasattr(node, "node") and hasattr(node.node, "text"):
                            doc_dict["content"] = node.node.text[:_MAX_NODE_TEXT_LEN]
                        if hasattr(node, "node") and hasattr(node.node, "metadata"):
                            doc_dict["metadata"] = node.node.metadata
                        docs.append(doc_dict)

                        # Extract score if available
                        if hasattr(node, "score") and node.score is not None:
                            scores.append(node.score)

                # Capture retrieval
                replay_capture.set_retrieval(
                    query=replay_data["query"],
                    documents=docs,
                    scores=scores if scores else None,
                    metadata=replay_data["metadata"],
                )

                # Attach to span
                object.__setattr__(span, "replay_snapshot", replay_capture.build())

                # Clean up
                del self._replay_captures[event_id]
            except Exception as e:
                logger.debug(f"Failed to capture retrieval replay data: {e}")

    def _capture_query_start(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture query start attributes."""
        # Query string
        if "query_str" in payload:
            query = payload["query_str"][:_MAX_CONTENT_LEN]
            span.set_attribute("query.input", query)

        # Query mode
        if "query_mode" in payload:
            span.set_attribute("query.mode", payload["query_mode"])

    def _capture_query_end(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture query end attributes."""
        # Response
        if "response" in payload:
            response = payload["response"]
            if hasattr(response, "response"):
                text = response.response[:_MAX_CONTENT_LEN]
                span.set_attribute("query.output", text)

            # Source nodes used
            if hasattr(response, "source_nodes"):
                source_count = len(response.source_nodes)
                span.set_attribute("query.source_count", source_count)

    def _capture_synthesize_start(
        self, span: Span, payload: dict[str, Any]
    ) -> None:
        """Capture synthesis start attributes."""
        # Query
        if "query_str" in payload:
            query = payload["query_str"][:_MAX_CONTENT_LEN]
            span.set_attribute("synthesis.query", query)

        # Node count
        if "nodes" in payload:
            span.set_attribute("synthesis.node_count", len(payload["nodes"]))

    def _capture_synthesize_end(self, span: Span, payload: dict[str, Any]) -> None:
        """Capture synthesis end attributes."""
        # Response
        if "response" in payload:
            response = payload["response"]
            if hasattr(response, "response"):
                text = response.response[:_MAX_CONTENT_LEN]
                span.set_attribute("synthesis.output", text)


class LlamaIndexInstrumentor(Instrumentor):
    """Instrumentor for LlamaIndex framework.

    This instrumentor adds automatic tracing to LlamaIndex operations by
    injecting a callback handler into the global callback manager.

    Example:
        ```python
        from prela.instrumentation.llamaindex import LlamaIndexInstrumentor
        from prela.core.tracer import Tracer

        tracer = Tracer()
        instrumentor = LlamaIndexInstrumentor()
        instrumentor.instrument(tracer)

        # All LlamaIndex operations now traced
        ```
    """

    def __init__(self) -> None:
        """Initialize the instrumentor."""
        self._handler: Optional[PrelaHandler] = None
        self._instrumented = False

    def instrument(self, tracer: Tracer) -> None:
        """Enable instrumentation for LlamaIndex.

        Args:
            tracer: The tracer to use for creating spans

        Raises:
            RuntimeError: If llama-index-core is not installed
        """
        if self._instrumented:
            logger.debug("LlamaIndex already instrumented, skipping")
            return

        try:
            from llama_index.core import Settings
            from llama_index.core.callbacks import CallbackManager
        except ImportError as e:
            raise RuntimeError(
                "llama-index-core is not installed. "
                "Install it with: pip install llama-index-core"
            ) from e

        # Create handler
        self._handler = PrelaHandler(tracer)

        # Inject into global callback manager
        if Settings.callback_manager is None:
            Settings.callback_manager = CallbackManager([self._handler])
        else:
            Settings.callback_manager.add_handler(self._handler)

        self._instrumented = True
        logger.debug("LlamaIndex instrumentation enabled")

    def uninstrument(self) -> None:
        """Disable instrumentation and remove callback handler."""
        if not self._instrumented:
            return

        try:
            from llama_index.core import Settings

            if Settings.callback_manager and self._handler:
                Settings.callback_manager.remove_handler(self._handler)

            self._handler = None
            self._instrumented = False
            logger.debug("LlamaIndex instrumentation disabled")

        except ImportError:
            # LlamaIndex not available, nothing to uninstrument
            pass

    @property
    def is_instrumented(self) -> bool:
        """Check if LlamaIndex is currently instrumented.

        Returns:
            True if instrumented, False otherwise
        """
        return self._instrumented

    def get_handler(self) -> Optional[PrelaHandler]:
        """Get the callback handler instance.

        Returns:
            The PrelaHandler instance if instrumented, None otherwise
        """
        return self._handler
