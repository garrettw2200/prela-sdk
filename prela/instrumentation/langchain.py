"""Instrumentation for LangChain (langchain>=0.1.0).

This module provides automatic tracing for LangChain operations via callbacks:
- LLM calls (OpenAI, Anthropic, etc. through LangChain)
- Chain executions (sequential, map-reduce, etc.)
- Tool invocations
- Retriever queries
- Agent actions and decisions

The instrumentation works by injecting a PrelaCallbackHandler into LangChain's
global callback system, which automatically captures all executions.

Example:
    ```python
    from prela.instrumentation.langchain import LangChainInstrumentor
    from prela.core.tracer import Tracer
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Setup instrumentation
    tracer = Tracer()
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer)

    # Now all LangChain operations are automatically traced
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run("colorful socks")
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from prela.core.clock import now
from prela.core.span import Span, SpanStatus, SpanType
from prela.instrumentation.base import Instrumentor

if TYPE_CHECKING:
    from prela.core.tracer import Tracer

logger = logging.getLogger(__name__)


class PrelaCallbackHandler:
    """LangChain callback handler that creates Prela spans.

    This handler implements LangChain's BaseCallbackHandler interface and
    creates spans for all major LangChain operations. It maintains a mapping
    from run_id to span to properly handle concurrent executions and nested
    operations.

    The handler tracks:
    - LLM calls: Model invocations with prompts and responses
    - Chains: Sequential operations and workflows
    - Tools: External tool invocations
    - Retrievers: Document retrieval operations
    - Agents: Agent reasoning and actions
    """

    def __init__(self, tracer: Tracer) -> None:
        """Initialize the callback handler.

        Args:
            tracer: The tracer to use for creating spans
        """
        self._tracer = tracer
        # Map run_id -> span for tracking concurrent operations
        self._spans: dict[str, Span] = {}
        # Map run_id -> context manager for proper cleanup
        self._contexts: dict[str, Any] = {}
        # Map run_id -> ReplayCapture for replay data
        self._replay_captures: dict[str, Any] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts running.

        Args:
            serialized: Serialized LLM configuration
            prompts: Input prompts to the LLM
            run_id: Unique identifier for this LLM run
            parent_run_id: ID of parent operation (if nested)
            tags: Optional tags for categorization
            metadata: Optional metadata
            **kwargs: Additional LLM parameters
        """
        try:
            # Extract LLM info from serialized config
            llm_type = serialized.get("name", "unknown")
            model = serialized.get("kwargs", {}).get("model_name", "unknown")

            # Start span
            ctx = self._tracer.span(
                name=f"langchain.llm.{llm_type}",
                span_type=SpanType.LLM,
                attributes={
                    "llm.vendor": "langchain",
                    "llm.type": llm_type,
                    "llm.model": model,
                    "llm.prompt_count": len(prompts),
                },
            )

            # Enter context and store span
            span = ctx.__enter__()
            self._spans[str(run_id)] = span
            self._contexts[str(run_id)] = ctx

            # Add prompts as attributes (truncate if too long)
            for i, prompt in enumerate(prompts[:5]):  # Limit to first 5 prompts
                truncated = prompt[:500] if len(prompt) > 500 else prompt
                span.set_attribute(f"llm.prompt.{i}", truncated)

            # Add tags and metadata
            if tags:
                span.set_attribute("langchain.tags", tags)
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"langchain.metadata.{key}", str(value))

            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["callbacks", "run_manager"]:
                    span.set_attribute(f"llm.{key}", str(value))

            # NEW: Initialize replay capture if enabled
            if self._tracer.capture_for_replay:
                try:
                    from prela.core.replay import ReplayCapture

                    replay_capture = ReplayCapture()

                    # Extract parameters
                    invocation_params = kwargs.get("invocation_params", {})
                    temperature = invocation_params.get("temperature")
                    max_tokens = invocation_params.get("max_tokens")

                    # Capture request
                    replay_capture.set_llm_request(
                        model=model,
                        prompt=prompts[0] if prompts else None,  # First prompt
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **{k: v for k, v in invocation_params.items()
                           if k not in ["temperature", "max_tokens"]}
                    )

                    self._replay_captures[str(run_id)] = replay_capture
                except Exception as e:
                    logger.debug(f"Failed to initialize replay capture: {e}")

        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}", exc_info=True)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes running.

        Args:
            response: LLM response object
            run_id: Unique identifier for this LLM run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Extract response information
                if hasattr(response, "generations"):
                    generations = response.generations
                    for i, gen_list in enumerate(generations[:5]):
                        if gen_list:
                            text = getattr(gen_list[0], "text", "")
                            truncated = text[:500] if len(text) > 500 else text
                            span.set_attribute(f"llm.response.{i}", truncated)

                # Extract token usage if available
                if hasattr(response, "llm_output") and response.llm_output:
                    token_usage = response.llm_output.get("token_usage", {})
                    if "prompt_tokens" in token_usage:
                        span.set_attribute("llm.usage.prompt_tokens", token_usage["prompt_tokens"])
                    if "completion_tokens" in token_usage:
                        span.set_attribute("llm.usage.completion_tokens", token_usage["completion_tokens"])
                    if "total_tokens" in token_usage:
                        span.set_attribute("llm.usage.total_tokens", token_usage["total_tokens"])

                # Mark as successful
                span.set_status(SpanStatus.SUCCESS)

                # NEW: Complete replay capture if enabled
                if self._tracer.capture_for_replay and run_id_str in self._replay_captures:
                    try:
                        replay_capture = self._replay_captures[run_id_str]

                        # Extract response text
                        text = ""
                        if hasattr(response, "generations") and response.generations:
                            gen_list = response.generations[0]
                            if gen_list:
                                text = getattr(gen_list[0], "text", "")

                        # Extract token usage
                        prompt_tokens = None
                        completion_tokens = None
                        if hasattr(response, "llm_output") and response.llm_output:
                            token_usage = response.llm_output.get("token_usage", {})
                            prompt_tokens = token_usage.get("prompt_tokens")
                            completion_tokens = token_usage.get("completion_tokens")

                        # Capture response
                        replay_capture.set_llm_response(
                            text=text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )

                        # Attach to span
                        object.__setattr__(span, "replay_snapshot", replay_capture.build())

                        # Clean up replay capture
                        del self._replay_captures[run_id_str]
                    except Exception as e:
                        logger.debug(f"Failed to capture replay data: {e}")

                # Exit context
                ctx.__exit__(None, None, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}", exc_info=True)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors.

        Args:
            error: The error that occurred
            run_id: Unique identifier for this LLM run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Mark as error
                span.set_status(SpanStatus.ERROR, str(error))
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))

                # Exit context
                ctx.__exit__(type(error), error, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

                # Clean up replay capture if present
                if run_id_str in self._replay_captures:
                    del self._replay_captures[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_llm_error: {e}", exc_info=True)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts running.

        Args:
            serialized: Serialized chain configuration
            inputs: Input values to the chain
            run_id: Unique identifier for this chain run
            parent_run_id: ID of parent operation
            tags: Optional tags
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        try:
            # Extract chain info
            chain_type = serialized.get("name", "unknown")

            # Start span
            ctx = self._tracer.span(
                name=f"langchain.chain.{chain_type}",
                span_type=SpanType.AGENT,  # Chains are agent-level operations
                attributes={
                    "langchain.type": "chain",
                    "langchain.chain_type": chain_type,
                },
            )

            # Enter context and store span
            span = ctx.__enter__()
            self._spans[str(run_id)] = span
            self._contexts[str(run_id)] = ctx

            # Add inputs as attributes (truncate if needed)
            for key, value in inputs.items():
                value_str = str(value)
                truncated = value_str[:500] if len(value_str) > 500 else value_str
                span.set_attribute(f"chain.input.{key}", truncated)

            # Add tags and metadata
            if tags:
                span.set_attribute("langchain.tags", tags)
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"langchain.metadata.{key}", str(value))

        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}", exc_info=True)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain finishes running.

        Args:
            outputs: Output values from the chain
            run_id: Unique identifier for this chain run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Add outputs as attributes
                for key, value in outputs.items():
                    value_str = str(value)
                    truncated = value_str[:500] if len(value_str) > 500 else value_str
                    span.set_attribute(f"chain.output.{key}", truncated)

                # Mark as successful
                span.set_status(SpanStatus.SUCCESS)

                # Exit context
                ctx.__exit__(None, None, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}", exc_info=True)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors.

        Args:
            error: The error that occurred
            run_id: Unique identifier for this chain run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Mark as error
                span.set_status(SpanStatus.ERROR, str(error))
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))

                # Exit context
                ctx.__exit__(type(error), error, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_chain_error: {e}", exc_info=True)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running.

        Args:
            serialized: Serialized tool configuration
            input_str: Input string to the tool
            run_id: Unique identifier for this tool run
            parent_run_id: ID of parent operation
            tags: Optional tags
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        try:
            # Extract tool info
            tool_name = serialized.get("name", "unknown")

            # Start span
            ctx = self._tracer.span(
                name=f"langchain.tool.{tool_name}",
                span_type=SpanType.TOOL,
                attributes={
                    "tool.name": tool_name,
                    "tool.description": serialized.get("description", ""),
                },
            )

            # Enter context and store span
            span = ctx.__enter__()
            self._spans[str(run_id)] = span
            self._contexts[str(run_id)] = ctx

            # Add input
            truncated = input_str[:500] if len(input_str) > 500 else input_str
            span.set_attribute("tool.input", truncated)

            # Add tags and metadata
            if tags:
                span.set_attribute("langchain.tags", tags)
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"langchain.metadata.{key}", str(value))

            # NEW: Initialize replay capture for tools if enabled
            if self._tracer.capture_for_replay:
                try:
                    from prela.core.replay import ReplayCapture

                    replay_capture = ReplayCapture()
                    tool_name = serialized.get("name", "unknown")
                    tool_description = serialized.get("description")

                    replay_capture.set_tool_call(
                        name=tool_name,
                        description=tool_description,
                        input_args=input_str,
                    )
                    self._replay_captures[str(run_id)] = replay_capture
                except Exception as e:
                    logger.debug(f"Failed to initialize tool replay capture: {e}")

        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}", exc_info=True)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes running.

        Args:
            output: Output from the tool
            run_id: Unique identifier for this tool run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Add output
                truncated = output[:500] if len(output) > 500 else output
                span.set_attribute("tool.output", truncated)

                # Mark as successful
                span.set_status(SpanStatus.SUCCESS)

                # NEW: Complete tool replay capture if enabled
                if self._tracer.capture_for_replay and run_id_str in self._replay_captures:
                    try:
                        replay_capture = self._replay_captures[run_id_str]
                        replay_capture._snapshot.tool_output = output

                        # Attach to span
                        object.__setattr__(span, "replay_snapshot", replay_capture.build())

                        # Clean up replay capture
                        del self._replay_captures[run_id_str]
                    except Exception as e:
                        logger.debug(f"Failed to capture tool replay data: {e}")

                # Exit context
                ctx.__exit__(None, None, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}", exc_info=True)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors.

        Args:
            error: The error that occurred
            run_id: Unique identifier for this tool run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Mark as error
                span.set_status(SpanStatus.ERROR, str(error))
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))

                # Exit context
                ctx.__exit__(type(error), error, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

                # Clean up replay capture if present
                if run_id_str in self._replay_captures:
                    del self._replay_captures[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_tool_error: {e}", exc_info=True)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a retriever starts.

        Args:
            serialized: Serialized retriever configuration
            query: Query string for retrieval
            run_id: Unique identifier for this retriever run
            parent_run_id: ID of parent operation
            tags: Optional tags
            metadata: Optional metadata
            **kwargs: Additional parameters
        """
        try:
            # Extract retriever info
            retriever_type = serialized.get("name", "unknown")

            # Start span
            ctx = self._tracer.span(
                name=f"langchain.retriever.{retriever_type}",
                span_type=SpanType.RETRIEVAL,
                attributes={
                    "retriever.type": retriever_type,
                },
            )

            # Enter context and store span
            span = ctx.__enter__()
            self._spans[str(run_id)] = span
            self._contexts[str(run_id)] = ctx

            # Add query
            truncated = query[:500] if len(query) > 500 else query
            span.set_attribute("retriever.query", truncated)

            # Add tags and metadata
            if tags:
                span.set_attribute("langchain.tags", tags)
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"langchain.metadata.{key}", str(value))

            # NEW: Initialize replay capture for retrieval if enabled
            if self._tracer.capture_for_replay:
                try:
                    from prela.core.replay import ReplayCapture

                    replay_capture = ReplayCapture()
                    retriever_type = serialized.get("name", "unknown")

                    # Store for completion in on_retriever_end
                    # We'll add query and metadata now, documents later
                    self._replay_captures[str(run_id)] = {
                        "capture": replay_capture,
                        "query": query,
                        "retriever_type": retriever_type,
                        "metadata": metadata or {},
                    }
                except Exception as e:
                    logger.debug(f"Failed to initialize retrieval replay capture: {e}")

        except Exception as e:
            logger.error(f"Error in on_retriever_start: {e}", exc_info=True)

    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a retriever finishes.

        Args:
            documents: Retrieved documents
            run_id: Unique identifier for this retriever run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)
            ctx = self._contexts.get(run_id_str)

            if span and ctx:
                # Add document count
                span.set_attribute("retriever.document_count", len(documents))

                # Add document metadata
                for i, doc in enumerate(documents[:5]):  # Limit to first 5 docs
                    if hasattr(doc, "page_content"):
                        content = doc.page_content[:200]  # Truncate content
                        span.set_attribute(f"retriever.doc.{i}.content", content)
                    if hasattr(doc, "metadata"):
                        for key, value in doc.metadata.items():
                            span.set_attribute(f"retriever.doc.{i}.metadata.{key}", str(value))

                # Mark as successful
                span.set_status(SpanStatus.SUCCESS)

                # NEW: Complete retrieval replay capture if enabled
                if self._tracer.capture_for_replay and run_id_str in self._replay_captures:
                    try:
                        replay_data = self._replay_captures[run_id_str]
                        replay_capture = replay_data["capture"]

                        # Extract document data
                        docs = []
                        scores = []
                        for doc in documents[:5]:  # Limit to first 5
                            doc_dict = {}
                            if hasattr(doc, "page_content"):
                                doc_dict["content"] = doc.page_content[:200]  # Truncate
                            if hasattr(doc, "metadata"):
                                doc_dict["metadata"] = doc.metadata
                            docs.append(doc_dict)

                            # Extract score if available
                            if hasattr(doc, "metadata") and "score" in doc.metadata:
                                scores.append(doc.metadata["score"])

                        # Capture retrieval
                        replay_capture.set_retrieval(
                            query=replay_data["query"],
                            documents=docs,
                            scores=scores if scores else None,
                            metadata=replay_data["metadata"],
                        )

                        # Attach to span
                        object.__setattr__(span, "replay_snapshot", replay_capture.build())

                        # Clean up replay capture
                        del self._replay_captures[run_id_str]
                    except Exception as e:
                        logger.debug(f"Failed to capture retrieval replay data: {e}")

                # Exit context
                ctx.__exit__(None, None, None)

                # Clean up
                del self._spans[run_id_str]
                del self._contexts[run_id_str]

        except Exception as e:
            logger.error(f"Error in on_retriever_end: {e}", exc_info=True)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action.

        Args:
            action: The agent action
            run_id: Unique identifier for this run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)

            if span:
                # Record action event
                span.add_event(
                    "agent.action",
                    attributes={
                        "action.tool": str(getattr(action, "tool", "unknown")),
                        "action.tool_input": str(getattr(action, "tool_input", ""))[:500],
                        "action.log": str(getattr(action, "log", ""))[:500],
                    },
                )

        except Exception as e:
            logger.error(f"Error in on_agent_action: {e}", exc_info=True)

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes.

        Args:
            finish: The agent finish data
            run_id: Unique identifier for this run
            parent_run_id: ID of parent operation
            **kwargs: Additional parameters
        """
        try:
            run_id_str = str(run_id)
            span = self._spans.get(run_id_str)

            if span:
                # Record finish event
                return_values = getattr(finish, "return_values", {})
                span.add_event(
                    "agent.finish",
                    attributes={
                        "finish.output": str(return_values)[:500],
                        "finish.log": str(getattr(finish, "log", ""))[:500],
                    },
                )

        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}", exc_info=True)


class LangChainInstrumentor(Instrumentor):
    """Instrumentor for LangChain framework.

    This instrumentor injects a PrelaCallbackHandler into LangChain's global
    callback system, which automatically traces all LangChain operations
    including LLM calls, chains, tools, retrievers, and agent actions.

    Unlike other instrumentors that use function wrapping, this uses LangChain's
    built-in callback mechanism for more robust and comprehensive tracing.
    """

    def __init__(self) -> None:
        """Initialize the LangChain instrumentor."""
        self._callback_handler: PrelaCallbackHandler | None = None
        self._langchain_core_module: Any = None

    def instrument(self, tracer: Tracer) -> None:
        """Enable instrumentation for LangChain.

        This adds a PrelaCallbackHandler to LangChain's global callback
        manager, which will receive events for all LangChain operations.

        Args:
            tracer: The tracer to use for creating spans

        Raises:
            ImportError: If langchain-core package is not installed
            RuntimeError: If instrumentation fails
        """
        if self.is_instrumented:
            logger.debug("LangChain is already instrumented, skipping")
            return

        try:
            from langchain_core import callbacks as lc_callbacks
        except ImportError as e:
            raise ImportError(
                "langchain-core package is not installed. "
                "Install it with: pip install langchain-core>=0.1.0"
            ) from e

        self._langchain_core_module = lc_callbacks

        # Create callback handler
        self._callback_handler = PrelaCallbackHandler(tracer)

        # Add to global callbacks
        try:
            if hasattr(lc_callbacks, "get_callback_manager"):
                # Older API
                callback_manager = lc_callbacks.get_callback_manager()
                if hasattr(callback_manager, "add_handler"):
                    callback_manager.add_handler(self._callback_handler)
            else:
                # Newer API - add to default handlers
                if not hasattr(lc_callbacks, "_prela_handlers"):
                    lc_callbacks._prela_handlers = []
                lc_callbacks._prela_handlers.append(self._callback_handler)

            logger.info("Successfully instrumented LangChain")

        except Exception as e:
            self._callback_handler = None
            raise RuntimeError(f"Failed to instrument LangChain: {e}") from e

    def uninstrument(self) -> None:
        """Disable instrumentation and remove callback handler.

        Raises:
            RuntimeError: If uninstrumentation fails
        """
        if not self.is_instrumented:
            logger.debug("LangChain is not instrumented, skipping")
            return

        try:
            if self._langchain_core_module and self._callback_handler:
                lc_callbacks = self._langchain_core_module

                # Remove from global callbacks
                if hasattr(lc_callbacks, "get_callback_manager"):
                    # Older API
                    callback_manager = lc_callbacks.get_callback_manager()
                    if hasattr(callback_manager, "remove_handler"):
                        callback_manager.remove_handler(self._callback_handler)
                else:
                    # Newer API - remove from default handlers
                    if hasattr(lc_callbacks, "_prela_handlers"):
                        lc_callbacks._prela_handlers.remove(self._callback_handler)
                        if not lc_callbacks._prela_handlers:
                            delattr(lc_callbacks, "_prela_handlers")

            self._callback_handler = None
            self._langchain_core_module = None

            logger.info("Successfully uninstrumented LangChain")

        except Exception as e:
            raise RuntimeError(f"Failed to uninstrument LangChain: {e}") from e

    @property
    def is_instrumented(self) -> bool:
        """Check if LangChain is currently instrumented.

        Returns:
            True if instrumentation is active, False otherwise
        """
        return self._callback_handler is not None

    def get_callback(self) -> PrelaCallbackHandler | None:
        """Get the active callback handler.

        This can be used to manually add the handler to specific LangChain
        operations if needed, though auto-instrumentation is recommended.

        Returns:
            PrelaCallbackHandler | None: The callback handler if instrumented

        Example:
            ```python
            instrumentor = LangChainInstrumentor()
            instrumentor.instrument(tracer)

            # Optional: Get handler for manual use
            handler = instrumentor.get_callback()
            chain.run(input_text, callbacks=[handler])
            ```
        """
        return self._callback_handler
