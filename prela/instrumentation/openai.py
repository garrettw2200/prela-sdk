"""Instrumentation for OpenAI SDK (openai>=1.0.0).

This module provides automatic tracing for OpenAI's API, including:
- Synchronous and asynchronous chat completions
- Legacy completions API
- Embeddings API
- Streaming responses
- Function/tool calling

Example:
    ```python
    from prela.instrumentation.openai import OpenAIInstrumentor
    from prela.core.tracer import Tracer
    import openai

    tracer = Tracer()
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer)

    # Now all OpenAI API calls will be automatically traced
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from prela.core.clock import monotonic_ns, duration_ms
from prela.core.span import SpanType, SpanStatus
from prela.instrumentation.base import (
    Instrumentor,
    wrap_function,
    unwrap_function,
    _ORIGINALS_ATTR,
)

if TYPE_CHECKING:
    from prela.core.tracer import Tracer

logger = logging.getLogger(__name__)


class OpenAIInstrumentor(Instrumentor):
    """Instrumentor for OpenAI SDK.

    Patches the following methods:
    - openai.OpenAI.chat.completions.create (sync)
    - openai.AsyncOpenAI.chat.completions.create (async)
    - openai.OpenAI.completions.create (sync, legacy)
    - openai.OpenAI.embeddings.create (sync)

    Captures detailed information about requests, responses, tool usage,
    and streaming events.
    """

    def __init__(self) -> None:
        """Initialize the OpenAI instrumentor."""
        self._tracer: Tracer | None = None
        self._openai_module: Any = None
        self._chat_completions_module: Any = None
        self._async_chat_completions_module: Any = None
        self._completions_module: Any = None
        self._embeddings_module: Any = None

    def instrument(self, tracer: Tracer) -> None:
        """Enable instrumentation for OpenAI SDK.

        Args:
            tracer: The tracer to use for creating spans

        Raises:
            ImportError: If openai package is not installed
            RuntimeError: If instrumentation fails
        """
        if self.is_instrumented:
            logger.debug("OpenAI SDK is already instrumented, skipping")
            return

        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai>=1.0.0"
            ) from e

        self._tracer = tracer
        self._openai_module = openai

        try:
            # Get the completions modules for sync and async
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI.__new__(openai.OpenAI)
                if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                    self._chat_completions_module = client.chat.completions.__class__
                if hasattr(client, "completions"):
                    self._completions_module = client.completions.__class__
                if hasattr(client, "embeddings"):
                    self._embeddings_module = client.embeddings.__class__

            if hasattr(openai, "AsyncOpenAI"):
                async_client = openai.AsyncOpenAI.__new__(openai.AsyncOpenAI)
                if hasattr(async_client, "chat") and hasattr(
                    async_client.chat, "completions"
                ):
                    self._async_chat_completions_module = (
                        async_client.chat.completions.__class__
                    )

            # Wrap sync chat completions
            if self._chat_completions_module is not None:
                wrap_function(
                    self._chat_completions_module,
                    "create",
                    lambda orig: self._create_chat_completions_wrapper(
                        orig, is_async=False
                    ),
                )
                logger.debug("Wrapped openai.OpenAI.chat.completions.create")

            # Wrap async chat completions
            if self._async_chat_completions_module is not None:
                wrap_function(
                    self._async_chat_completions_module,
                    "create",
                    lambda orig: self._create_chat_completions_wrapper(
                        orig, is_async=True
                    ),
                )
                logger.debug("Wrapped openai.AsyncOpenAI.chat.completions.create")

            # Wrap legacy completions
            if self._completions_module is not None:
                wrap_function(
                    self._completions_module,
                    "create",
                    lambda orig: self._create_completions_wrapper(orig),
                )
                logger.debug("Wrapped openai.OpenAI.completions.create")

            # Wrap embeddings
            if self._embeddings_module is not None:
                wrap_function(
                    self._embeddings_module,
                    "create",
                    lambda orig: self._create_embeddings_wrapper(orig),
                )
                logger.debug("Wrapped openai.OpenAI.embeddings.create")

            logger.info("Successfully instrumented OpenAI SDK")

        except Exception as e:
            self._tracer = None
            self._openai_module = None
            self._chat_completions_module = None
            self._async_chat_completions_module = None
            self._completions_module = None
            self._embeddings_module = None
            raise RuntimeError(f"Failed to instrument OpenAI SDK: {e}") from e

    def uninstrument(self) -> None:
        """Disable instrumentation and restore original functions."""
        if not self.is_instrumented:
            logger.debug("OpenAI SDK is not instrumented, skipping")
            return

        try:
            # Unwrap chat completions
            if self._chat_completions_module is not None:
                unwrap_function(self._chat_completions_module, "create")

            if self._async_chat_completions_module is not None:
                unwrap_function(self._async_chat_completions_module, "create")

            # Unwrap legacy completions
            if self._completions_module is not None:
                unwrap_function(self._completions_module, "create")

            # Unwrap embeddings
            if self._embeddings_module is not None:
                unwrap_function(self._embeddings_module, "create")

            logger.info("Successfully uninstrumented OpenAI SDK")

        finally:
            self._tracer = None
            self._openai_module = None
            self._chat_completions_module = None
            self._async_chat_completions_module = None
            self._completions_module = None
            self._embeddings_module = None

    @property
    def is_instrumented(self) -> bool:
        """Check if OpenAI SDK is currently instrumented."""
        return (
            self._tracer is not None
            and self._chat_completions_module is not None
            and hasattr(self._chat_completions_module, _ORIGINALS_ATTR)
        )

    def _create_chat_completions_wrapper(
        self, original_func: Callable[..., Any], is_async: bool
    ) -> Callable[..., Any]:
        """Create a wrapper for chat.completions.create method.

        Args:
            original_func: The original create function
            is_async: Whether this is an async function

        Returns:
            Wrapped function that creates spans
        """
        if is_async:

            @wraps(original_func)
            async def async_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                return await self._trace_chat_completions(
                    original_func, self_obj, is_async=True, *args, **kwargs
                )

            return async_wrapper
        else:

            @wraps(original_func)
            def sync_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                return self._trace_chat_completions(
                    original_func, self_obj, is_async=False, *args, **kwargs
                )

            return sync_wrapper

    def _create_completions_wrapper(
        self, original_func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper for legacy completions.create method.

        Args:
            original_func: The original create function

        Returns:
            Wrapped function that creates spans
        """

        @wraps(original_func)
        def wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
            return self._trace_completions(original_func, self_obj, *args, **kwargs)

        return wrapper

    def _create_embeddings_wrapper(
        self, original_func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a wrapper for embeddings.create method.

        Args:
            original_func: The original create function

        Returns:
            Wrapped function that creates spans
        """

        @wraps(original_func)
        def wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
            return self._trace_embeddings(original_func, self_obj, *args, **kwargs)

        return wrapper

    def _trace_chat_completions(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        is_async: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a chat.completions.create call (sync or async).

        Args:
            original_func: The original create function
            self_obj: The completions object (self)
            is_async: Whether this is an async call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The response from the API call
        """
        if is_async:
            return self._trace_chat_completions_async(
                original_func, self_obj, *args, **kwargs
            )
        else:
            return self._trace_chat_completions_sync(
                original_func, self_obj, *args, **kwargs
            )

    def _trace_chat_completions_sync(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a synchronous chat.completions.create call.

        Args:
            original_func: The original create function
            self_obj: The completions object (self)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The response from the API call
        """
        if self._tracer is None:
            return original_func(self_obj, *args, **kwargs)

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        stream = kwargs.get("stream", False)

        # Start timing
        start_time = monotonic_ns()

        # Initialize replay capture if enabled
        replay_capture = None
        if self._tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_llm_request(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ("model", "messages", "temperature", "max_tokens", "stream")}
            )

        # Create span
        span = self._tracer.start_span(
            name="openai.chat.completions.create",
            span_type=SpanType.LLM,
        )

        try:
            # Set request attributes
            span.set_attribute("llm.vendor", "openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.model", model)

            if temperature is not None:
                span.set_attribute("llm.temperature", temperature)
            if max_tokens is not None:
                span.set_attribute("llm.max_tokens", max_tokens)
            if stream:
                span.set_attribute("llm.stream", True)

            # Add request event
            span.add_event(
                name="llm.request",
                attributes={"messages": messages},
            )

            # Make the API call
            response = original_func(self_obj, *args, **kwargs)

            # Handle streaming response
            if stream:
                return TracedChatCompletionStream(
                    stream=response,
                    span=span,
                    tracer=self._tracer,
                    start_time=start_time,
                    replay_capture=replay_capture,
                )

            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(start_time, end_time)
            span.set_attribute("llm.latency_ms", latency_ms)

            # Extract response attributes
            self._extract_chat_completion_attributes(span, response)

            # Add response event
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message"):
                    span.add_event(
                        name="llm.response",
                        attributes={"content": first_choice.message.content},
                    )

            # Handle tool calls
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and hasattr(
                    first_choice.message, "tool_calls"
                ):
                    if first_choice.message.tool_calls:
                        self._handle_tool_calls(span, first_choice.message.tool_calls)

            # Finalize replay capture
            if replay_capture:
                try:
                    # Extract response text
                    response_text = ""
                    if hasattr(response, "choices") and response.choices:
                        first_choice = response.choices[0]
                        if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                            response_text = first_choice.message.content or ""

                    # Extract usage
                    prompt_tokens = None
                    completion_tokens = None
                    if hasattr(response, "usage"):
                        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                        completion_tokens = getattr(response.usage, "completion_tokens", None)

                    # Extract finish reason
                    finish_reason = None
                    if hasattr(response, "choices") and response.choices:
                        first_choice = response.choices[0]
                        finish_reason = getattr(first_choice, "finish_reason", None)

                    replay_capture.set_llm_response(
                        text=response_text,
                        finish_reason=finish_reason,
                        model=getattr(response, "model", model),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                    replay_capture.set_model_info(
                        model=getattr(response, "model", model),
                        created=getattr(response, "created", None),
                        id=getattr(response, "id", None),
                    )

                    # Attach replay snapshot to span
                    object.__setattr__(span, "replay_snapshot", replay_capture.build())
                except Exception as e:
                    logger.debug(f"Failed to capture replay data: {e}")

            # Mark as successful
            span.set_status(SpanStatus.SUCCESS)

            return response

        except Exception as e:
            # Handle errors
            self._handle_error(span, e)
            raise

        finally:
            span.end()

    async def _trace_chat_completions_async(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace an asynchronous chat.completions.create call.

        Args:
            original_func: The original create function
            self_obj: The completions object (self)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The response from the API call
        """
        if self._tracer is None:
            return await original_func(self_obj, *args, **kwargs)

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        stream = kwargs.get("stream", False)

        # Start timing
        start_time = monotonic_ns()

        # Initialize replay capture if enabled
        replay_capture = None
        if self._tracer.capture_for_replay:
            from prela.core.replay import ReplayCapture

            replay_capture = ReplayCapture()
            replay_capture.set_llm_request(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ("model", "messages", "temperature", "max_tokens", "stream")}
            )

        # Create span
        span = self._tracer.start_span(
            name="openai.chat.completions.create",
            span_type=SpanType.LLM,
        )

        try:
            # Set request attributes
            span.set_attribute("llm.vendor", "openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.model", model)

            if temperature is not None:
                span.set_attribute("llm.temperature", temperature)
            if max_tokens is not None:
                span.set_attribute("llm.max_tokens", max_tokens)
            if stream:
                span.set_attribute("llm.stream", True)

            # Add request event
            span.add_event(
                name="llm.request",
                attributes={"messages": messages},
            )

            # Make the API call
            response = await original_func(self_obj, *args, **kwargs)

            # Handle streaming response
            if stream:
                return TracedAsyncChatCompletionStream(
                    stream=response,
                    span=span,
                    tracer=self._tracer,
                    start_time=start_time,
                    replay_capture=replay_capture,
                )

            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(start_time, end_time)
            span.set_attribute("llm.latency_ms", latency_ms)

            # Extract response attributes
            self._extract_chat_completion_attributes(span, response)

            # Add response event
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message"):
                    span.add_event(
                        name="llm.response",
                        attributes={"content": first_choice.message.content},
                    )

            # Handle tool calls
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and hasattr(
                    first_choice.message, "tool_calls"
                ):
                    if first_choice.message.tool_calls:
                        self._handle_tool_calls(span, first_choice.message.tool_calls)

            # Finalize replay capture
            if replay_capture:
                try:
                    # Extract response text
                    response_text = ""
                    if hasattr(response, "choices") and response.choices:
                        first_choice = response.choices[0]
                        if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                            response_text = first_choice.message.content or ""

                    # Extract usage
                    prompt_tokens = None
                    completion_tokens = None
                    if hasattr(response, "usage"):
                        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                        completion_tokens = getattr(response.usage, "completion_tokens", None)

                    # Extract finish reason
                    finish_reason = None
                    if hasattr(response, "choices") and response.choices:
                        first_choice = response.choices[0]
                        finish_reason = getattr(first_choice, "finish_reason", None)

                    replay_capture.set_llm_response(
                        text=response_text,
                        finish_reason=finish_reason,
                        model=getattr(response, "model", model),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                    replay_capture.set_model_info(
                        model=getattr(response, "model", model),
                        created=getattr(response, "created", None),
                        id=getattr(response, "id", None),
                    )

                    # Attach replay snapshot to span
                    object.__setattr__(span, "replay_snapshot", replay_capture.build())
                except Exception as e:
                    logger.debug(f"Failed to capture replay data: {e}")

            # Mark as successful
            span.set_status(SpanStatus.SUCCESS)

            return response

        except Exception as e:
            # Handle errors
            self._handle_error(span, e)
            raise

        finally:
            span.end()

    def _trace_completions(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a legacy completions.create call.

        Args:
            original_func: The original create function
            self_obj: The completions object (self)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The response from the API call
        """
        if self._tracer is None:
            return original_func(self_obj, *args, **kwargs)

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        prompt = kwargs.get("prompt", "")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        # Start timing
        start_time = monotonic_ns()

        # Create span
        span = self._tracer.start_span(
            name="openai.completions.create",
            span_type=SpanType.LLM,
        )

        try:
            # Set request attributes
            span.set_attribute("llm.vendor", "openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.model", model)

            if temperature is not None:
                span.set_attribute("llm.temperature", temperature)
            if max_tokens is not None:
                span.set_attribute("llm.max_tokens", max_tokens)

            # Add request event
            span.add_event(
                name="llm.request",
                attributes={"prompt": prompt},
            )

            # Make the API call
            response = original_func(self_obj, *args, **kwargs)

            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(start_time, end_time)
            span.set_attribute("llm.latency_ms", latency_ms)

            # Extract response attributes
            self._extract_completion_attributes(span, response)

            # Add response event
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "text"):
                    span.add_event(
                        name="llm.response",
                        attributes={"text": first_choice.text},
                    )

            # Mark as successful
            span.set_status(SpanStatus.SUCCESS)

            return response

        except Exception as e:
            # Handle errors
            self._handle_error(span, e)
            raise

        finally:
            span.end()

    def _trace_embeddings(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace an embeddings.create call.

        Args:
            original_func: The original create function
            self_obj: The embeddings object (self)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The response from the API call
        """
        if self._tracer is None:
            return original_func(self_obj, *args, **kwargs)

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        input_data = kwargs.get("input", [])

        # Start timing
        start_time = monotonic_ns()

        # Create span
        span = self._tracer.start_span(
            name="openai.embeddings.create",
            span_type=SpanType.EMBEDDING,
        )

        try:
            # Set request attributes
            span.set_attribute("llm.vendor", "openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.model", model)

            # Count inputs
            if isinstance(input_data, list):
                span.set_attribute("embedding.input_count", len(input_data))
            else:
                span.set_attribute("embedding.input_count", 1)

            # Make the API call
            response = original_func(self_obj, *args, **kwargs)

            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(start_time, end_time)
            span.set_attribute("llm.latency_ms", latency_ms)

            # Extract response attributes
            self._extract_embedding_attributes(span, response)

            # Mark as successful
            span.set_status(SpanStatus.SUCCESS)

            return response

        except Exception as e:
            # Handle errors
            self._handle_error(span, e)
            raise

        finally:
            span.end()

    def _extract_chat_completion_attributes(self, span: Any, response: Any) -> None:
        """Extract attributes from a chat completion response.

        Args:
            span: The span to add attributes to
            response: The response object from the API
        """
        try:
            # Model (actual model used)
            if hasattr(response, "model"):
                span.set_attribute("llm.response.model", response.model)

            # Response ID
            if hasattr(response, "id"):
                span.set_attribute("llm.response.id", response.id)

            # Usage statistics
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                if hasattr(usage, "completion_tokens"):
                    span.set_attribute("llm.completion_tokens", usage.completion_tokens)
                if hasattr(usage, "total_tokens"):
                    span.set_attribute("llm.total_tokens", usage.total_tokens)

            # Finish reason
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    span.set_attribute("llm.finish_reason", first_choice.finish_reason)

        except Exception as e:
            # Don't let attribute extraction failures break the instrumentation
            logger.debug(f"Failed to extract chat completion attributes: {e}")

    def _extract_completion_attributes(self, span: Any, response: Any) -> None:
        """Extract attributes from a legacy completion response.

        Args:
            span: The span to add attributes to
            response: The response object from the API
        """
        try:
            # Model
            if hasattr(response, "model"):
                span.set_attribute("llm.response.model", response.model)

            # Response ID
            if hasattr(response, "id"):
                span.set_attribute("llm.response.id", response.id)

            # Usage statistics
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                if hasattr(usage, "completion_tokens"):
                    span.set_attribute("llm.completion_tokens", usage.completion_tokens)
                if hasattr(usage, "total_tokens"):
                    span.set_attribute("llm.total_tokens", usage.total_tokens)

        except Exception as e:
            logger.debug(f"Failed to extract completion attributes: {e}")

    def _extract_embedding_attributes(self, span: Any, response: Any) -> None:
        """Extract attributes from an embedding response.

        Args:
            span: The span to add attributes to
            response: The response object from the API
        """
        try:
            # Model
            if hasattr(response, "model"):
                span.set_attribute("llm.response.model", response.model)

            # Usage statistics
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                if hasattr(usage, "total_tokens"):
                    span.set_attribute("llm.total_tokens", usage.total_tokens)

            # Embedding count and dimensions
            if hasattr(response, "data") and response.data:
                span.set_attribute("embedding.count", len(response.data))
                if response.data and hasattr(response.data[0], "embedding"):
                    span.set_attribute(
                        "embedding.dimensions", len(response.data[0].embedding)
                    )

        except Exception as e:
            logger.debug(f"Failed to extract embedding attributes: {e}")

    def _handle_tool_calls(self, span: Any, tool_calls: Any) -> None:
        """Handle tool calls in the response.

        Args:
            span: The span to add tool call information to
            tool_calls: The tool calls from the response
        """
        try:
            calls = []
            for tool_call in tool_calls:
                call_info = {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", None),
                }

                if hasattr(tool_call, "function"):
                    function = tool_call.function
                    call_info["function"] = {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    }

                calls.append(call_info)

            if calls:
                span.add_event(
                    name="llm.tool_calls",
                    attributes={"tool_calls": calls},
                )

        except Exception as e:
            logger.debug(f"Failed to handle tool calls: {e}")

    def _handle_error(self, span: Any, error: Exception) -> None:
        """Handle an error during API call.

        Args:
            span: The span to record the error on
            error: The exception that was raised
        """
        try:
            # Set error status
            span.set_status(SpanStatus.ERROR, str(error))

            # Extract error details
            error_attrs: dict[str, Any] = {
                "error.type": type(error).__name__,
                "error.message": str(error),
            }

            # Handle openai-specific errors
            if hasattr(error, "status_code"):
                error_attrs["error.status_code"] = error.status_code

            span.add_event(name="error", attributes=error_attrs)

        except Exception as e:
            logger.debug(f"Failed to handle error: {e}")


class TracedChatCompletionStream:
    """Wrapper for streaming chat completion responses."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        tracer: Tracer,
        start_time: int,
        replay_capture: Any = None,
    ) -> None:
        """Initialize the traced stream.

        Args:
            stream: The original stream
            span: The span to record events on
            tracer: The tracer instance
            start_time: Start time in nanoseconds
            replay_capture: Optional ReplayCapture instance
        """
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._start_time = start_time
        self._first_token_time: int | None = None
        self._content_chunks: list[str] = []
        self._finish_reason: str | None = None
        self._replay_capture = replay_capture

    def __enter__(self) -> TracedChatCompletionStream:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and finalize span."""
        try:
            if exc_type is None:
                self._finalize_span()
            else:
                self._span.set_status(SpanStatus.ERROR, str(exc_val))

        finally:
            self._span.end()

    def __iter__(self) -> Any:
        """Iterate over stream chunks."""
        try:
            for chunk in self._stream:
                self._process_chunk(chunk)
                yield chunk

        except Exception as e:
            self._span.set_status(SpanStatus.ERROR, str(e))
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process a streaming chunk.

        Args:
            chunk: The streaming chunk
        """
        try:
            # Capture first token time
            if self._first_token_time is None:
                self._first_token_time = monotonic_ns()

            # Extract content from chunk
            if hasattr(chunk, "choices") and chunk.choices:
                first_choice = chunk.choices[0]

                # Get content delta
                if hasattr(first_choice, "delta"):
                    delta = first_choice.delta
                    if hasattr(delta, "content") and delta.content:
                        self._content_chunks.append(delta.content)

                # Get finish reason
                if hasattr(first_choice, "finish_reason") and first_choice.finish_reason:
                    self._finish_reason = first_choice.finish_reason

        except Exception as e:
            logger.debug(f"Failed to process chunk: {e}")

    def _finalize_span(self) -> None:
        """Finalize the span with aggregated data."""
        try:
            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(self._start_time, end_time)
            self._span.set_attribute("llm.latency_ms", latency_ms)

            # Time to first token
            if self._first_token_time is not None:
                ttft_ms = duration_ms(self._start_time, self._first_token_time)
                self._span.set_attribute("llm.time_to_first_token_ms", ttft_ms)

            # Aggregated content
            if self._content_chunks:
                full_content = "".join(self._content_chunks)
                self._span.add_event(
                    name="llm.response",
                    attributes={"content": full_content},
                )

            # Finish reason
            if self._finish_reason:
                self._span.set_attribute("llm.finish_reason", self._finish_reason)

            # Finalize replay capture for streaming
            if self._replay_capture:
                try:
                    full_content = "".join(self._content_chunks) if self._content_chunks else ""
                    self._replay_capture.set_llm_response(
                        text=full_content,
                        finish_reason=self._finish_reason,
                    )
                    # Attach replay snapshot to span
                    object.__setattr__(self._span, "replay_snapshot", self._replay_capture.build())
                except Exception as e:
                    logger.debug(f"Failed to capture streaming replay data: {e}")

            # Mark as successful
            self._span.set_status(SpanStatus.SUCCESS)

        except Exception as e:
            logger.debug(f"Failed to finalize span: {e}")


class TracedAsyncChatCompletionStream:
    """Wrapper for async streaming chat completion responses."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        tracer: Tracer,
        start_time: int,
        replay_capture: Any = None,
    ) -> None:
        """Initialize the traced async stream.

        Args:
            stream: The original async stream
            span: The span to record events on
            tracer: The tracer instance
            start_time: Start time in nanoseconds
            replay_capture: Optional ReplayCapture instance
        """
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._start_time = start_time
        self._first_token_time: int | None = None
        self._content_chunks: list[str] = []
        self._finish_reason: str | None = None
        self._replay_capture = replay_capture

    async def __aenter__(self) -> TracedAsyncChatCompletionStream:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and finalize span."""
        try:
            if exc_type is None:
                self._finalize_span()
            else:
                self._span.set_status(SpanStatus.ERROR, str(exc_val))

        finally:
            self._span.end()

    async def __aiter__(self) -> Any:
        """Async iterate over stream chunks."""
        try:
            async for chunk in self._stream:
                self._process_chunk(chunk)
                yield chunk

        except Exception as e:
            self._span.set_status(SpanStatus.ERROR, str(e))
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process a streaming chunk.

        Args:
            chunk: The streaming chunk
        """
        try:
            # Capture first token time
            if self._first_token_time is None:
                self._first_token_time = monotonic_ns()

            # Extract content from chunk
            if hasattr(chunk, "choices") and chunk.choices:
                first_choice = chunk.choices[0]

                # Get content delta
                if hasattr(first_choice, "delta"):
                    delta = first_choice.delta
                    if hasattr(delta, "content") and delta.content:
                        self._content_chunks.append(delta.content)

                # Get finish reason
                if hasattr(first_choice, "finish_reason") and first_choice.finish_reason:
                    self._finish_reason = first_choice.finish_reason

        except Exception as e:
            logger.debug(f"Failed to process chunk: {e}")

    def _finalize_span(self) -> None:
        """Finalize the span with aggregated data."""
        try:
            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(self._start_time, end_time)
            self._span.set_attribute("llm.latency_ms", latency_ms)

            # Time to first token
            if self._first_token_time is not None:
                ttft_ms = duration_ms(self._start_time, self._first_token_time)
                self._span.set_attribute("llm.time_to_first_token_ms", ttft_ms)

            # Aggregated content
            if self._content_chunks:
                full_content = "".join(self._content_chunks)
                self._span.add_event(
                    name="llm.response",
                    attributes={"content": full_content},
                )

            # Finish reason
            if self._finish_reason:
                self._span.set_attribute("llm.finish_reason", self._finish_reason)

            # Finalize replay capture for streaming
            if self._replay_capture:
                try:
                    full_content = "".join(self._content_chunks) if self._content_chunks else ""
                    self._replay_capture.set_llm_response(
                        text=full_content,
                        finish_reason=self._finish_reason,
                    )
                    # Attach replay snapshot to span
                    object.__setattr__(self._span, "replay_snapshot", self._replay_capture.build())
                except Exception as e:
                    logger.debug(f"Failed to capture streaming replay data: {e}")

            # Mark as successful
            self._span.set_status(SpanStatus.SUCCESS)

        except Exception as e:
            logger.debug(f"Failed to finalize span: {e}")
