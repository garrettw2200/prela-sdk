"""Instrumentation for Anthropic SDK (anthropic>=0.40.0).

This module provides automatic tracing for Anthropic's Claude API, including:
- Synchronous and asynchronous messages.create calls
- Streaming responses (MessageStream and AsyncMessageStream)
- Tool use detection and tracking
- Extended thinking blocks (if enabled)
- Comprehensive error handling

Example:
    ```python
    from prela.instrumentation.anthropic import AnthropicInstrumentor
    from prela.core.tracer import Tracer
    import anthropic

    tracer = Tracer()
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer)

    # Now all Anthropic API calls will be automatically traced
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
"""

from __future__ import annotations

import logging
import time
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


class AnthropicInstrumentor(Instrumentor):
    """Instrumentor for Anthropic SDK.

    Patches the following methods:
    - anthropic.Anthropic.messages.create (sync)
    - anthropic.AsyncAnthropic.messages.create (async)
    - anthropic.Anthropic.messages.stream (sync)
    - anthropic.AsyncAnthropic.messages.stream (async)

    Captures detailed information about requests, responses, tool usage,
    and streaming events.
    """

    def __init__(self) -> None:
        """Initialize the Anthropic instrumentor."""
        self._tracer: Tracer | None = None
        self._anthropic_module: Any = None
        self._messages_module: Any = None
        self._async_messages_module: Any = None

    def instrument(self, tracer: Tracer) -> None:
        """Enable instrumentation for Anthropic SDK.

        Args:
            tracer: The tracer to use for creating spans

        Raises:
            ImportError: If anthropic package is not installed
            RuntimeError: If instrumentation fails
        """
        if self.is_instrumented:
            logger.debug("Anthropic SDK is already instrumented, skipping")
            return

        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is not installed. "
                "Install it with: pip install anthropic>=0.40.0"
            ) from e

        self._tracer = tracer
        self._anthropic_module = anthropic

        try:
            # Get the messages modules for sync and async
            if hasattr(anthropic, "Anthropic"):
                client = anthropic.Anthropic.__new__(anthropic.Anthropic)
                if hasattr(client, "messages"):
                    self._messages_module = client.messages.__class__

            if hasattr(anthropic, "AsyncAnthropic"):
                async_client = anthropic.AsyncAnthropic.__new__(
                    anthropic.AsyncAnthropic
                )
                if hasattr(async_client, "messages"):
                    self._async_messages_module = async_client.messages.__class__

            # Wrap sync messages.create
            if self._messages_module is not None:
                wrap_function(
                    self._messages_module,
                    "create",
                    lambda orig: self._create_messages_wrapper(orig, is_async=False),
                )
                logger.debug("Wrapped anthropic.Anthropic.messages.create")

            # Wrap async messages.create
            if self._async_messages_module is not None:
                wrap_function(
                    self._async_messages_module,
                    "create",
                    lambda orig: self._create_messages_wrapper(orig, is_async=True),
                )
                logger.debug("Wrapped anthropic.AsyncAnthropic.messages.create")

            # Wrap sync messages.stream
            if self._messages_module is not None:
                wrap_function(
                    self._messages_module,
                    "stream",
                    lambda orig: self._create_stream_wrapper(orig, is_async=False),
                )
                logger.debug("Wrapped anthropic.Anthropic.messages.stream")

            # Wrap async messages.stream
            if self._async_messages_module is not None:
                wrap_function(
                    self._async_messages_module,
                    "stream",
                    lambda orig: self._create_stream_wrapper(orig, is_async=True),
                )
                logger.debug("Wrapped anthropic.AsyncAnthropic.messages.stream")

            logger.info("Successfully instrumented Anthropic SDK")

        except Exception as e:
            self._tracer = None
            self._anthropic_module = None
            self._messages_module = None
            self._async_messages_module = None
            raise RuntimeError(f"Failed to instrument Anthropic SDK: {e}") from e

    def uninstrument(self) -> None:
        """Disable instrumentation and restore original functions."""
        if not self.is_instrumented:
            logger.debug("Anthropic SDK is not instrumented, skipping")
            return

        try:
            # Unwrap sync methods
            if self._messages_module is not None:
                unwrap_function(self._messages_module, "create")
                unwrap_function(self._messages_module, "stream")

            # Unwrap async methods
            if self._async_messages_module is not None:
                unwrap_function(self._async_messages_module, "create")
                unwrap_function(self._async_messages_module, "stream")

            logger.info("Successfully uninstrumented Anthropic SDK")

        finally:
            self._tracer = None
            self._anthropic_module = None
            self._messages_module = None
            self._async_messages_module = None

    @property
    def is_instrumented(self) -> bool:
        """Check if Anthropic SDK is currently instrumented."""
        return (
            self._tracer is not None
            and self._messages_module is not None
            and hasattr(self._messages_module, _ORIGINALS_ATTR)
        )

    def _create_messages_wrapper(
        self, original_func: Callable[..., Any], is_async: bool
    ) -> Callable[..., Any]:
        """Create a wrapper for messages.create method.

        Args:
            original_func: The original create function
            is_async: Whether this is an async function

        Returns:
            Wrapped function that creates spans
        """
        if is_async:

            @wraps(original_func)
            async def async_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                return await self._trace_messages_create(
                    original_func, self_obj, is_async=True, *args, **kwargs
                )

            return async_wrapper
        else:

            @wraps(original_func)
            def sync_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                return self._trace_messages_create(
                    original_func, self_obj, is_async=False, *args, **kwargs
                )

            return sync_wrapper

    def _create_stream_wrapper(
        self, original_func: Callable[..., Any], is_async: bool
    ) -> Callable[..., Any]:
        """Create a wrapper for messages.stream method.

        Args:
            original_func: The original stream function
            is_async: Whether this is an async function

        Returns:
            Wrapped function that creates spans and wraps streams
        """
        if is_async:

            @wraps(original_func)
            async def async_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                return await self._trace_messages_stream(
                    original_func, self_obj, is_async=True, *args, **kwargs
                )

            return async_wrapper
        else:

            @wraps(original_func)
            def sync_wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                return self._trace_messages_stream(
                    original_func, self_obj, is_async=False, *args, **kwargs
                )

            return sync_wrapper

    def _trace_messages_create(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        is_async: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a messages.create call (sync or async).

        Args:
            original_func: The original create function
            self_obj: The messages object (self)
            is_async: Whether this is an async call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The response from the API call
        """
        if is_async:
            return self._trace_messages_create_async(
                original_func, self_obj, *args, **kwargs
            )
        else:
            return self._trace_messages_create_sync(
                original_func, self_obj, *args, **kwargs
            )

    def _trace_messages_create_sync(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a synchronous messages.create call.

        Args:
            original_func: The original create function
            self_obj: The messages object (self)
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
        system = kwargs.get("system")
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")

        # Start timing
        start_time = monotonic_ns()

        # Create span
        span = self._tracer.start_span(
            name="anthropic.messages.create",
            span_type=SpanType.LLM,
        )

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
                system=system,  # Anthropic-specific
                # Capture all other parameters
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model", "messages", "temperature", "max_tokens", "system"]
                },
            )

        try:
            # Set request attributes
            span.set_attribute("llm.vendor", "anthropic")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.model", model)

            if system:
                span.set_attribute("llm.system", system)
            if temperature is not None:
                span.set_attribute("llm.temperature", temperature)
            if max_tokens is not None:
                span.set_attribute("llm.max_tokens", max_tokens)

            # Add request event
            span.add_event(
                name="llm.request",
                attributes={
                    "messages": messages,
                    **({"system": system} if system else {}),
                },
            )

            # Make the API call
            response = original_func(self_obj, *args, **kwargs)

            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(start_time, end_time)
            span.set_attribute("llm.latency_ms", latency_ms)

            # Extract response attributes
            self._extract_response_attributes(span, response)

            # Add response event (serialize content to avoid TextBlock serialization issues)
            span.add_event(
                name="llm.response",
                attributes={"content": self._serialize_content(response.content)},
            )

            # Capture replay data if enabled
            if replay_capture:
                try:
                    # Extract response text
                    response_text = ""
                    if hasattr(response, "content") and response.content:
                        for block in response.content:
                            if hasattr(block, "type") and block.type == "text":
                                if hasattr(block, "text"):
                                    response_text += block.text

                    # Capture response
                    replay_capture.set_llm_response(
                        text=response_text,
                        finish_reason=getattr(response, "stop_reason", None),
                        model=getattr(response, "model", None),
                        prompt_tokens=getattr(response.usage, "input_tokens", None)
                        if hasattr(response, "usage")
                        else None,
                        completion_tokens=getattr(response.usage, "output_tokens", None)
                        if hasattr(response, "usage")
                        else None,
                    )

                    # Capture model info
                    if hasattr(response, "id"):
                        replay_capture.set_model_info(
                            model=getattr(response, "model", None),
                            id=response.id,
                        )

                    # Attach to span
                    object.__setattr__(span, "replay_snapshot", replay_capture.build())

                except Exception as e:
                    logger.debug(f"Failed to capture replay data: {e}")

            # Handle tool use if present
            if hasattr(response, "stop_reason") and response.stop_reason == "tool_use":
                self._handle_tool_use(span, response)

            # Handle extended thinking if present
            self._handle_thinking_blocks(span, response)

            # Mark as successful
            span.set_status(SpanStatus.SUCCESS)

            return response

        except Exception as e:
            # Handle errors
            self._handle_error(span, e)
            raise

        finally:
            span.end()

    async def _trace_messages_create_async(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace an asynchronous messages.create call.

        Args:
            original_func: The original create function
            self_obj: The messages object (self)
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
        system = kwargs.get("system")
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")

        # Start timing
        start_time = monotonic_ns()

        # Create span
        span = self._tracer.start_span(
            name="anthropic.messages.create",
            span_type=SpanType.LLM,
        )

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
                system=system,  # Anthropic-specific
                # Capture all other parameters
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model", "messages", "temperature", "max_tokens", "system"]
                },
            )

        try:
            # Set request attributes
            span.set_attribute("llm.vendor", "anthropic")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.model", model)

            if system:
                span.set_attribute("llm.system", system)
            if temperature is not None:
                span.set_attribute("llm.temperature", temperature)
            if max_tokens is not None:
                span.set_attribute("llm.max_tokens", max_tokens)

            # Add request event
            span.add_event(
                name="llm.request",
                attributes={
                    "messages": messages,
                    **({"system": system} if system else {}),
                },
            )

            # Make the API call
            response = await original_func(self_obj, *args, **kwargs)

            # Calculate latency
            end_time = monotonic_ns()
            latency_ms = duration_ms(start_time, end_time)
            span.set_attribute("llm.latency_ms", latency_ms)

            # Extract response attributes
            self._extract_response_attributes(span, response)

            # Add response event (serialize content to avoid TextBlock serialization issues)
            span.add_event(
                name="llm.response",
                attributes={"content": self._serialize_content(response.content)},
            )

            # Capture replay data if enabled
            if replay_capture:
                try:
                    # Extract response text
                    response_text = ""
                    if hasattr(response, "content") and response.content:
                        for block in response.content:
                            if hasattr(block, "type") and block.type == "text":
                                if hasattr(block, "text"):
                                    response_text += block.text

                    # Capture response
                    replay_capture.set_llm_response(
                        text=response_text,
                        finish_reason=getattr(response, "stop_reason", None),
                        model=getattr(response, "model", None),
                        prompt_tokens=getattr(response.usage, "input_tokens", None)
                        if hasattr(response, "usage")
                        else None,
                        completion_tokens=getattr(response.usage, "output_tokens", None)
                        if hasattr(response, "usage")
                        else None,
                    )

                    # Capture model info
                    if hasattr(response, "id"):
                        replay_capture.set_model_info(
                            model=getattr(response, "model", None),
                            id=response.id,
                        )

                    # Attach to span
                    object.__setattr__(span, "replay_snapshot", replay_capture.build())

                except Exception as e:
                    logger.debug(f"Failed to capture replay data: {e}")

            # Handle tool use if present
            if hasattr(response, "stop_reason") and response.stop_reason == "tool_use":
                self._handle_tool_use(span, response)

            # Handle extended thinking if present
            self._handle_thinking_blocks(span, response)

            # Mark as successful
            span.set_status(SpanStatus.SUCCESS)

            return response

        except Exception as e:
            # Handle errors
            self._handle_error(span, e)
            raise

        finally:
            span.end()

    def _trace_messages_stream(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        is_async: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a messages.stream call (sync or async).

        Args:
            original_func: The original stream function
            self_obj: The messages object (self)
            is_async: Whether this is an async stream
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The wrapped stream object
        """
        if is_async:
            return self._trace_messages_stream_async(
                original_func, self_obj, *args, **kwargs
            )
        else:
            return self._trace_messages_stream_sync(
                original_func, self_obj, *args, **kwargs
            )

    def _trace_messages_stream_sync(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace a synchronous messages.stream call.

        Args:
            original_func: The original stream function
            self_obj: The messages object (self)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The wrapped stream object
        """
        if self._tracer is None:
            return original_func(self_obj, *args, **kwargs)

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")

        # Start timing
        start_time = monotonic_ns()

        # Create span
        span = self._tracer.start_span(
            name="anthropic.messages.stream",
            span_type=SpanType.LLM,
        )

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
                system=system,  # Anthropic-specific
                # Capture all other parameters
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model", "messages", "temperature", "max_tokens", "system"]
                },
            )

        # Set request attributes
        span.set_attribute("llm.vendor", "anthropic")
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.request.model", model)
        span.set_attribute("llm.stream", True)

        if system:
            span.set_attribute("llm.system", system)
        if temperature is not None:
            span.set_attribute("llm.temperature", temperature)
        if max_tokens is not None:
            span.set_attribute("llm.max_tokens", max_tokens)

        # Add request event
        span.add_event(
            name="llm.request",
            attributes={
                "messages": messages,
                **({"system": system} if system else {}),
            },
        )

        try:
            # Create the stream
            stream = original_func(self_obj, *args, **kwargs)

            # Wrap the stream to capture events
            return TracedMessageStream(
                stream=stream,
                span=span,
                tracer=self._tracer,
                start_time=start_time,
                replay_capture=replay_capture,
            )

        except Exception as e:
            # Handle errors during stream creation
            self._handle_error(span, e)
            span.end()
            raise

    async def _trace_messages_stream_async(
        self,
        original_func: Callable[..., Any],
        self_obj: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace an asynchronous messages.stream call.

        Args:
            original_func: The original stream function
            self_obj: The messages object (self)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The wrapped async stream object
        """
        if self._tracer is None:
            return await original_func(self_obj, *args, **kwargs)

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")

        # Start timing
        start_time = monotonic_ns()

        # Create span
        span = self._tracer.start_span(
            name="anthropic.messages.stream",
            span_type=SpanType.LLM,
        )

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
                system=system,  # Anthropic-specific
                # Capture all other parameters
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model", "messages", "temperature", "max_tokens", "system"]
                },
            )

        # Set request attributes
        span.set_attribute("llm.vendor", "anthropic")
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.request.model", model)
        span.set_attribute("llm.stream", True)

        if system:
            span.set_attribute("llm.system", system)
        if temperature is not None:
            span.set_attribute("llm.temperature", temperature)
        if max_tokens is not None:
            span.set_attribute("llm.max_tokens", max_tokens)

        # Add request event
        span.add_event(
            name="llm.request",
            attributes={
                "messages": messages,
                **({"system": system} if system else {}),
            },
        )

        try:
            # Create the async stream
            stream = await original_func(self_obj, *args, **kwargs)

            # Wrap the stream to capture events
            return TracedAsyncMessageStream(
                stream=stream,
                span=span,
                tracer=self._tracer,
                start_time=start_time,
                replay_capture=replay_capture,
            )

        except Exception as e:
            # Handle errors during stream creation
            self._handle_error(span, e)
            span.end()
            raise

    def _serialize_content(self, content: Any) -> list[dict[str, Any]]:
        """Serialize response content blocks to JSON-serializable format.

        Args:
            content: The content blocks from the API response

        Returns:
            List of serialized content blocks as dicts
        """
        serialized = []
        try:
            for block in content:
                # Try model_dump() first (Pydantic v2)
                if hasattr(block, "model_dump"):
                    serialized.append(block.model_dump())
                # Fall back to dict() (Pydantic v1)
                elif hasattr(block, "dict"):
                    serialized.append(block.dict())
                # Manual extraction as last resort
                else:
                    block_dict = {"type": getattr(block, "type", "unknown")}
                    if hasattr(block, "text"):
                        block_dict["text"] = block.text
                    if hasattr(block, "id"):
                        block_dict["id"] = block.id
                    if hasattr(block, "name"):
                        block_dict["name"] = block.name
                    if hasattr(block, "input"):
                        block_dict["input"] = block.input
                    serialized.append(block_dict)
        except Exception as e:
            logger.debug(f"Failed to serialize content blocks: {e}")
            # Return empty list on failure
            return []

        return serialized

    def _extract_response_attributes(self, span: Any, response: Any) -> None:
        """Extract attributes from a response object.

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
                if hasattr(usage, "input_tokens"):
                    span.set_attribute("llm.input_tokens", usage.input_tokens)
                if hasattr(usage, "output_tokens"):
                    span.set_attribute("llm.output_tokens", usage.output_tokens)

            # Stop reason
            if hasattr(response, "stop_reason"):
                span.set_attribute("llm.stop_reason", response.stop_reason)

        except Exception as e:
            # Don't let attribute extraction failures break the instrumentation
            logger.debug(f"Failed to extract response attributes: {e}")

    def _handle_tool_use(self, span: Any, response: Any) -> None:
        """Handle tool use in the response.

        Args:
            span: The span to add tool use information to
            response: The response object containing tool use
        """
        try:
            if not hasattr(response, "content"):
                return

            tool_calls = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_call = {
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                    tool_calls.append(tool_call)

            if tool_calls:
                span.add_event(
                    name="llm.tool_use",
                    attributes={"tool_calls": tool_calls},
                )

        except Exception as e:
            # Don't let tool use handling failures break the instrumentation
            logger.debug(f"Failed to handle tool use: {e}")

    def _handle_thinking_blocks(self, span: Any, response: Any) -> None:
        """Handle extended thinking blocks in the response.

        Args:
            span: The span to add thinking information to
            response: The response object that may contain thinking blocks
        """
        try:
            if not hasattr(response, "content"):
                return

            thinking_content = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "thinking":
                    if hasattr(block, "thinking"):
                        thinking_content.append(block.thinking)

            if thinking_content:
                span.add_event(
                    name="llm.thinking",
                    attributes={"thinking": thinking_content},
                )

        except Exception as e:
            # Don't let thinking block handling failures break the instrumentation
            logger.debug(f"Failed to handle thinking blocks: {e}")

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

            # Handle anthropic-specific errors
            if hasattr(error, "status_code"):
                error_attrs["error.status_code"] = error.status_code

            span.add_event(name="error", attributes=error_attrs)

        except Exception as e:
            # Don't let error handling failures break the instrumentation
            logger.debug(f"Failed to handle error: {e}")


class TracedMessageStream:
    """Wrapper for MessageStream that captures streaming events."""

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
            stream: The original MessageStream
            span: The span to record events on
            tracer: The tracer instance
            start_time: Start time in nanoseconds
            replay_capture: Optional ReplayCapture instance for replay data
        """
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._start_time = start_time
        self._replay_capture = replay_capture
        self._first_token_time: int | None = None
        self._text_content: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._thinking_content: list[str] = []
        self._streaming_chunks: list[dict[str, Any]] = []

    def __enter__(self) -> TracedMessageStream:
        """Enter context manager."""
        self._message_stream = self._stream.__enter__()
        return self

    @property
    def text_stream(self):
        """Expose the underlying MessageStream's text_stream iterator.

        This allows users to iterate over text deltas:

        ```python
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        ```
        """
        return self._message_stream.text_stream

    def get_final_message(self):
        """Get the final message from the stream.

        Returns:
            The final Message object with complete content and usage data
        """
        return self._message_stream.get_final_message()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and finalize span."""
        try:
            self._stream.__exit__(exc_type, exc_val, exc_tb)

            if exc_type is None:
                # Success - record final attributes
                self._finalize_span()
            else:
                # Error during streaming
                self._span.set_status(SpanStatus.ERROR, str(exc_val))

        finally:
            self._span.end()

    def __iter__(self) -> Any:
        """Iterate over stream events."""
        try:
            for event in self._stream:
                self._process_event(event)
                yield event

        except Exception as e:
            self._span.set_status(SpanStatus.ERROR, str(e))
            raise

    def _process_event(self, event: Any) -> None:
        """Process a streaming event.

        Args:
            event: The streaming event
        """
        try:
            event_type = getattr(event, "type", None)

            # Capture raw event for replay if enabled
            if self._replay_capture:
                try:
                    chunk_data = {
                        "type": event_type,
                    }

                    # Capture event-specific data
                    if event_type == "content_block_delta" and hasattr(event, "delta"):
                        delta = event.delta
                        chunk_data["delta"] = {
                            "type": getattr(delta, "type", None),
                            "text": getattr(delta, "text", None),
                        }
                    elif event_type == "content_block_start" and hasattr(event, "content_block"):
                        block = event.content_block
                        chunk_data["content_block"] = {
                            "type": getattr(block, "type", None),
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                        }
                    elif event_type == "message_delta":
                        if hasattr(event, "usage"):
                            usage = event.usage
                            chunk_data["usage"] = {
                                "output_tokens": getattr(usage, "output_tokens", None),
                            }
                        if hasattr(event, "delta"):
                            delta = event.delta
                            chunk_data["delta"] = {
                                "stop_reason": getattr(delta, "stop_reason", None),
                            }

                    self._streaming_chunks.append(chunk_data)
                except Exception as e:
                    logger.debug(f"Failed to capture streaming chunk: {e}")

            if event_type == "content_block_delta":
                # Capture first token time
                if self._first_token_time is None:
                    self._first_token_time = monotonic_ns()

                # Aggregate text content
                if hasattr(event, "delta"):
                    delta = event.delta
                    if hasattr(delta, "type") and delta.type == "text_delta":
                        if hasattr(delta, "text"):
                            self._text_content.append(delta.text)

            elif event_type == "content_block_start":
                # Detect tool use or thinking blocks
                if hasattr(event, "content_block"):
                    block = event.content_block
                    if hasattr(block, "type"):
                        if block.type == "tool_use":
                            self._tool_calls.append(
                                {
                                    "id": getattr(block, "id", None),
                                    "name": getattr(block, "name", None),
                                }
                            )
                        elif block.type == "thinking":
                            # Mark that we have thinking content
                            pass

            elif event_type == "message_delta":
                # Extract final usage stats
                if hasattr(event, "usage"):
                    usage = event.usage
                    if hasattr(usage, "output_tokens"):
                        self._span.set_attribute(
                            "llm.output_tokens", usage.output_tokens
                        )

                # Extract stop reason
                if hasattr(event, "delta"):
                    delta = event.delta
                    if hasattr(delta, "stop_reason"):
                        self._span.set_attribute("llm.stop_reason", delta.stop_reason)

        except Exception as e:
            # Don't let event processing failures break the stream
            logger.debug(f"Failed to process streaming event: {e}")

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

            # Aggregated text content
            if self._text_content:
                full_text = "".join(self._text_content)
                self._span.add_event(
                    name="llm.response",
                    attributes={"content": [{"type": "text", "text": full_text}]},
                )

            # Tool calls
            if self._tool_calls:
                self._span.add_event(
                    name="llm.tool_use",
                    attributes={"tool_calls": self._tool_calls},
                )

            # Finalize replay capture if enabled
            if self._replay_capture:
                try:
                    # Capture aggregated response text
                    full_text = "".join(self._text_content)
                    self._replay_capture.set_llm_response(
                        text=full_text,
                        finish_reason=self._span.attributes.get("llm.stop_reason"),
                    )

                    # Add streaming chunks
                    if self._streaming_chunks:
                        for chunk in self._streaming_chunks:
                            self._replay_capture.add_streaming_chunk(chunk)

                    # Attach to span
                    object.__setattr__(
                        self._span, "replay_snapshot", self._replay_capture.build()
                    )

                except Exception as e:
                    logger.debug(f"Failed to finalize replay capture: {e}")

            # Mark as successful
            self._span.set_status(SpanStatus.SUCCESS)

        except Exception as e:
            # Don't let finalization failures break the instrumentation
            logger.debug(f"Failed to finalize span: {e}")


class TracedAsyncMessageStream:
    """Wrapper for AsyncMessageStream that captures streaming events."""

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
            stream: The original AsyncMessageStream
            span: The span to record events on
            tracer: The tracer instance
            start_time: Start time in nanoseconds
            replay_capture: Optional ReplayCapture instance for replay data
        """
        self._stream = stream
        self._span = span
        self._tracer = tracer
        self._start_time = start_time
        self._replay_capture = replay_capture
        self._first_token_time: int | None = None
        self._text_content: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._thinking_content: list[str] = []
        self._streaming_chunks: list[dict[str, Any]] = []

    async def __aenter__(self) -> TracedAsyncMessageStream:
        """Enter async context manager."""
        self._message_stream = await self._stream.__aenter__()
        return self

    @property
    def text_stream(self):
        """Expose the underlying AsyncMessageStream's text_stream async iterator.

        This allows users to iterate over text deltas:

        ```python
        async with client.messages.stream(...) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
        ```
        """
        return self._message_stream.text_stream

    async def get_final_message(self):
        """Get the final message from the async stream.

        Returns:
            The final Message object with complete content and usage data
        """
        return await self._message_stream.get_final_message()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and finalize span."""
        try:
            await self._stream.__aexit__(exc_type, exc_val, exc_tb)

            if exc_type is None:
                # Success - record final attributes
                self._finalize_span()
            else:
                # Error during streaming
                self._span.set_status(SpanStatus.ERROR, str(exc_val))

        finally:
            self._span.end()

    async def __aiter__(self) -> Any:
        """Async iterate over stream events."""
        try:
            async for event in self._stream:
                self._process_event(event)
                yield event

        except Exception as e:
            self._span.set_status(SpanStatus.ERROR, str(e))
            raise

    def _process_event(self, event: Any) -> None:
        """Process a streaming event.

        Args:
            event: The streaming event
        """
        try:
            event_type = getattr(event, "type", None)

            # Capture raw event for replay if enabled
            if self._replay_capture:
                try:
                    chunk_data = {
                        "type": event_type,
                    }

                    # Capture event-specific data
                    if event_type == "content_block_delta" and hasattr(event, "delta"):
                        delta = event.delta
                        chunk_data["delta"] = {
                            "type": getattr(delta, "type", None),
                            "text": getattr(delta, "text", None),
                        }
                    elif event_type == "content_block_start" and hasattr(event, "content_block"):
                        block = event.content_block
                        chunk_data["content_block"] = {
                            "type": getattr(block, "type", None),
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                        }
                    elif event_type == "message_delta":
                        if hasattr(event, "usage"):
                            usage = event.usage
                            chunk_data["usage"] = {
                                "output_tokens": getattr(usage, "output_tokens", None),
                            }
                        if hasattr(event, "delta"):
                            delta = event.delta
                            chunk_data["delta"] = {
                                "stop_reason": getattr(delta, "stop_reason", None),
                            }

                    self._streaming_chunks.append(chunk_data)
                except Exception as e:
                    logger.debug(f"Failed to capture streaming chunk: {e}")

            if event_type == "content_block_delta":
                # Capture first token time
                if self._first_token_time is None:
                    self._first_token_time = monotonic_ns()

                # Aggregate text content
                if hasattr(event, "delta"):
                    delta = event.delta
                    if hasattr(delta, "type") and delta.type == "text_delta":
                        if hasattr(delta, "text"):
                            self._text_content.append(delta.text)

            elif event_type == "content_block_start":
                # Detect tool use or thinking blocks
                if hasattr(event, "content_block"):
                    block = event.content_block
                    if hasattr(block, "type"):
                        if block.type == "tool_use":
                            self._tool_calls.append(
                                {
                                    "id": getattr(block, "id", None),
                                    "name": getattr(block, "name", None),
                                }
                            )
                        elif block.type == "thinking":
                            # Mark that we have thinking content
                            pass

            elif event_type == "message_delta":
                # Extract final usage stats
                if hasattr(event, "usage"):
                    usage = event.usage
                    if hasattr(usage, "output_tokens"):
                        self._span.set_attribute(
                            "llm.output_tokens", usage.output_tokens
                        )

                # Extract stop reason
                if hasattr(event, "delta"):
                    delta = event.delta
                    if hasattr(delta, "stop_reason"):
                        self._span.set_attribute("llm.stop_reason", delta.stop_reason)

        except Exception as e:
            # Don't let event processing failures break the stream
            logger.debug(f"Failed to process streaming event: {e}")

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

            # Aggregated text content
            if self._text_content:
                full_text = "".join(self._text_content)
                self._span.add_event(
                    name="llm.response",
                    attributes={"content": [{"type": "text", "text": full_text}]},
                )

            # Tool calls
            if self._tool_calls:
                self._span.add_event(
                    name="llm.tool_use",
                    attributes={"tool_calls": self._tool_calls},
                )

            # Finalize replay capture if enabled
            if self._replay_capture:
                try:
                    # Capture aggregated response text
                    full_text = "".join(self._text_content)
                    self._replay_capture.set_llm_response(
                        text=full_text,
                        finish_reason=self._span.attributes.get("llm.stop_reason"),
                    )

                    # Add streaming chunks
                    if self._streaming_chunks:
                        for chunk in self._streaming_chunks:
                            self._replay_capture.add_streaming_chunk(chunk)

                    # Attach to span
                    object.__setattr__(
                        self._span, "replay_snapshot", self._replay_capture.build()
                    )

                except Exception as e:
                    logger.debug(f"Failed to finalize replay capture: {e}")

            # Mark as successful
            self._span.set_status(SpanStatus.SUCCESS)

        except Exception as e:
            # Don't let finalization failures break the instrumentation
            logger.debug(f"Failed to finalize span: {e}")
