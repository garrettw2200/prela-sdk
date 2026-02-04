"""Base classes and utilities for instrumenting external libraries.

This module provides the foundation for auto-instrumentation of LLM SDKs
and agent frameworks. It includes:

1. Instrumentor abstract base class
2. Monkey-patching utilities for function wrapping
3. Attribute extraction helpers for LLM requests/responses
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable
from types import ModuleType

if TYPE_CHECKING:
    # Avoid circular imports - Tracer will be implemented separately
    from prela.core.tracer import Tracer

logger = logging.getLogger(__name__)

# Attribute name for storing original functions on modules
_ORIGINALS_ATTR = "__prela_originals__"


class Instrumentor(ABC):
    """Abstract base class for library instrumentors.

    Instrumentors provide automatic tracing for external libraries by
    monkey-patching their functions to create spans around operations.

    Example:
        ```python
        class OpenAIInstrumentor(Instrumentor):
            def instrument(self, tracer: Tracer) -> None:
                # Wrap OpenAI API calls
                wrap_function(openai, "create", wrapper)

            def uninstrument(self) -> None:
                # Restore original functions
                unwrap_function(openai, "create")

            @property
            def is_instrumented(self) -> bool:
                return hasattr(openai, _ORIGINALS_ATTR)
        ```
    """

    @abstractmethod
    def instrument(self, tracer: Tracer) -> None:
        """Enable instrumentation for this library.

        This method should wrap the library's functions to create spans
        automatically. It should be idempotent - calling it multiple times
        should not create multiple layers of wrapping.

        Args:
            tracer: The tracer to use for creating spans

        Raises:
            RuntimeError: If instrumentation fails
        """
        pass

    @abstractmethod
    def uninstrument(self) -> None:
        """Disable instrumentation and restore original functions.

        This method should unwrap all previously wrapped functions and
        restore the library to its original state. It should be idempotent -
        calling it when not instrumented should be a no-op.

        Raises:
            RuntimeError: If uninstrumentation fails
        """
        pass

    @property
    @abstractmethod
    def is_instrumented(self) -> bool:
        """Check if this library is currently instrumented.

        Returns:
            True if instrumentation is active, False otherwise
        """
        pass


def wrap_function(
    module: ModuleType,
    func_name: str,
    wrapper: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    """Wrap a function on a module with instrumentation.

    This function replaces `module.func_name` with a wrapped version created
    by calling `wrapper(original_func)`. The original function is stored in
    `module.__prela_originals__` for later restoration.

    If the function is already wrapped (i.e., it exists in __prela_originals__),
    this function does nothing to prevent double-wrapping.

    Args:
        module: The module containing the function to wrap
        func_name: Name of the function/attribute to wrap
        wrapper: A function that takes the original function and returns
                a wrapped version. Should preserve the function signature.

    Raises:
        AttributeError: If the function doesn't exist on the module
        RuntimeError: If wrapping fails

    Example:
        ```python
        def trace_wrapper(original_func):
            def wrapper(*args, **kwargs):
                with tracer.span("api_call"):
                    return original_func(*args, **kwargs)
            return wrapper

        wrap_function(openai, "create", trace_wrapper)
        ```
    """
    # Check if the attribute exists
    if not hasattr(module, func_name):
        raise AttributeError(
            f"Module {module.__name__} has no attribute '{func_name}'"
        )

    # Get or create the originals dict
    if not hasattr(module, _ORIGINALS_ATTR):
        setattr(module, _ORIGINALS_ATTR, {})

    originals = getattr(module, _ORIGINALS_ATTR)

    # Check if already wrapped
    if func_name in originals:
        logger.debug(
            f"{module.__name__}.{func_name} is already wrapped, skipping"
        )
        return

    # Store the original function
    original_func = getattr(module, func_name)
    originals[func_name] = original_func

    # Create and set the wrapped version
    try:
        wrapped_func = wrapper(original_func)
        setattr(module, func_name, wrapped_func)
        logger.debug(f"Successfully wrapped {module.__name__}.{func_name}")
    except Exception as e:
        # Restore original on failure
        del originals[func_name]
        if not originals:
            delattr(module, _ORIGINALS_ATTR)
        raise RuntimeError(
            f"Failed to wrap {module.__name__}.{func_name}: {e}"
        ) from e


def unwrap_function(module: ModuleType, func_name: str) -> None:
    """Restore a wrapped function to its original implementation.

    This function looks up the original implementation in
    `module.__prela_originals__` and restores it to `module.func_name`.

    If the function is not currently wrapped, this function does nothing.

    Args:
        module: The module containing the wrapped function
        func_name: Name of the function/attribute to unwrap

    Example:
        ```python
        unwrap_function(openai, "create")
        ```
    """
    # Check if the module has any wrapped functions
    if not hasattr(module, _ORIGINALS_ATTR):
        logger.debug(
            f"Module {module.__name__} has no wrapped functions, skipping"
        )
        return

    originals = getattr(module, _ORIGINALS_ATTR)

    # Check if this specific function is wrapped
    if func_name not in originals:
        logger.debug(
            f"{module.__name__}.{func_name} is not wrapped, skipping"
        )
        return

    # Restore the original function
    original_func = originals.pop(func_name)
    setattr(module, func_name, original_func)

    # Clean up the originals dict if empty
    if not originals:
        delattr(module, _ORIGINALS_ATTR)

    logger.debug(f"Successfully unwrapped {module.__name__}.{func_name}")


def extract_llm_request_attributes(
    model: str,
    messages: list[dict[str, Any]] | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract standardized attributes from an LLM request.

    This function extracts common attributes from LLM API calls in a
    vendor-agnostic format. It handles both chat-style (messages) and
    completion-style (prompt) APIs.

    Args:
        model: The model identifier (e.g., "gpt-4", "claude-3-opus")
        messages: Chat messages (list of dicts) or text prompt (string)
        **kwargs: Additional request parameters (temperature, max_tokens, etc.)

    Returns:
        Dictionary of span attributes following semantic conventions:
        - llm.model: Model identifier
        - llm.request.type: "chat" or "completion"
        - llm.request.messages: Message count for chat
        - llm.request.prompt_length: Character count for completion
        - llm.request.temperature: Sampling temperature (if provided)
        - llm.request.max_tokens: Maximum tokens (if provided)
        - llm.request.top_p: Nucleus sampling (if provided)
        - llm.request.stream: Whether streaming is enabled

    Example:
        ```python
        attrs = extract_llm_request_attributes(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        # Returns: {
        #   "llm.model": "gpt-4",
        #   "llm.request.type": "chat",
        #   "llm.request.messages": 1,
        #   "llm.request.temperature": 0.7,
        #   "llm.request.max_tokens": 100
        # }
        ```
    """
    attributes: dict[str, Any] = {"llm.model": model}

    # Determine request type and extract message/prompt info
    if messages is not None:
        if isinstance(messages, list):
            # Chat-style API
            attributes["llm.request.type"] = "chat"
            attributes["llm.request.messages"] = len(messages)
        elif isinstance(messages, str):
            # Completion-style API with text prompt
            attributes["llm.request.type"] = "completion"
            attributes["llm.request.prompt_length"] = len(messages)

    # Extract common parameters
    # Temperature
    if "temperature" in kwargs:
        attributes["llm.request.temperature"] = kwargs["temperature"]

    # Max tokens (handle various parameter names)
    for param in ["max_tokens", "max_completion_tokens", "maxTokens"]:
        if param in kwargs:
            attributes["llm.request.max_tokens"] = kwargs[param]
            break

    # Top-p sampling
    if "top_p" in kwargs:
        attributes["llm.request.top_p"] = kwargs["top_p"]

    # Streaming
    if "stream" in kwargs:
        attributes["llm.request.stream"] = kwargs["stream"]

    # Stop sequences
    if "stop" in kwargs:
        stop = kwargs["stop"]
        if isinstance(stop, list):
            attributes["llm.request.stop_sequences"] = len(stop)
        elif stop is not None:
            attributes["llm.request.stop_sequences"] = 1

    # Frequency penalty
    if "frequency_penalty" in kwargs:
        attributes["llm.request.frequency_penalty"] = kwargs["frequency_penalty"]

    # Presence penalty
    if "presence_penalty" in kwargs:
        attributes["llm.request.presence_penalty"] = kwargs["presence_penalty"]

    return attributes


def extract_llm_response_attributes(
    response: Any,
    vendor: str,
) -> dict[str, Any]:
    """Extract standardized attributes from an LLM response.

    This function extracts common attributes from LLM API responses in a
    vendor-agnostic format. It handles different response structures from
    OpenAI, Anthropic, and other providers.

    Args:
        response: The response object from the LLM API
        vendor: The vendor identifier ("openai", "anthropic", etc.)

    Returns:
        Dictionary of span attributes:
        - llm.response.model: Actual model used (may differ from request)
        - llm.response.id: Response/completion ID
        - llm.response.finish_reason: Why generation stopped
        - llm.usage.prompt_tokens: Input token count
        - llm.usage.completion_tokens: Output token count
        - llm.usage.total_tokens: Total token count

    Example:
        ```python
        # OpenAI response
        attrs = extract_llm_response_attributes(
            response=openai_response,
            vendor="openai"
        )

        # Anthropic response
        attrs = extract_llm_response_attributes(
            response=anthropic_response,
            vendor="anthropic"
        )
        ```
    """
    attributes: dict[str, Any] = {}

    if vendor == "openai":
        # OpenAI response structure
        # Handle both dict and object responses
        if isinstance(response, dict):
            # Response ID
            if "id" in response:
                attributes["llm.response.id"] = response["id"]

            # Model
            if "model" in response:
                attributes["llm.response.model"] = response["model"]

            # Usage stats
            if "usage" in response:
                usage = response["usage"]
                if "prompt_tokens" in usage:
                    attributes["llm.usage.prompt_tokens"] = usage["prompt_tokens"]
                if "completion_tokens" in usage:
                    attributes["llm.usage.completion_tokens"] = usage["completion_tokens"]
                if "total_tokens" in usage:
                    attributes["llm.usage.total_tokens"] = usage["total_tokens"]

            # Finish reason (from first choice)
            if "choices" in response and response["choices"]:
                first_choice = response["choices"][0]
                if "finish_reason" in first_choice:
                    attributes["llm.response.finish_reason"] = first_choice["finish_reason"]
        else:
            # Object response (openai SDK objects)
            if hasattr(response, "id"):
                attributes["llm.response.id"] = response.id
            if hasattr(response, "model"):
                attributes["llm.response.model"] = response.model
            if hasattr(response, "usage"):
                if hasattr(response.usage, "prompt_tokens"):
                    attributes["llm.usage.prompt_tokens"] = response.usage.prompt_tokens
                if hasattr(response.usage, "completion_tokens"):
                    attributes["llm.usage.completion_tokens"] = response.usage.completion_tokens
                if hasattr(response.usage, "total_tokens"):
                    attributes["llm.usage.total_tokens"] = response.usage.total_tokens
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    attributes["llm.response.finish_reason"] = first_choice.finish_reason

    elif vendor == "anthropic":
        # Anthropic response structure
        if isinstance(response, dict):
            # Response ID
            if "id" in response:
                attributes["llm.response.id"] = response["id"]

            # Model
            if "model" in response:
                attributes["llm.response.model"] = response["model"]

            # Usage stats
            if "usage" in response:
                usage = response["usage"]
                if "input_tokens" in usage:
                    attributes["llm.usage.prompt_tokens"] = usage["input_tokens"]
                if "output_tokens" in usage:
                    attributes["llm.usage.completion_tokens"] = usage["output_tokens"]
                # Calculate total
                if "input_tokens" in usage and "output_tokens" in usage:
                    attributes["llm.usage.total_tokens"] = (
                        usage["input_tokens"] + usage["output_tokens"]
                    )

            # Stop reason
            if "stop_reason" in response:
                attributes["llm.response.finish_reason"] = response["stop_reason"]
        else:
            # Object response
            if hasattr(response, "id"):
                attributes["llm.response.id"] = response.id
            if hasattr(response, "model"):
                attributes["llm.response.model"] = response.model
            if hasattr(response, "usage"):
                if hasattr(response.usage, "input_tokens"):
                    attributes["llm.usage.prompt_tokens"] = response.usage.input_tokens
                if hasattr(response.usage, "output_tokens"):
                    attributes["llm.usage.completion_tokens"] = response.usage.output_tokens
                    # Calculate total
                    if hasattr(response.usage, "input_tokens"):
                        attributes["llm.usage.total_tokens"] = (
                            response.usage.input_tokens + response.usage.output_tokens
                        )
            if hasattr(response, "stop_reason"):
                attributes["llm.response.finish_reason"] = response.stop_reason

    return attributes
