"""Instrumentation package for automatic tracing of LLM SDKs and frameworks.

This package provides base classes and utilities for instrumenting external
libraries (OpenAI, Anthropic, LangChain, etc.) to automatically create spans
for LLM calls and agent operations.
"""

from __future__ import annotations

from prela.instrumentation.base import (
    Instrumentor,
    extract_llm_request_attributes,
    extract_llm_response_attributes,
    unwrap_function,
    wrap_function,
)

# Optional imports - only available if dependencies are installed
try:
    from prela.instrumentation.anthropic import AnthropicInstrumentor
except ImportError:
    AnthropicInstrumentor = None  # type: ignore

try:
    from prela.instrumentation.openai import OpenAIInstrumentor
except ImportError:
    OpenAIInstrumentor = None  # type: ignore

try:
    from prela.instrumentation.langchain import LangChainInstrumentor
except ImportError:
    LangChainInstrumentor = None  # type: ignore

try:
    from prela.instrumentation.llamaindex import LlamaIndexInstrumentor
except ImportError:
    LlamaIndexInstrumentor = None  # type: ignore

__all__ = [
    "Instrumentor",
    "wrap_function",
    "unwrap_function",
    "extract_llm_request_attributes",
    "extract_llm_response_attributes",
    "AnthropicInstrumentor",
    "OpenAIInstrumentor",
    "LangChainInstrumentor",
    "LlamaIndexInstrumentor",
]
