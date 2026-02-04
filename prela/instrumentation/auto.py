"""Auto-instrumentation for detecting and instrumenting LLM SDKs."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prela.core.tracer import Tracer

logger = logging.getLogger(__name__)

# Registry of available instrumentors
# Format: "library_name": ("module.path", "InstrumentorClassName")
INSTRUMENTORS = {
    # LLM providers
    "anthropic": ("prela.instrumentation.anthropic", "AnthropicInstrumentor"),
    "openai": ("prela.instrumentation.openai", "OpenAIInstrumentor"),
    # Agent frameworks
    "langchain": ("prela.instrumentation.langchain", "LangChainInstrumentor"),
    "llamaindex": ("prela.instrumentation.llamaindex", "LlamaIndexInstrumentor"),
    # Multi-agent frameworks
    "crewai": ("prela.instrumentation.multi_agent.crewai", "CrewAIInstrumentor"),
    "autogen": ("prela.instrumentation.multi_agent.autogen", "AutoGenInstrumentor"),
    "langgraph": ("prela.instrumentation.multi_agent.langgraph", "LangGraphInstrumentor"),
    "swarm": ("prela.instrumentation.multi_agent.swarm", "SwarmInstrumentor"),
}

# Package detection mapping
# Maps library name to the import name used to check if it's installed
PACKAGE_DETECTION = {
    "anthropic": "anthropic",
    "openai": "openai",
    "langchain": "langchain_core",  # LangChain uses langchain-core as the base package
    "llamaindex": "llama_index.core",  # LlamaIndex uses llama-index-core package
    # Multi-agent frameworks
    "crewai": "crewai",
    "autogen": "autogen",
    "langgraph": "langgraph",
    "swarm": "swarm",
}


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.

    Args:
        package_name: Name of the package to check (e.g., "anthropic", "openai")

    Returns:
        bool: True if the package can be imported, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def auto_instrument(tracer: Tracer) -> list[str]:
    """
    Automatically instrument all detected libraries.

    This function:
    1. Checks which supported LLM SDKs are installed
    2. Imports and initializes their instrumentors
    3. Calls instrument(tracer) on each
    4. Returns list of successfully instrumented libraries

    The function is designed to be safe:
    - Missing libraries are skipped (not an error)
    - Instrumentation failures are logged but don't crash
    - Returns empty list if nothing was instrumented

    Args:
        tracer: The tracer instance to use for instrumentation

    Returns:
        List of library names that were successfully instrumented
        (e.g., ["anthropic", "openai"])

    Example:
        ```python
        from prela.core.tracer import Tracer
        from prela.instrumentation.auto import auto_instrument

        tracer = Tracer(service_name="my-app")
        instrumented = auto_instrument(tracer)
        print(f"Auto-instrumented: {instrumented}")
        # Output: Auto-instrumented: ['anthropic', 'openai']

        # Now all calls to these SDKs are automatically traced
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(...)  # Automatically traced!
        ```
    """
    instrumented = []

    for lib_name, (module_path, class_name) in INSTRUMENTORS.items():
        # Check if the library is installed
        package_name = PACKAGE_DETECTION.get(lib_name, lib_name)
        if not is_package_installed(package_name):
            logger.debug(
                f"Package '{package_name}' not installed, skipping instrumentation"
            )
            continue

        try:
            # Import instrumentor class
            module = importlib.import_module(module_path)
            instrumentor_class = getattr(module, class_name)

            # Create and instrument
            instrumentor = instrumentor_class()
            instrumentor.instrument(tracer)

            instrumented.append(lib_name)
            logger.debug(f"Successfully instrumented '{lib_name}'")

        except Exception as e:
            # Log warning but don't fail - one broken instrumentor
            # shouldn't prevent others from working
            logger.warning(f"Failed to instrument '{lib_name}': {e}")
            continue

    return instrumented
