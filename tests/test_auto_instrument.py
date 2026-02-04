"""Tests for auto-instrumentation system."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from prela.core.tracer import Tracer
from prela.instrumentation.auto import (
    INSTRUMENTORS,
    PACKAGE_DETECTION,
    auto_instrument,
    is_package_installed,
)


def test_is_package_installed_true():
    """Test detecting installed packages."""
    # Test with a package that's definitely installed (pytest)
    assert is_package_installed("pytest") is True


def test_is_package_installed_false():
    """Test detecting missing packages."""
    # Test with a package that definitely doesn't exist
    assert is_package_installed("this_package_does_not_exist_xyz123") is False


def test_auto_instrument_no_packages_installed():
    """Test auto_instrument when no LLM SDKs are installed."""
    tracer = Tracer()

    with patch("prela.instrumentation.auto.is_package_installed", return_value=False):
        instrumented = auto_instrument(tracer)

    assert instrumented == []


def test_auto_instrument_anthropic_installed():
    """Test auto_instrument with Anthropic SDK installed."""
    tracer = Tracer()

    # Mock package detection
    def mock_is_installed(package):
        return package == "anthropic"

    # Mock instrumentor
    mock_instrumentor_instance = Mock()
    mock_instrumentor_class = Mock(return_value=mock_instrumentor_instance)

    with patch(
        "prela.instrumentation.auto.is_package_installed", side_effect=mock_is_installed
    ):
        with patch(
            "prela.instrumentation.auto.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.AnthropicInstrumentor = mock_instrumentor_class
            mock_import.return_value = mock_module

            instrumented = auto_instrument(tracer)

    assert "anthropic" in instrumented
    mock_instrumentor_instance.instrument.assert_called_once_with(tracer)


def test_auto_instrument_openai_installed():
    """Test auto_instrument with OpenAI SDK installed."""
    tracer = Tracer()

    # Mock package detection
    def mock_is_installed(package):
        return package == "openai"

    # Mock instrumentor
    mock_instrumentor_instance = Mock()
    mock_instrumentor_class = Mock(return_value=mock_instrumentor_instance)

    with patch(
        "prela.instrumentation.auto.is_package_installed", side_effect=mock_is_installed
    ):
        with patch(
            "prela.instrumentation.auto.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.OpenAIInstrumentor = mock_instrumentor_class
            mock_import.return_value = mock_module

            instrumented = auto_instrument(tracer)

    assert "openai" in instrumented
    mock_instrumentor_instance.instrument.assert_called_once_with(tracer)


def test_auto_instrument_langchain_installed():
    """Test auto_instrument with LangChain SDK installed."""
    tracer = Tracer()

    # Mock package detection (langchain uses langchain_core)
    def mock_is_installed(package):
        return package == "langchain_core"

    # Mock instrumentor
    mock_instrumentor_instance = Mock()
    mock_instrumentor_class = Mock(return_value=mock_instrumentor_instance)

    with patch(
        "prela.instrumentation.auto.is_package_installed", side_effect=mock_is_installed
    ):
        with patch(
            "prela.instrumentation.auto.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.LangChainInstrumentor = mock_instrumentor_class
            mock_import.return_value = mock_module

            instrumented = auto_instrument(tracer)

    assert "langchain" in instrumented
    mock_instrumentor_instance.instrument.assert_called_once_with(tracer)


def test_auto_instrument_llamaindex_installed():
    """Test auto_instrument with LlamaIndex SDK installed."""
    tracer = Tracer()

    # Mock package detection (llamaindex uses llama_index.core)
    def mock_is_installed(package):
        return package == "llama_index.core"

    # Mock instrumentor
    mock_instrumentor_instance = Mock()
    mock_instrumentor_class = Mock(return_value=mock_instrumentor_instance)

    with patch(
        "prela.instrumentation.auto.is_package_installed", side_effect=mock_is_installed
    ):
        with patch(
            "prela.instrumentation.auto.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.LlamaIndexInstrumentor = mock_instrumentor_class
            mock_import.return_value = mock_module

            instrumented = auto_instrument(tracer)

    assert "llamaindex" in instrumented
    mock_instrumentor_instance.instrument.assert_called_once_with(tracer)


def test_auto_instrument_multiple_packages():
    """Test auto_instrument with multiple SDKs installed."""
    tracer = Tracer()

    # Mock all four packages installed
    def mock_is_installed(package):
        return package in ("anthropic", "openai", "langchain_core", "llama_index.core")

    # Mock instrumentors
    mock_anthropic_instance = Mock()
    mock_openai_instance = Mock()
    mock_langchain_instance = Mock()
    mock_llamaindex_instance = Mock()

    def mock_import(module_path):
        mock_module = Mock()
        if "anthropic" in module_path:
            mock_module.AnthropicInstrumentor = Mock(
                return_value=mock_anthropic_instance
            )
        elif "openai" in module_path:
            mock_module.OpenAIInstrumentor = Mock(return_value=mock_openai_instance)
        elif "langchain" in module_path:
            mock_module.LangChainInstrumentor = Mock(return_value=mock_langchain_instance)
        elif "llamaindex" in module_path:
            mock_module.LlamaIndexInstrumentor = Mock(return_value=mock_llamaindex_instance)
        return mock_module

    with patch(
        "prela.instrumentation.auto.is_package_installed", side_effect=mock_is_installed
    ):
        with patch(
            "prela.instrumentation.auto.importlib.import_module",
            side_effect=mock_import,
        ):
            instrumented = auto_instrument(tracer)

    assert "anthropic" in instrumented
    assert "openai" in instrumented
    assert "langchain" in instrumented
    assert "llamaindex" in instrumented
    assert len(instrumented) == 4


def test_auto_instrument_handles_instrumentor_import_error():
    """Test that auto_instrument handles import errors gracefully."""
    tracer = Tracer()

    # Mock package installed but import fails
    with patch("prela.instrumentation.auto.is_package_installed", return_value=True):
        with patch(
            "prela.instrumentation.auto.importlib.import_module",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise, just skip the instrumentor
            instrumented = auto_instrument(tracer)

    assert instrumented == []


def test_auto_instrument_handles_instrumentation_error():
    """Test that auto_instrument handles instrumentation errors gracefully."""
    tracer = Tracer()

    # Mock package installed but instrument() raises error
    mock_instrumentor_instance = Mock()
    mock_instrumentor_instance.instrument.side_effect = Exception("Instrumentation failed")
    mock_instrumentor_class = Mock(return_value=mock_instrumentor_instance)

    def mock_import(module_path):
        mock_module = Mock()
        if "anthropic" in module_path:
            mock_module.AnthropicInstrumentor = mock_instrumentor_class
        elif "openai" in module_path:
            mock_module.OpenAIInstrumentor = mock_instrumentor_class
        elif "langchain" in module_path:
            mock_module.LangChainInstrumentor = mock_instrumentor_class
        elif "llamaindex" in module_path:
            mock_module.LlamaIndexInstrumentor = mock_instrumentor_class
        elif "crewai" in module_path:
            mock_module.CrewAIInstrumentor = mock_instrumentor_class
        elif "autogen" in module_path:
            mock_module.AutoGenInstrumentor = mock_instrumentor_class
        elif "langgraph" in module_path:
            mock_module.LangGraphInstrumentor = mock_instrumentor_class
        elif "swarm" in module_path:
            mock_module.SwarmInstrumentor = mock_instrumentor_class
        return mock_module

    with patch("prela.instrumentation.auto.is_package_installed", return_value=True):
        with patch(
            "prela.instrumentation.auto.importlib.import_module",
            side_effect=mock_import
        ):
            # Should not raise, just skip the failed instrumentors
            instrumented = auto_instrument(tracer)

    assert instrumented == []


def test_instrumentors_registry_format():
    """Test that INSTRUMENTORS registry has correct format."""
    assert isinstance(INSTRUMENTORS, dict)

    for lib_name, (module_path, class_name) in INSTRUMENTORS.items():
        assert isinstance(lib_name, str)
        assert isinstance(module_path, str)
        assert isinstance(class_name, str)
        assert "." in module_path  # Should be a module path


def test_package_detection_registry_format():
    """Test that PACKAGE_DETECTION registry has correct format."""
    assert isinstance(PACKAGE_DETECTION, dict)

    for lib_name, package_name in PACKAGE_DETECTION.items():
        assert isinstance(lib_name, str)
        assert isinstance(package_name, str)


def test_instrumentors_match_package_detection():
    """Test that all INSTRUMENTORS have corresponding PACKAGE_DETECTION entries."""
    for lib_name in INSTRUMENTORS.keys():
        assert lib_name in PACKAGE_DETECTION or lib_name == lib_name.lower()


def test_auto_instrument_with_logging_debug(caplog):
    """Test that auto_instrument logs debug messages."""
    import logging

    tracer = Tracer()

    caplog.set_level(logging.DEBUG, logger="prela.instrumentation.auto")

    with patch("prela.instrumentation.auto.is_package_installed", return_value=False):
        auto_instrument(tracer)

    # Should have debug messages about packages not being installed
    assert any("not installed" in record.message for record in caplog.records)


def test_auto_instrument_with_logging_warning(caplog):
    """Test that auto_instrument logs warnings on failures."""
    import logging

    tracer = Tracer()

    caplog.set_level(logging.WARNING, logger="prela.instrumentation.auto")

    # Mock instrumentation failure
    mock_instrumentor_instance = Mock()
    mock_instrumentor_instance.instrument.side_effect = Exception("Test failure")
    mock_instrumentor_class = Mock(return_value=mock_instrumentor_instance)

    with patch("prela.instrumentation.auto.is_package_installed", return_value=True):
        with patch(
            "prela.instrumentation.auto.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.AnthropicInstrumentor = mock_instrumentor_class
            mock_import.return_value = mock_module

            auto_instrument(tracer)

    # Should have warning about failed instrumentation
    assert any("Failed to instrument" in record.message for record in caplog.records)
