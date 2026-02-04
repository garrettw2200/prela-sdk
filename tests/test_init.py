"""Tests for prela.init() public API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import prela
from prela import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    BaseExporter,
    ConsoleExporter,
    ExportResult,
    FileExporter,
    ProbabilitySampler,
    Tracer,
    auto_instrument,
    get_tracer,
)


def test_init_defaults():
    """Test init() with default parameters."""
    tracer = prela.init()

    assert isinstance(tracer, Tracer)
    assert tracer.service_name == "default"
    assert isinstance(tracer.exporter, ConsoleExporter)
    assert isinstance(tracer.sampler, AlwaysOnSampler)
    assert get_tracer() is tracer


def test_init_custom_service_name():
    """Test init() with custom service name."""
    tracer = prela.init(service_name="my-app")
    assert tracer.service_name == "my-app"


def test_init_console_exporter():
    """Test init() with console exporter."""
    tracer = prela.init(exporter="console")
    assert isinstance(tracer.exporter, ConsoleExporter)


def test_init_file_exporter():
    """Test init() with file exporter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracer = prela.init(exporter="file", directory=temp_dir)
        assert isinstance(tracer.exporter, FileExporter)
        assert tracer.exporter.directory == Path(temp_dir)


def test_init_custom_exporter():
    """Test init() with custom exporter instance."""

    class CustomExporter(BaseExporter):
        def export(self, spans):
            return ExportResult.SUCCESS

        def shutdown(self):
            pass

    custom = CustomExporter()
    tracer = prela.init(exporter=custom)
    assert tracer.exporter is custom


def test_init_sample_rate_1_0():
    """Test init() with sample rate 1.0 (always on)."""
    tracer = prela.init(sample_rate=1.0)
    assert isinstance(tracer.sampler, AlwaysOnSampler)


def test_init_sample_rate_0_0():
    """Test init() with sample rate 0.0 (always off)."""
    tracer = prela.init(sample_rate=0.0)
    assert isinstance(tracer.sampler, AlwaysOffSampler)


def test_init_sample_rate_0_5():
    """Test init() with sample rate 0.5 (probabilistic)."""
    tracer = prela.init(sample_rate=0.5)
    assert isinstance(tracer.sampler, ProbabilitySampler)


def test_init_auto_instrument_disabled():
    """Test init() with auto_instrument=False."""
    with patch("prela._auto_instrument") as mock_auto:
        prela.init(auto_instrument=False)
        mock_auto.assert_not_called()


def test_init_auto_instrument_enabled():
    """Test init() with auto_instrument=True."""
    with patch("prela._auto_instrument") as mock_auto:
        mock_auto.return_value = ["anthropic"]
        prela.init(auto_instrument=True)
        mock_auto.assert_called_once()


def test_init_env_service_name(monkeypatch):
    """Test init() reads PRELA_SERVICE_NAME from environment."""
    monkeypatch.setenv("PRELA_SERVICE_NAME", "env-service")
    tracer = prela.init()
    assert tracer.service_name == "env-service"


def test_init_env_exporter(monkeypatch):
    """Test init() reads PRELA_EXPORTER from environment."""
    monkeypatch.setenv("PRELA_EXPORTER", "console")
    tracer = prela.init()
    assert isinstance(tracer.exporter, ConsoleExporter)


def test_init_env_sample_rate(monkeypatch):
    """Test init() reads PRELA_SAMPLE_RATE from environment."""
    monkeypatch.setenv("PRELA_SAMPLE_RATE", "0.5")
    tracer = prela.init()
    assert isinstance(tracer.sampler, ProbabilitySampler)


def test_init_env_auto_instrument_false(monkeypatch):
    """Test init() reads PRELA_AUTO_INSTRUMENT=false from environment."""
    monkeypatch.setenv("PRELA_AUTO_INSTRUMENT", "false")

    with patch("prela.instrumentation.auto.auto_instrument") as mock_auto:
        prela.init()
        mock_auto.assert_not_called()


def test_init_env_auto_instrument_0(monkeypatch):
    """Test init() reads PRELA_AUTO_INSTRUMENT=0 from environment."""
    monkeypatch.setenv("PRELA_AUTO_INSTRUMENT", "0")

    with patch("prela.instrumentation.auto.auto_instrument") as mock_auto:
        prela.init()
        mock_auto.assert_not_called()


def test_init_env_file_path(monkeypatch):
    """Test init() reads PRELA_TRACE_DIR from environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("PRELA_TRACE_DIR", temp_dir)
        tracer = prela.init(exporter="file")
        assert isinstance(tracer.exporter, FileExporter)
        assert tracer.exporter.directory == Path(temp_dir)


def test_init_env_debug_true(monkeypatch, caplog):
    """Test init() enables debug logging with PRELA_DEBUG=true."""
    import logging

    monkeypatch.setenv("PRELA_DEBUG", "true")
    caplog.set_level(logging.DEBUG)

    prela.init()

    # Debug logging should be enabled
    logger = logging.getLogger("prela")
    assert logger.level == logging.DEBUG


def test_init_parameter_overrides_env(monkeypatch):
    """Test that init() parameters override environment variables."""
    monkeypatch.setenv("PRELA_SERVICE_NAME", "env-service")
    tracer = prela.init(service_name="param-service")
    assert tracer.service_name == "param-service"


def test_init_unknown_exporter_raises():
    """Test init() raises ValueError for unknown exporter."""
    with pytest.raises(ValueError, match="Unknown exporter"):
        prela.init(exporter="unknown-exporter")


def test_init_file_exporter_kwargs():
    """Test init() passes kwargs to file exporter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracer = prela.init(
            exporter="file",
            directory=temp_dir,
            max_file_size_mb=10,
            rotate=False,
            format="ndjson"
        )
        assert isinstance(tracer.exporter, FileExporter)
        assert tracer.exporter.max_file_size_bytes == 10 * 1024 * 1024  # 10MB in bytes
        assert tracer.exporter.rotate is False
        assert tracer.exporter.format == "ndjson"


def test_init_console_exporter_kwargs():
    """Test init() passes kwargs to console exporter."""
    tracer = prela.init(exporter="console", verbosity="minimal")
    assert isinstance(tracer.exporter, ConsoleExporter)
    assert tracer.exporter.verbosity == "minimal"


def test_init_sets_global_tracer():
    """Test init() sets tracer as global."""
    tracer = prela.init(service_name="global-test")
    assert get_tracer() is tracer


def test_init_multiple_calls():
    """Test calling init() multiple times (replaces global tracer)."""
    tracer1 = prela.init(service_name="first")
    tracer2 = prela.init(service_name="second")

    assert get_tracer() is tracer2
    assert get_tracer().service_name == "second"


def test_auto_instrument_function_with_tracer():
    """Test auto_instrument() function with explicit tracer."""
    tracer = Tracer(service_name="test")

    with patch("prela._auto_instrument") as mock_auto:
        mock_auto.return_value = ["anthropic"]
        result = auto_instrument(tracer)

    mock_auto.assert_called_once_with(tracer)
    assert result == ["anthropic"]


def test_auto_instrument_function_without_tracer():
    """Test auto_instrument() function uses global tracer."""
    prela.init(service_name="test", auto_instrument=False)

    with patch("prela._auto_instrument") as mock_auto:
        mock_auto.return_value = ["openai"]
        result = auto_instrument()

    mock_auto.assert_called_once()
    assert result == ["openai"]


def test_auto_instrument_function_no_global_tracer_raises():
    """Test auto_instrument() raises if no global tracer is set."""
    from prela.core.tracer import set_global_tracer

    set_global_tracer(None)

    with pytest.raises(RuntimeError, match="No global tracer set"):
        auto_instrument()


def test_init_integration_with_span_creation():
    """Test end-to-end integration: init() -> create spans -> export."""

    class MockExporter(BaseExporter):
        def __init__(self):
            self.exported_spans = []

        def export(self, spans):
            self.exported_spans.extend(spans)
            return ExportResult.SUCCESS

        def shutdown(self):
            pass

    mock_exporter = MockExporter()
    tracer = prela.init(service_name="integration-test", exporter=mock_exporter)

    # Create a span
    with tracer.span("test-operation") as span:
        span.set_attribute("key", "value")

    # Span should be exported
    assert len(mock_exporter.exported_spans) == 1
    exported = mock_exporter.exported_spans[0]
    assert exported.name == "test-operation"
    assert exported.attributes.get("service.name") == "integration-test"
    assert exported.attributes.get("key") == "value"


def test_init_file_path_default(monkeypatch):
    """Test init() uses default directory for file exporter."""
    # Clear environment variable
    monkeypatch.delenv("PRELA_TRACE_DIR", raising=False)

    tracer = prela.init(exporter="file")
    assert isinstance(tracer.exporter, FileExporter)
    assert tracer.exporter.directory == Path("./traces")

    # Cleanup
    if Path("./traces").exists():
        import shutil
        shutil.rmtree("./traces")
