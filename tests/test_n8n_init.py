"""
Tests for n8n webhook integration in prela.init()
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

import prela


class TestN8nWebhookInit:
    """Test n8n webhook initialization via prela.init()"""

    def test_init_without_n8n_webhook(self):
        """Should not start webhook handler when n8n_webhook_port is None"""
        tracer = prela.init(
            service_name="test",
            exporter="console",
            auto_instrument=False,
            n8n_webhook_port=None,
        )

        assert not hasattr(tracer, "_n8n_webhook_handler")
        assert not hasattr(tracer, "_n8n_webhook_thread")

    @patch("prela.instrumentation.n8n.webhook.N8nWebhookHandler.start_background")
    def test_init_with_n8n_webhook_port(self, mock_start_background):
        """Should start webhook handler when n8n_webhook_port is provided"""
        # Make mock block so thread stays alive
        import threading

        event = threading.Event()
        mock_start_background.side_effect = lambda: event.wait(timeout=1)

        tracer = prela.init(
            service_name="test-n8n",
            exporter="console",
            auto_instrument=False,
            n8n_webhook_port=8787,
            n8n_webhook_host="127.0.0.1",
        )

        # Check handler was created
        assert hasattr(tracer, "_n8n_webhook_handler")
        handler = tracer._n8n_webhook_handler

        # Check handler configuration
        assert handler.port == 8787
        assert handler.host == "127.0.0.1"
        assert handler.tracer == tracer

        # Check thread was created
        assert hasattr(tracer, "_n8n_webhook_thread")
        thread = tracer._n8n_webhook_thread

        # Check thread properties
        assert thread.daemon  # Should be daemon thread
        assert thread.is_alive()  # Thread should be running

        # Check start_background was called
        mock_start_background.assert_called_once()

        # Clean up: signal thread to exit
        event.set()

    @patch("prela.instrumentation.n8n.webhook.N8nWebhookHandler.start_background")
    def test_init_with_env_var(self, mock_start_background):
        """Should read PRELA_N8N_WEBHOOK_PORT environment variable"""
        # Set environment variable
        os.environ["PRELA_N8N_WEBHOOK_PORT"] = "9999"

        try:
            tracer = prela.init(
                service_name="test-n8n-env",
                exporter="console",
                auto_instrument=False,
            )

            # Check handler was created with port from env var
            assert hasattr(tracer, "_n8n_webhook_handler")
            assert tracer._n8n_webhook_handler.port == 9999

            # Check start_background was called
            mock_start_background.assert_called_once()

        finally:
            # Clean up environment variable
            del os.environ["PRELA_N8N_WEBHOOK_PORT"]

    @patch("prela.instrumentation.n8n.webhook.N8nWebhookHandler.start_background")
    def test_parameter_overrides_env_var(self, mock_start_background):
        """Parameter should override environment variable"""
        # Set environment variable
        os.environ["PRELA_N8N_WEBHOOK_PORT"] = "9999"

        try:
            tracer = prela.init(
                service_name="test-override",
                exporter="console",
                auto_instrument=False,
                n8n_webhook_port=7777,  # Should override env var
            )

            # Check handler uses parameter value, not env var
            assert tracer._n8n_webhook_handler.port == 7777

        finally:
            del os.environ["PRELA_N8N_WEBHOOK_PORT"]

    @patch("prela.instrumentation.n8n.webhook.N8nWebhookHandler.start_background")
    def test_default_host(self, mock_start_background):
        """Should use default host 0.0.0.0 when not specified"""
        tracer = prela.init(
            service_name="test-default-host",
            exporter="console",
            auto_instrument=False,
            n8n_webhook_port=8787,
        )

        # Check default host
        assert tracer._n8n_webhook_handler.host == "0.0.0.0"

    @patch("prela.instrumentation.n8n.webhook.N8nWebhookHandler.start_background")
    def test_custom_host(self, mock_start_background):
        """Should use custom host when specified"""
        tracer = prela.init(
            service_name="test-custom-host",
            exporter="console",
            auto_instrument=False,
            n8n_webhook_port=8787,
            n8n_webhook_host="localhost",
        )

        # Check custom host
        assert tracer._n8n_webhook_handler.host == "localhost"

    @patch(
        "prela.instrumentation.n8n.webhook.N8nWebhookHandler.start_background",
        side_effect=Exception("Failed to start"),
    )
    def test_handler_failure_doesnt_crash_init(self, mock_start_background):
        """Should log warning but not crash if webhook handler fails to start"""
        # Should not raise exception
        tracer = prela.init(
            service_name="test-failure",
            exporter="console",
            auto_instrument=False,
            n8n_webhook_port=8787,
        )

        # Init should complete successfully despite handler failure
        assert tracer.service_name == "test-failure"

    def test_handler_not_started_with_none_env_var(self):
        """Should not start handler if env var is not set"""
        # Ensure env var is not set
        if "PRELA_N8N_WEBHOOK_PORT" in os.environ:
            del os.environ["PRELA_N8N_WEBHOOK_PORT"]

        tracer = prela.init(
            service_name="test-no-env",
            exporter="console",
            auto_instrument=False,
        )

        # No handler should be created
        assert not hasattr(tracer, "_n8n_webhook_handler")
