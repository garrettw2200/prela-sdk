"""Tests for replay engine API call integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prela.core.span import Span, SpanStatus, SpanType
from prela.replay.engine import ReplayEngine
from prela.replay.loader import Trace


class TestVendorDetection:
    """Tests for vendor detection from model names."""

    def test_detect_openai_gpt4(self):
        """Test detection of GPT-4 models."""
        engine = self._create_engine()
        assert engine._detect_vendor("gpt-4") == "openai"
        assert engine._detect_vendor("gpt-4-turbo") == "openai"
        assert engine._detect_vendor("gpt-4o") == "openai"

    def test_detect_openai_gpt35(self):
        """Test detection of GPT-3.5 models."""
        engine = self._create_engine()
        assert engine._detect_vendor("gpt-3.5-turbo") == "openai"

    def test_detect_openai_o1(self):
        """Test detection of O1 models."""
        engine = self._create_engine()
        assert engine._detect_vendor("o1-preview") == "openai"
        assert engine._detect_vendor("o1-mini") == "openai"

    def test_detect_anthropic_claude(self):
        """Test detection of Claude models."""
        engine = self._create_engine()
        assert engine._detect_vendor("claude-3-opus-20240229") == "anthropic"
        assert engine._detect_vendor("claude-3-sonnet-20240229") == "anthropic"
        assert engine._detect_vendor("claude-3-haiku-20240307") == "anthropic"
        assert engine._detect_vendor("claude-2.1") == "anthropic"

    def test_detect_unsupported_vendor(self):
        """Test error on unsupported vendor."""
        engine = self._create_engine()
        with pytest.raises(ValueError, match="Cannot detect vendor"):
            engine._detect_vendor("llama-2-70b")

    def _create_engine(self) -> ReplayEngine:
        """Create minimal replay engine for testing."""
        # Create minimal span with replay data
        span = Span(
            trace_id="test-trace",
            span_id="test-span",
            name="test",
            span_type=SpanType.LLM,
            status=SpanStatus.SUCCESS,
        )

        # Mock replay_snapshot
        from prela.core.replay import ReplayCapture
        capture = ReplayCapture()
        capture.set_llm_request(model="gpt-4", prompt="test")
        capture.set_llm_response(text="response", prompt_tokens=10, completion_tokens=20)
        object.__setattr__(span, "replay_snapshot", capture.build())

        trace = Trace(trace_id="test-trace", spans=[span])
        return ReplayEngine(trace)


class TestOpenAIAPICall:
    """Tests for OpenAI API integration."""

    def test_call_openai_basic(self):
        """Test basic OpenAI API call."""
        # Mock the openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response

        # Create engine
        engine = self._create_engine()

        # Mock import
        with patch.dict("sys.modules", {"openai": mock_openai}):
            request = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            text, tokens, cost = engine._call_openai_api(request)

            # Verify
            assert text == "Hello!"
            assert tokens == 30
            assert cost > 0
            mock_client.chat.completions.create.assert_called_once()

    def test_call_openai_with_temperature(self):
        """Test OpenAI call with temperature."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response

        engine = self._create_engine()

        with patch.dict("sys.modules", {"openai": mock_openai}):
            request = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.7,
                "max_tokens": 100,
            }
            text, tokens, cost = engine._call_openai_api(request)

            assert text == "Response"
            assert tokens == 30

            # Verify parameters passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 100

    def test_call_openai_missing_sdk(self):
        """Test error when OpenAI SDK not installed."""
        engine = self._create_engine()

        request = {"model": "gpt-4", "messages": []}

        # Remove openai from sys.modules to simulate missing package
        import sys
        original_modules = sys.modules.copy()

        # Block the import
        if "openai" in sys.modules:
            del sys.modules["openai"]

        try:
            with pytest.raises(ImportError, match="openai package is required"):
                engine._call_openai_api(request)
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def _create_engine(self) -> ReplayEngine:
        """Create minimal replay engine."""
        span = Span(
            trace_id="test-trace",
            span_id="test-span",
            name="test",
            span_type=SpanType.LLM,
            status=SpanStatus.SUCCESS,
        )

        from prela.core.replay import ReplayCapture
        capture = ReplayCapture()
        capture.set_llm_request(model="gpt-4", prompt="test")
        capture.set_llm_response(text="response", prompt_tokens=10, completion_tokens=20)
        object.__setattr__(span, "replay_snapshot", capture.build())

        trace = Trace(trace_id="test-trace", spans=[span])
        return ReplayEngine(trace)


class TestAnthropicAPICall:
    """Tests for Anthropic API integration."""

    def test_call_anthropic_basic(self):
        """Test basic Anthropic API call."""
        # Mock the anthropic module
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_content_block = MagicMock()
        mock_content_block.text = "Hello from Claude!"

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_response

        # Create engine
        engine = self._create_engine()

        # Mock import
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            request = {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            text, tokens, cost = engine._call_anthropic_api(request)

            # Verify
            assert text == "Hello from Claude!"
            assert tokens == 30
            assert cost > 0
            mock_client.messages.create.assert_called_once()

    def test_call_anthropic_with_system_message(self):
        """Test Anthropic call with system message."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_content_block = MagicMock()
        mock_content_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_response

        engine = self._create_engine()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            request = {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"}
                ],
                "temperature": 0.5,
                "max_tokens": 200,
            }
            text, tokens, cost = engine._call_anthropic_api(request)

            assert text == "Response"
            assert tokens == 30

            # Verify system message separated
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are helpful"
            assert len(call_kwargs["messages"]) == 1  # Only user message
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 200

    def test_call_anthropic_multiple_content_blocks(self):
        """Test Anthropic response with multiple content blocks."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_block1 = MagicMock()
        mock_block1.text = "Part 1 "
        mock_block2 = MagicMock()
        mock_block2.text = "Part 2"

        mock_response = MagicMock()
        mock_response.content = [mock_block1, mock_block2]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_response

        engine = self._create_engine()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            request = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            text, tokens, cost = engine._call_anthropic_api(request)

            assert text == "Part 1 Part 2"

    def test_call_anthropic_missing_sdk(self):
        """Test error when Anthropic SDK not installed."""
        engine = self._create_engine()

        request = {"model": "claude-3-sonnet-20240229", "messages": []}

        # Remove anthropic from sys.modules to simulate missing package
        import sys
        original_modules = sys.modules.copy()

        if "anthropic" in sys.modules:
            del sys.modules["anthropic"]

        try:
            with pytest.raises(ImportError, match="anthropic package is required"):
                engine._call_anthropic_api(request)
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def _create_engine(self) -> ReplayEngine:
        """Create minimal replay engine."""
        span = Span(
            trace_id="test-trace",
            span_id="test-span",
            name="test",
            span_type=SpanType.LLM,
            status=SpanStatus.SUCCESS,
        )

        from prela.core.replay import ReplayCapture
        capture = ReplayCapture()
        capture.set_llm_request(model="claude-3-sonnet-20240229", prompt="test")
        capture.set_llm_response(text="response", prompt_tokens=10, completion_tokens=20)
        object.__setattr__(span, "replay_snapshot", capture.build())

        trace = Trace(trace_id="test-trace", spans=[span])
        return ReplayEngine(trace)


class TestCallLLMAPIRouting:
    """Tests for _call_llm_api routing logic."""

    def test_routes_to_openai(self):
        """Test routing to OpenAI API."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response

        engine = self._create_engine()

        with patch.dict("sys.modules", {"openai": mock_openai}):
            request = {
                "model": "gpt-4-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            text, tokens, cost = engine._call_llm_api(request)

            assert text == "Hi"
            mock_client.chat.completions.create.assert_called_once()

    def test_routes_to_anthropic(self):
        """Test routing to Anthropic API."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_content_block = MagicMock()
        mock_content_block.text = "Hello"

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_response

        engine = self._create_engine()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            request = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            text, tokens, cost = engine._call_llm_api(request)

            assert text == "Hello"
            mock_client.messages.create.assert_called_once()

    def test_error_missing_model(self):
        """Test error when model is missing."""
        engine = self._create_engine()

        request = {"messages": [{"role": "user", "content": "Hi"}]}

        with pytest.raises(ValueError, match="Model is required"):
            engine._call_llm_api(request)

    def test_error_unsupported_vendor(self):
        """Test error for unsupported vendor."""
        engine = self._create_engine()

        request = {
            "model": "llama-2-70b",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with pytest.raises(ValueError, match="Cannot detect vendor"):
            engine._call_llm_api(request)

    def _create_engine(self) -> ReplayEngine:
        """Create minimal replay engine."""
        span = Span(
            trace_id="test-trace",
            span_id="test-span",
            name="test",
            span_type=SpanType.LLM,
            status=SpanStatus.SUCCESS,
        )

        from prela.core.replay import ReplayCapture
        capture = ReplayCapture()
        capture.set_llm_request(model="gpt-4", prompt="test")
        capture.set_llm_response(text="response", prompt_tokens=10, completion_tokens=20)
        object.__setattr__(span, "replay_snapshot", capture.build())

        trace = Trace(trace_id="test-trace", spans=[span])
        return ReplayEngine(trace)
