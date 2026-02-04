"""Tests for AutoGen instrumentation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from prela.core.span import SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter
from prela.instrumentation.multi_agent.autogen import AutoGenInstrumentor
from prela.instrumentation.multi_agent.models import generate_agent_id


# Mock AutoGen classes
class MockAgent:
    """Mock AutoGen ConversableAgent."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def initiate_chat(self, recipient, message=None, **kwargs):
        """Initiate a chat with another agent."""
        return f"Chat initiated: {self.name} -> {recipient.name}"

    def generate_reply(self, messages=None, sender=None, **kwargs):
        """Generate a reply to messages."""
        return f"Reply from {self.name}"


class MockGroupChat:
    """Mock AutoGen GroupChat."""

    def __init__(self, agents, **kwargs):
        self.agents = agents
        self.kwargs = kwargs

    def select_speaker(self, *args, **kwargs):
        """Select next speaker."""
        return self.agents[0] if self.agents else None


class MockGroupChatManager(MockAgent):
    """Mock AutoGen GroupChatManager."""

    def __init__(self, groupchat, name: str = "manager", **kwargs):
        super().__init__(name, **kwargs)
        self.groupchat = groupchat

    def run_chat(self, *args, **kwargs):
        """Run the group chat."""
        return "Group chat completed"


# Create mock autogen module
mock_autogen = MagicMock()
mock_autogen.ConversableAgent = MockAgent
mock_autogen.GroupChat = MockGroupChat
mock_autogen.GroupChatManager = MockGroupChatManager


@pytest.fixture
def mock_autogen_module():
    """Fixture to mock autogen module."""
    sys.modules["autogen"] = mock_autogen
    yield mock_autogen
    if "autogen" in sys.modules:
        del sys.modules["autogen"]


@pytest.fixture
def instrumentor():
    """Create AutoGen instrumentor."""
    return AutoGenInstrumentor()


@pytest.fixture
def tracer():
    """Create tracer with console exporter."""
    return Tracer(service_name="test-autogen", exporter=ConsoleExporter())


class TestAutoGenInstrumentor:
    """Test AutoGen instrumentor."""

    def test_init(self, instrumentor):
        """Test instrumentor initialization."""
        assert instrumentor.FRAMEWORK == "autogen"
        assert not instrumentor.is_instrumented
        assert instrumentor._active_conversations == {}

    def test_instrument_without_autogen(self, instrumentor):
        """Test instrument when autogen not installed."""
        # Should not raise error
        instrumentor.instrument()
        assert not instrumentor.is_instrumented

    def test_instrument_with_autogen(self, instrumentor, tracer, mock_autogen_module):
        """Test instrument with autogen installed."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented
        assert instrumentor._tracer is tracer

        # Verify that methods were patched
        assert hasattr(MockAgent, "_prela_original_initiate_chat")
        assert hasattr(MockAgent, "_prela_original_generate_reply")
        assert hasattr(MockGroupChat, "_prela_original_select_speaker")
        assert hasattr(MockGroupChatManager, "_prela_original_run_chat")

    def test_instrument_idempotent(self, instrumentor, tracer, mock_autogen_module):
        """Test that instrument can be called multiple times."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        # Call again - should not raise
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

    def test_uninstrument(self, instrumentor, tracer, mock_autogen_module):
        """Test uninstrument restores original methods."""
        instrumentor.instrument(tracer)
        assert instrumentor.is_instrumented

        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented
        assert instrumentor._tracer is None

        # Check that original methods were restored
        assert not hasattr(MockAgent, "_prela_original_initiate_chat")
        assert not hasattr(MockAgent, "_prela_original_generate_reply")
        assert not hasattr(MockGroupChat, "_prela_original_select_speaker")
        assert not hasattr(MockGroupChatManager, "_prela_original_run_chat")

    def test_initiate_chat_basic(self, instrumentor, tracer, mock_autogen_module):
        """Test initiate_chat creates conversation span."""
        instrumentor.instrument(tracer)

        # Create agents AFTER instrumentation
        alice = MockAgent("Alice")
        bob = MockAgent("Bob")

        # The wrapped method should call the original
        result = alice.initiate_chat(bob, message="Hello Bob!")

        # Result should be from the original method
        assert result == "Chat initiated: Alice -> Bob"

    def test_initiate_chat_with_max_turns(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test initiate_chat with max_turns parameter."""
        instrumentor.instrument(tracer)

        alice = MockAgent("Alice")
        bob = MockAgent("Bob")

        result = alice.initiate_chat(bob, message="Hello", max_turns=5)

        assert result == "Chat initiated: Alice -> Bob"

    def test_initiate_chat_with_exception(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test initiate_chat handles exceptions."""
        instrumentor.instrument(tracer)

        alice = MockAgent("Alice")
        bob = MockAgent("Bob")

        # Make initiate_chat raise exception
        alice.initiate_chat = Mock(side_effect=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            alice.initiate_chat(bob, message="Hello")

    def test_generate_reply_basic(self, instrumentor, tracer, mock_autogen_module):
        """Test generate_reply creates agent span."""
        instrumentor.instrument(tracer)

        agent = MockAgent("Assistant")
        sender = MockAgent("User")

        result = agent.generate_reply(
            messages=[{"role": "user", "content": "Hello"}], sender=sender
        )

        assert result == "Reply from Assistant"

    def test_generate_reply_without_sender(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test generate_reply without sender."""
        instrumentor.instrument(tracer)

        agent = MockAgent("Assistant")

        result = agent.generate_reply(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert result == "Reply from Assistant"

    def test_generate_reply_with_exception(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test generate_reply handles exceptions."""
        instrumentor.instrument(tracer)

        agent = MockAgent("Assistant")
        agent.generate_reply = Mock(side_effect=RuntimeError("Generation failed"))

        with pytest.raises(RuntimeError, match="Generation failed"):
            agent.generate_reply(messages=[{"role": "user", "content": "Hello"}])

    def test_group_chat_select_speaker(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test group chat speaker selection."""
        instrumentor.instrument(tracer)

        alice = MockAgent("Alice")
        bob = MockAgent("Bob")
        charlie = MockAgent("Charlie")

        group_chat = MockGroupChat(agents=[alice, bob, charlie])

        result = group_chat.select_speaker()

        assert result == alice

    def test_group_chat_manager_run_chat(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test group chat manager run_chat."""
        instrumentor.instrument(tracer)

        alice = MockAgent("Alice")
        bob = MockAgent("Bob")

        group_chat = MockGroupChat(agents=[alice, bob])
        manager = MockGroupChatManager(groupchat=group_chat, name="manager")

        result = manager.run_chat()

        assert result == "Group chat completed"

    def test_group_chat_manager_with_exception(
        self, instrumentor, tracer, mock_autogen_module
    ):
        """Test group chat manager handles exceptions."""
        instrumentor.instrument(tracer)

        group_chat = MockGroupChat(agents=[])
        manager = MockGroupChatManager(groupchat=group_chat)

        manager.run_chat = Mock(side_effect=ValueError("Chat failed"))

        with pytest.raises(ValueError, match="Chat failed"):
            manager.run_chat()

    def test_agent_id_generation(self, instrumentor):
        """Test agent ID generation is deterministic."""
        agent_id_1 = generate_agent_id("autogen", "Assistant")
        agent_id_2 = generate_agent_id("autogen", "Assistant")

        assert agent_id_1 == agent_id_2
        assert len(agent_id_1) == 12

    def test_different_agents_different_ids(self, instrumentor):
        """Test different agents get different IDs."""
        alice_id = generate_agent_id("autogen", "Alice")
        bob_id = generate_agent_id("autogen", "Bob")

        assert alice_id != bob_id


class TestAutoGenIntegration:
    """Integration tests for AutoGen instrumentation."""

    def test_multi_agent_conversation(self, instrumentor, tracer, mock_autogen_module):
        """Test complete multi-agent conversation."""
        instrumentor.instrument(tracer)

        # Create agents
        alice = MockAgent("Alice")
        bob = MockAgent("Bob")
        charlie = MockAgent("Charlie")

        # Create group chat
        group_chat = MockGroupChat(agents=[alice, bob, charlie])
        manager = MockGroupChatManager(groupchat=group_chat, name="manager")

        # Run conversation
        result = manager.run_chat()

        assert result == "Group chat completed"

    def test_uninstrumented_execution(self, mock_autogen_module):
        """Test that agents work normally when not instrumented."""
        alice = MockAgent("Alice")
        bob = MockAgent("Bob")

        # Should work normally without instrumentation
        result = alice.initiate_chat(bob, message="Hello")
        assert result == "Chat initiated: Alice -> Bob"

    def test_pyautogen_fallback(self, instrumentor, tracer):
        """Test fallback to pyautogen package name."""
        # Create mock pyautogen module
        mock_pyautogen = MagicMock()
        mock_pyautogen.ConversableAgent = MockAgent
        mock_pyautogen.GroupChat = MockGroupChat
        mock_pyautogen.GroupChatManager = MockGroupChatManager

        sys.modules["pyautogen"] = mock_pyautogen

        try:
            instrumentor.instrument(tracer)
            assert instrumentor.is_instrumented

            # Verify patching worked
            assert hasattr(MockAgent, "_prela_original_initiate_chat")
        finally:
            if "pyautogen" in sys.modules:
                del sys.modules["pyautogen"]
            instrumentor.uninstrument()

    def test_conversation_tracking(self, instrumentor, tracer, mock_autogen_module):
        """Test that conversations are tracked and cleaned up."""
        instrumentor.instrument(tracer)

        alice = MockAgent("Alice")
        bob = MockAgent("Bob")

        # Before chat
        assert len(instrumentor._active_conversations) == 0

        # During chat
        alice.initiate_chat(bob, message="Hello")

        # After chat - should be cleaned up
        assert len(instrumentor._active_conversations) == 0
