"""AutoGen instrumentation for Prela.

This module provides automatic instrumentation for Microsoft AutoGen (>=0.2.0),
capturing multi-agent conversations, group chats, and agent interactions.
"""

from __future__ import annotations

import functools
import uuid
from datetime import datetime
from typing import Any, Optional

from prela.core.span import SpanType
from prela.core.tracer import Tracer, get_tracer
from prela.instrumentation.base import Instrumentor
from prela.instrumentation.multi_agent.models import (
    AgentDefinition,
    AgentMessage,
    AgentRole,
    ConversationTurn,
    MessageType,
    generate_agent_id,
)
from prela.license import require_tier


class AutoGenInstrumentor(Instrumentor):
    """Instrumentor for Microsoft AutoGen multi-agent framework."""

    FRAMEWORK = "autogen"

    @property
    def is_instrumented(self) -> bool:
        """Check if AutoGen is currently instrumented."""
        return self._is_instrumented

    def __init__(self):
        super().__init__()
        self._active_conversations: dict[str, list[ConversationTurn]] = {}
        self._is_instrumented = False
        self._tracer: Optional[Tracer] = None

    @require_tier("AutoGen instrumentation", "lunch-money")
    def instrument(self, tracer: Optional[Tracer] = None) -> None:
        """Patch AutoGen classes for tracing.

        Args:
            tracer: Optional tracer to use. If None, uses global tracer.
        """
        if self.is_instrumented:
            return

        # Try both autogen and pyautogen package names
        try:
            import autogen
        except ImportError:
            try:
                import pyautogen as autogen  # type: ignore
            except ImportError:
                return  # AutoGen not installed

        self._tracer = tracer or get_tracer()

        # Patch ConversableAgent (base class for most agents)
        if hasattr(autogen, "ConversableAgent"):
            self._patch_conversable_agent(autogen.ConversableAgent)

        # Patch GroupChat for multi-agent conversations
        if hasattr(autogen, "GroupChat"):
            self._patch_group_chat(autogen.GroupChat)

        # Patch GroupChatManager
        if hasattr(autogen, "GroupChatManager"):
            self._patch_group_chat_manager(autogen.GroupChatManager)

        self._is_instrumented = True

    def uninstrument(self) -> None:
        """Restore original methods."""
        if not self.is_instrumented:
            return

        # Try both package names
        try:
            import autogen
        except ImportError:
            try:
                import pyautogen as autogen  # type: ignore
            except ImportError:
                return

        # Restore all patched methods
        for module_name, obj_name, method_name in [
            ("autogen", "ConversableAgent", "initiate_chat"),
            ("autogen", "ConversableAgent", "generate_reply"),
            ("autogen", "GroupChat", "select_speaker"),
            ("autogen", "GroupChatManager", "run_chat"),
        ]:
            try:
                # Try autogen first, then pyautogen
                try:
                    module = __import__(module_name, fromlist=[obj_name])
                except ImportError:
                    module = __import__("py" + module_name, fromlist=[obj_name])

                cls = getattr(module, obj_name, None)
                if cls and hasattr(cls, f"_prela_original_{method_name}"):
                    original = getattr(cls, f"_prela_original_{method_name}")
                    setattr(cls, method_name, original)
                    delattr(cls, f"_prela_original_{method_name}")
            except (ImportError, AttributeError):
                pass

        self._is_instrumented = False
        self._tracer = None

    def _patch_conversable_agent(self, agent_cls) -> None:
        """Patch ConversableAgent for tracing."""
        # Patch initiate_chat (main conversation entry point)
        if hasattr(agent_cls, "initiate_chat"):
            if hasattr(agent_cls, "_prela_original_initiate_chat"):
                return

            original_initiate_chat = agent_cls.initiate_chat
            agent_cls._prela_original_initiate_chat = original_initiate_chat

            instrumentor = self

            @functools.wraps(original_initiate_chat)
            def wrapped_initiate_chat(agent_self, recipient, *args, **kwargs):
                tracer = instrumentor._tracer
                if not tracer:
                    return original_initiate_chat(agent_self, recipient, *args, **kwargs)

                conversation_id = str(uuid.uuid4())
                instrumentor._active_conversations[conversation_id] = []

                # Extract initial message
                message = kwargs.get("message") or (args[0] if args else None)

                # Create conversation attributes
                conversation_attributes = {
                    "conversation.id": conversation_id,
                    "conversation.framework": instrumentor.FRAMEWORK,
                    "conversation.initiator": agent_self.name,
                    "conversation.recipient": recipient.name,
                    "conversation.initial_message": (
                        str(message)[:500] if message else None
                    ),
                    "conversation.max_turns": kwargs.get("max_turns"),
                }

                # NEW: Replay capture if enabled
                replay_capture = None
                if tracer.capture_for_replay:
                    from prela.core.replay import ReplayCapture

                    replay_capture = ReplayCapture()
                    # Capture agent context
                    replay_capture.set_agent_context(
                        system_prompt=getattr(agent_self, "system_message", None),
                        available_tools=[
                            {"name": name, "description": str(func)}
                            for name, func in getattr(agent_self, "_function_map", {}).items()
                        ],
                        memory={"messages": getattr(agent_self, "chat_messages", {})},
                        config={
                            "framework": instrumentor.FRAMEWORK,
                            "conversation_id": conversation_id,
                            "initiator": agent_self.name,
                            "recipient": recipient.name,
                            "max_turns": kwargs.get("max_turns"),
                        },
                    )

                with tracer.span(
                    name=f"autogen.conversation.{agent_self.name}->{recipient.name}",
                    span_type=SpanType.AGENT,
                    attributes=conversation_attributes,
                ) as span:
                    try:
                        result = original_initiate_chat(agent_self, recipient, *args, **kwargs)

                        # Add conversation statistics
                        turns = instrumentor._active_conversations.get(
                            conversation_id, []
                        )
                        span.set_attribute("conversation.total_turns", len(turns))
                        span.set_attribute(
                            "conversation.total_tokens",
                            sum(t.tokens_used for t in turns),
                        )

                        # NEW: Attach replay snapshot
                        if replay_capture:
                            try:
                                object.__setattr__(span, "replay_snapshot", replay_capture.build())
                            except Exception as e:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.debug(f"Failed to capture replay data: {e}")

                        return result
                    except Exception as e:
                        span.add_event(
                            "exception",
                            attributes={
                                "exception.type": type(e).__name__,
                                "exception.message": str(e),
                            },
                        )
                        raise
                    finally:
                        if conversation_id in instrumentor._active_conversations:
                            del instrumentor._active_conversations[conversation_id]

            agent_cls.initiate_chat = wrapped_initiate_chat

        # Patch generate_reply (individual agent responses)
        if hasattr(agent_cls, "generate_reply"):
            if hasattr(agent_cls, "_prela_original_generate_reply"):
                return

            original_generate_reply = agent_cls.generate_reply
            agent_cls._prela_original_generate_reply = original_generate_reply

            instrumentor = self

            @functools.wraps(original_generate_reply)
            def wrapped_generate_reply(
                agent_self, messages=None, sender=None, *args, **kwargs
            ):
                tracer = instrumentor._tracer
                if not tracer:
                    return original_generate_reply(agent_self, messages, sender, *args, **kwargs)

                agent_id = generate_agent_id(instrumentor.FRAMEWORK, agent_self.name)
                sender_name = sender.name if sender else "unknown"

                reply_attributes = {
                    "agent.id": agent_id,
                    "agent.name": agent_self.name,
                    "agent.type": type(agent_self).__name__,
                    "agent.framework": instrumentor.FRAMEWORK,
                    "reply.sender": sender_name,
                    "reply.num_messages": len(messages) if messages else 0,
                }

                with tracer.span(
                    name=f"autogen.agent.{agent_self.name}",
                    span_type=SpanType.AGENT,
                    attributes=reply_attributes,
                ) as span:
                    try:
                        result = original_generate_reply(agent_self, messages, sender, *args, **kwargs)
                        if result:
                            span.set_attribute(
                                "reply.content_length", len(str(result))
                            )
                        return result
                    except Exception as e:
                        span.add_event(
                            "exception",
                            attributes={
                                "exception.type": type(e).__name__,
                                "exception.message": str(e),
                            },
                        )
                        raise

            agent_cls.generate_reply = wrapped_generate_reply

    def _patch_group_chat(self, group_chat_cls) -> None:
        """Patch GroupChat for multi-agent conversations."""
        if hasattr(group_chat_cls, "select_speaker"):
            if hasattr(group_chat_cls, "_prela_original_select_speaker"):
                return

            original_select_speaker = group_chat_cls.select_speaker
            group_chat_cls._prela_original_select_speaker = original_select_speaker

            instrumentor = self

            @functools.wraps(original_select_speaker)
            def wrapped_select_speaker(gc_self, *args, **kwargs):
                tracer = instrumentor._tracer
                result = original_select_speaker(gc_self, *args, **kwargs)

                # Get current span from context
                if tracer:
                    try:
                        from prela.core.context import get_current_span

                        current_span = get_current_span()
                        if current_span and result:
                            current_span.add_event(
                                "group.speaker_selected",
                                attributes={
                                    "speaker.name": result.name,
                                    "group.num_agents": len(gc_self.agents),
                                },
                            )
                    except Exception:
                        # Defensive: don't crash if span event fails
                        pass

                return result

            group_chat_cls.select_speaker = wrapped_select_speaker

    def _patch_group_chat_manager(self, manager_cls) -> None:
        """Patch GroupChatManager."""
        if hasattr(manager_cls, "run_chat"):
            if hasattr(manager_cls, "_prela_original_run_chat"):
                return

            original_run_chat = manager_cls.run_chat
            manager_cls._prela_original_run_chat = original_run_chat

            instrumentor = self

            @functools.wraps(original_run_chat)
            def wrapped_run_chat(manager_self, *args, **kwargs):
                tracer = instrumentor._tracer
                if not tracer:
                    return original_run_chat(manager_self, *args, **kwargs)

                group_chat = manager_self.groupchat

                group_attributes = {
                    "group.manager": manager_self.name,
                    "group.framework": instrumentor.FRAMEWORK,
                    "group.num_agents": len(group_chat.agents) if group_chat else 0,
                    "group.agent_names": (
                        [a.name for a in group_chat.agents] if group_chat else []
                    ),
                }

                with tracer.span(
                    name=f"autogen.group_chat.{manager_self.name}",
                    span_type=SpanType.AGENT,
                    attributes=group_attributes,
                ) as span:
                    try:
                        return original_run_chat(manager_self, *args, **kwargs)
                    except Exception as e:
                        span.add_event(
                            "exception",
                            attributes={
                                "exception.type": type(e).__name__,
                                "exception.message": str(e),
                            },
                        )
                        raise

            manager_cls.run_chat = wrapped_run_chat
