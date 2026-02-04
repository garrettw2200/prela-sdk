"""Example: AutoGen Instrumentation

This example demonstrates how to use Prela to trace Microsoft AutoGen multi-agent conversations.

Requirements:
    pip install pyautogen prela

Note: This example requires actual AutoGen to be installed. The code shown
demonstrates the API but won't run without a real AutoGen installation.
"""

import prela
from prela.instrumentation.multi_agent import AutoGenInstrumentor


def example_manual_instrumentation():
    """Example 1: Manual instrumentation of AutoGen."""
    print("=" * 60)
    print("Example 1: Manual Instrumentation")
    print("=" * 60)

    # Initialize Prela tracer
    tracer = prela.init(service_name="autogen-demo", exporter="console")

    # Manually instrument AutoGen
    instrumentor = AutoGenInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ AutoGen instrumented successfully")
    print("  All conversations will now be traced automatically")

    # Now use AutoGen normally - all operations are traced!
    # from autogen import ConversableAgent
    # assistant = ConversableAgent(name="assistant", llm_config=llm_config)
    # user_proxy = ConversableAgent(name="user", human_input_mode="NEVER")
    # result = user_proxy.initiate_chat(assistant, message="Hello!")


def example_auto_instrumentation():
    """Example 2: Auto-instrumentation (when integrated)."""
    print("\n" + "=" * 60)
    print("Example 2: Auto-Instrumentation (Future)")
    print("=" * 60)

    # Once AutoGen is added to the auto-instrumentation registry,
    # this single line will instrument everything:
    prela.init(service_name="autogen-demo", auto_instrument=True)

    print("✓ Auto-instrumentation enabled")
    print("  AutoGen (and other frameworks) automatically detected")


def example_two_agent_conversation():
    """Example 3: Two-agent conversation with tracing."""
    print("\n" + "=" * 60)
    print("Example 3: Two-Agent Conversation")
    print("=" * 60)

    # Initialize with console export for this example
    tracer = prela.init(
        service_name="autogen-conversation",
        exporter="console",
    )

    # Instrument AutoGen
    instrumentor = AutoGenInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Tracing to console (use exporter='file' for persistence)")

    # This is what the traced conversation would look like:
    print("\nConversation Structure (conceptual):")
    print("  User Proxy")
    print("  └─ Assistant (GPT-4)")

    print("\nTraced Spans (example):")
    print("  autogen.conversation.user->assistant (3.5s) ✓")
    print("  ├─ autogen.agent.assistant (1.2s) ✓")
    print("  ├─ autogen.agent.user (0.8s) ✓")
    print("  ├─ autogen.agent.assistant (1.0s) ✓")
    print("  └─ autogen.agent.user (0.5s) ✓")


def example_group_chat():
    """Example 4: Group chat with multiple agents."""
    print("\n" + "=" * 60)
    print("Example 4: Group Chat Workflow")
    print("=" * 60)

    # Initialize with console export for this example
    tracer = prela.init(
        service_name="autogen-group-chat",
        exporter="console",
    )

    # Instrument AutoGen
    instrumentor = AutoGenInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Tracing group chat")

    # This is what the traced group chat would look like:
    print("\nGroup Chat Structure (conceptual):")
    print("  Manager (group chat manager)")
    print("  ├─ Coder (assistant with code execution)")
    print("  ├─ Product Manager (planning agent)")
    print("  └─ User Proxy (human-in-the-loop)")

    print("\nTraced Spans (example):")
    print("  autogen.group_chat.manager (12.3s) ✓")
    print("  ├─ autogen.agent.product_manager (2.1s) ✓")
    print("  ├─ autogen.agent.coder (5.2s) ✓")
    print("  ├─ autogen.agent.user_proxy (0.5s) ✓")
    print("  └─ autogen.agent.coder (4.5s) ✓")


def example_captured_attributes():
    """Example 5: What attributes are captured."""
    print("\n" + "=" * 60)
    print("Example 5: Captured Attributes")
    print("=" * 60)

    print("\nConversation-Level Attributes:")
    print("  • conversation.id: UUID for this conversation")
    print("  • conversation.framework: 'autogen'")
    print("  • conversation.initiator: Initiating agent name")
    print("  • conversation.recipient: Recipient agent name")
    print("  • conversation.initial_message: First message (truncated)")
    print("  • conversation.max_turns: Maximum conversation turns")
    print("  • conversation.total_turns: Actual turns completed")
    print("  • conversation.total_tokens: Aggregated token usage")

    print("\nAgent-Level Attributes:")
    print("  • agent.id: Deterministic hash of framework:name")
    print("  • agent.name: Agent name (e.g., 'assistant')")
    print("  • agent.type: Agent class name")
    print("  • agent.framework: 'autogen'")
    print("  • reply.sender: Name of message sender")
    print("  • reply.num_messages: Number of messages in context")
    print("  • reply.content_length: Length of generated reply")

    print("\nGroup Chat Attributes:")
    print("  • group.manager: GroupChatManager name")
    print("  • group.framework: 'autogen'")
    print("  • group.num_agents: Number of agents in group")
    print("  • group.agent_names: List of agent names")
    print("  • group.speaker_selected: Events when speakers are selected")


def example_error_handling():
    """Example 6: Error handling and exceptions."""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    tracer = prela.init(service_name="autogen-demo", exporter="console")
    instrumentor = AutoGenInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Error handling features:")
    print("  • Exceptions captured in span events")
    print("  • Conversation status tracked")
    print("  • Original exception re-raised (user code sees it)")
    print("  • Instrumentation never crashes user code")

    print("\nExample trace with error:")
    print("  autogen.conversation.user->assistant (1.2s) ✗")
    print("  └─ autogen.agent.assistant (1.1s) ✗")
    print("     └─ exception: ValueError('Invalid response format')")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" AutoGen Instrumentation Examples")
    print("=" * 60)

    example_manual_instrumentation()
    example_auto_instrumentation()
    example_two_agent_conversation()
    example_group_chat()
    example_captured_attributes()
    example_error_handling()

    print("\n" + "=" * 60)
    print("✓ All examples complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Install AutoGen: pip install pyautogen")
    print("  2. Create agents with LLM configuration")
    print("  3. Call prela.init() and instrument()")
    print("  4. Run conversations - they're automatically traced!")
    print("\nFor real examples, see:")
    print("  • AutoGen docs: https://microsoft.github.io/autogen/")
    print("  • Prela docs: https://docs.prela.dev")


if __name__ == "__main__":
    main()
