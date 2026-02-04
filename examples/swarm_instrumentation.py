"""Example: Swarm Instrumentation

This example demonstrates how to use Prela to trace OpenAI Swarm multi-agent executions.

Requirements:
    pip install git+https://github.com/openai/swarm.git prela

Note: This example requires actual Swarm to be installed. The code shown
demonstrates the API but won't run without a real Swarm installation.
"""

import prela
from prela.instrumentation.multi_agent import SwarmInstrumentor


def example_manual_instrumentation():
    """Example 1: Manual instrumentation of Swarm."""
    print("=" * 60)
    print("Example 1: Manual Instrumentation")
    print("=" * 60)

    # Initialize Prela tracer
    tracer = prela.init(service_name="swarm-demo", exporter="console")

    # Manually instrument Swarm
    instrumentor = SwarmInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Swarm instrumented successfully")
    print("  All executions will now be traced automatically")

    # Now use Swarm normally - all operations are traced!
    # from swarm import Swarm, Agent
    # client = Swarm()
    # agent = Agent(name="Assistant")
    # response = client.run(agent=agent, messages=[...])


def example_auto_instrumentation():
    """Example 2: Auto-instrumentation (when integrated)."""
    print("\n" + "=" * 60)
    print("Example 2: Auto-Instrumentation (Future)")
    print("=" * 60)

    # Once Swarm is added to the auto-instrumentation registry,
    # this single line will instrument everything:
    prela.init(service_name="swarm-demo", auto_instrument=True)

    print("✓ Auto-instrumentation enabled")
    print("  Swarm (and other frameworks) automatically detected")


def example_simple_agent():
    """Example 3: Simple agent execution with tracing."""
    print("\n" + "=" * 60)
    print("Example 3: Simple Agent Execution")
    print("=" * 60)

    # Initialize with console export for this example
    tracer = prela.init(
        service_name="swarm-simple",
        exporter="console",
    )

    # Instrument Swarm
    instrumentor = SwarmInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Tracing to console (use exporter='file' for persistence)")

    # This is what the traced execution would look like:
    print("\nExecution Structure (conceptual):")
    print("  Swarm Client")
    print("  └─ Assistant Agent")

    print("\nTraced Spans (example):")
    print("  swarm.run.assistant (2.5s) ✓")
    print("    execution_id: abc-123")
    print("    initial_agent: assistant")
    print("    final_agent: assistant")
    print("    num_messages: 2")


def example_agent_handoff():
    """Example 4: Agent handoff with tracing."""
    print("\n" + "=" * 60)
    print("Example 4: Agent Handoff Workflow")
    print("=" * 60)

    # Initialize with console export for this example
    tracer = prela.init(
        service_name="swarm-handoff",
        exporter="console",
    )

    # Instrument Swarm
    instrumentor = SwarmInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Tracing agent handoffs")

    # This is what the traced handoff would look like:
    print("\nHandoff Structure (conceptual):")
    print("  Swarm Client")
    print("  ├─ Triage Agent (analyzes request)")
    print("  ├─ Sales Agent (handles purchase)")
    print("  └─ Support Agent (provides help)")

    print("\nTraced Spans (example):")
    print("  swarm.run.triage_agent (5.2s) ✓")
    print("    initial_agent: triage_agent")
    print("    final_agent: sales_agent")
    print("    agents_used: [triage_agent, sales_agent]")
    print("    num_handoffs: 1")


def example_context_variables():
    """Example 5: Context variables tracking."""
    print("\n" + "=" * 60)
    print("Example 5: Context Variables")
    print("=" * 60)

    print("\nContext Variables Captured:")
    print("  • swarm.execution_id: UUID for this execution")
    print("  • swarm.framework: 'swarm'")
    print("  • swarm.initial_agent: Starting agent name")
    print("  • swarm.final_agent: Ending agent name")
    print("  • swarm.num_messages: Message count")
    print("  • swarm.context_variables: Context var keys")
    print("  • swarm.updated_context: Updated context keys")

    print("\nAgent-Level Attributes:")
    print("  • agent.id: Deterministic hash of framework:name")
    print("  • agent.name: Agent name (e.g., 'assistant')")
    print("  • agent.framework: 'swarm'")

    print("\nExecution Attributes:")
    print("  • swarm.agents_used: List of agents in execution")
    print("  • swarm.num_handoffs: Number of agent transitions")
    print("  • swarm.response_messages: Number of response messages")


def example_error_handling():
    """Example 6: Error handling and exceptions."""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    tracer = prela.init(service_name="swarm-demo", exporter="console")
    instrumentor = SwarmInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Error handling features:")
    print("  • Exceptions captured in span events")
    print("  • Execution status tracked")
    print("  • Original exception re-raised (user code sees it)")
    print("  • Instrumentation never crashes user code")

    print("\nExample trace with error:")
    print("  swarm.run.assistant (1.2s) ✗")
    print("     └─ exception: ValueError('Invalid agent transition')")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" Swarm Instrumentation Examples")
    print("=" * 60)

    example_manual_instrumentation()
    example_auto_instrumentation()
    example_simple_agent()
    example_agent_handoff()
    example_context_variables()
    example_error_handling()

    print("\n" + "=" * 60)
    print("✓ All examples complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Install Swarm: pip install git+https://github.com/openai/swarm.git")
    print("  2. Create agents with instructions and functions")
    print("  3. Call prela.init() and instrument()")
    print("  4. Run executions - they're automatically traced!")
    print("\nFor real examples, see:")
    print("  • Swarm docs: https://github.com/openai/swarm")
    print("  • Prela docs: https://docs.prela.dev")


if __name__ == "__main__":
    main()
