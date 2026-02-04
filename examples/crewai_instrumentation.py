"""Example: CrewAI Instrumentation

This example demonstrates how to use Prela to trace CrewAI multi-agent crews.

Requirements:
    pip install crewai prela

Note: This example requires actual CrewAI to be installed. The code shown
demonstrates the API but won't run without a real CrewAI installation.
"""

import prela
from prela.instrumentation.multi_agent import CrewAIInstrumentor


def example_manual_instrumentation():
    """Example 1: Manual instrumentation of CrewAI."""
    print("=" * 60)
    print("Example 1: Manual Instrumentation")
    print("=" * 60)

    # Initialize Prela tracer
    tracer = prela.init(service_name="crewai-demo", exporter="console")

    # Manually instrument CrewAI
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ CrewAI instrumented successfully")
    print("  All crew executions will now be traced automatically")

    # Now use CrewAI normally - all operations are traced!
    # from crewai import Agent, Task, Crew
    # agent = Agent(role="researcher", goal="Research AI", backstory="Expert")
    # task = Task(description="Research AI agents", agent=agent)
    # crew = Crew(agents=[agent], tasks=[task])
    # result = crew.kickoff()  # ✨ Automatically traced!


def example_auto_instrumentation():
    """Example 2: Auto-instrumentation (when integrated)."""
    print("\n" + "=" * 60)
    print("Example 2: Auto-Instrumentation (Future)")
    print("=" * 60)

    # Once CrewAI is added to the auto-instrumentation registry,
    # this single line will instrument everything:
    prela.init(service_name="crewai-demo", auto_instrument=True)

    print("✓ Auto-instrumentation enabled")
    print("  CrewAI (and other frameworks) automatically detected")


def example_multi_agent_crew():
    """Example 3: Multi-agent crew with full tracing."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Agent Crew Workflow")
    print("=" * 60)

    # Initialize with console export for this example
    tracer = prela.init(
        service_name="content-creation-crew",
        exporter="console",
    )

    # Instrument CrewAI
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Tracing to console (use exporter='file' for persistence)")

    # This is what the traced crew execution would look like:
    print("\nCrew Structure (conceptual):")
    print("  Manager (delegation enabled)")
    print("  ├─ AI Researcher (tools: search, scrape)")
    print("  ├─ Technical Writer (tools: write)")
    print("  └─ Content Critic")

    print("\nTasks:")
    print("  1. Research AI agents → AI Researcher")
    print("  2. Write article → Technical Writer")
    print("  3. Review article → Content Critic")

    print("\nTraced Spans (example):")
    print("  crewai.crew.content_creation_crew (5.2s) ✓")
    print("  ├─ crewai.agent.AI Researcher (2.1s) ✓")
    print("  │  └─ crewai.task.Research AI agents (2.0s) ✓")
    print("  ├─ crewai.agent.Technical Writer (1.8s) ✓")
    print("  │  └─ crewai.task.Write article (1.7s) ✓")
    print("  └─ crewai.agent.Content Critic (1.3s) ✓")
    print("     └─ crewai.task.Review article (1.2s) ✓")


def example_captured_attributes():
    """Example 4: What attributes are captured."""
    print("\n" + "=" * 60)
    print("Example 4: Captured Attributes")
    print("=" * 60)

    print("\nCrew-Level Attributes:")
    print("  • crew.execution_id: UUID for this execution")
    print("  • crew.framework: 'crewai'")
    print("  • crew.num_agents: Number of agents in crew")
    print("  • crew.num_tasks: Number of tasks")
    print("  • crew.process: 'sequential' or 'hierarchical'")
    print("  • crew.agent_names: List of agent roles")
    print("  • crew.total_llm_calls: Aggregated LLM calls")
    print("  • crew.total_tokens: Aggregated token usage")
    print("  • crew.total_cost_usd: Estimated cost")

    print("\nAgent-Level Attributes:")
    print("  • agent.id: Deterministic hash of framework:role")
    print("  • agent.name: Agent role (e.g., 'AI Researcher')")
    print("  • agent.framework: 'crewai'")
    print("  • agent.goal: Agent's goal")
    print("  • agent.tools: List of tool names")
    print("  • agent.output_length: Length of agent output")

    print("\nTask-Level Attributes:")
    print("  • task.id: UUID for this task execution")
    print("  • task.description: Task description")
    print("  • task.expected_output: Expected output format")
    print("  • task.agent: Assigned agent role")
    print("  • task.output: Task result (truncated to 1000 chars)")
    print("  • task.status: 'completed' or 'failed'")


def example_error_handling():
    """Example 5: Error handling and exceptions."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    tracer = prela.init(service_name="crewai-demo", exporter="console")
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer)

    print("✓ Error handling features:")
    print("  • Exceptions captured in span events")
    print("  • Crew status set to 'failed' on error")
    print("  • Task status set to 'failed' on error")
    print("  • Original exception re-raised (user code sees it)")
    print("  • Instrumentation never crashes user code")

    print("\nExample trace with error:")
    print("  crewai.crew.test_crew (1.2s) ✗")
    print("  └─ crewai.agent.researcher (1.1s) ✗")
    print("     └─ crewai.task.Research topic (1.0s) ✗")
    print("        └─ exception: ValueError('API key missing')")


def example_role_mapping():
    """Example 6: How CrewAI roles are mapped."""
    print("\n" + "=" * 60)
    print("Example 6: Role Mapping")
    print("=" * 60)

    print("\nCrewAI Role → Prela AgentRole:")
    print("  'Project Manager' → MANAGER")
    print("  'Team Lead' → MANAGER")
    print("  'Code Critic' → CRITIC")
    print("  'Content Reviewer' → CRITIC")
    print("  'AI Specialist' → SPECIALIST")
    print("  'Security Expert' → SPECIALIST")
    print("  'Data Processor' → WORKER (default)")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" CrewAI Instrumentation Examples")
    print("=" * 60)

    example_manual_instrumentation()
    example_auto_instrumentation()
    example_multi_agent_crew()
    example_captured_attributes()
    example_error_handling()
    example_role_mapping()

    print("\n" + "=" * 60)
    print("✓ All examples complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Install CrewAI: pip install crewai")
    print("  2. Create a crew with agents and tasks")
    print("  3. Call prela.init() and instrument()")
    print("  4. Run crew.kickoff() - it's automatically traced!")
    print("\nFor real examples, see:")
    print("  • CrewAI docs: https://docs.crewai.com")
    print("  • Prela docs: https://docs.prela.dev")


if __name__ == "__main__":
    main()
