"""
Example usage of multi-agent assertions for evaluating agent collaboration.

This example demonstrates how to use the multi-agent assertions to verify
that agents properly collaborate, delegate tasks, and complete work in
multi-agent systems like CrewAI, AutoGen, LangGraph, and Swarm.
"""

from prela.core.span import Span, SpanType
from prela.evals.assertions import (
    AgentCollaborationAssertion,
    AgentUsedAssertion,
    ConversationTurnsAssertion,
    DelegationOccurredAssertion,
    HandoffOccurredAssertion,
    NoCircularDelegationAssertion,
    TaskCompletedAssertion,
)
import uuid


def create_example_trace():
    """Create an example trace with multi-agent collaboration."""
    trace_id = str(uuid.uuid4())

    # Manager agent delegates to researcher
    manager_span = Span(
        trace_id=trace_id,
        span_id=str(uuid.uuid4()),
        name="manager.execute",
        span_type=SpanType.AGENT,
        attributes={"agent.name": "manager"},
    )
    manager_span.add_event(
        name="agent.delegation",
        attributes={
            "delegation.from": "manager",
            "delegation.to": "researcher",
        },
    )
    manager_span.end()

    # Researcher agent completes research task
    researcher_span = Span(
        trace_id=trace_id,
        span_id=str(uuid.uuid4()),
        name="researcher.execute",
        span_type=SpanType.AGENT,
        attributes={"agent.name": "researcher"},
    )
    researcher_span.end()

    # Task completion
    task_span = Span(
        trace_id=trace_id,
        span_id=str(uuid.uuid4()),
        name="task.research_market",
        span_type=SpanType.CUSTOM,
        attributes={
            "task.description": "Research the AI market",
            "task.status": "completed",
        },
    )
    task_span.end()

    return [manager_span, researcher_span, task_span]


def example_agent_used():
    """Example: Verify that a specific agent was used."""
    print("=" * 60)
    print("Example 1: AgentUsedAssertion")
    print("=" * 60)

    trace = create_example_trace()

    # Check that manager agent was used
    assertion = AgentUsedAssertion(agent_name="manager", min_invocations=1)
    result = assertion.evaluate(output=None, expected=None, trace=trace)
    print(f"\n{result}")
    print(f"Passed: {result.passed}")
    print(f"Details: {result.details}")


def example_agent_collaboration():
    """Example: Verify that multiple agents collaborated."""
    print("\n" + "=" * 60)
    print("Example 2: AgentCollaborationAssertion")
    print("=" * 60)

    trace = create_example_trace()

    # Check that at least 2 agents collaborated
    assertion = AgentCollaborationAssertion(
        min_agents=2,
        required_agents=["manager", "researcher"],
    )
    result = assertion.evaluate(output=None, expected=None, trace=trace)
    print(f"\n{result}")
    print(f"Passed: {result.passed}")
    print(f"Agents found: {result.details['agents']}")


def example_delegation():
    """Example: Verify that delegation occurred."""
    print("\n" + "=" * 60)
    print("Example 3: DelegationOccurredAssertion")
    print("=" * 60)

    trace = create_example_trace()

    # Check that manager delegated to researcher
    assertion = DelegationOccurredAssertion(
        from_agent="manager",
        to_agent="researcher",
    )
    result = assertion.evaluate(output=None, expected=None, trace=trace)
    print(f"\n{result}")
    print(f"Passed: {result.passed}")
    print(f"Delegations: {result.details['delegations']}")


def example_task_completed():
    """Example: Verify that a task was completed."""
    print("\n" + "=" * 60)
    print("Example 4: TaskCompletedAssertion")
    print("=" * 60)

    trace = create_example_trace()

    # Check that research task was completed
    assertion = TaskCompletedAssertion(
        task_description_contains="research",
        expected_status="completed",
    )
    result = assertion.evaluate(output=None, expected=None, trace=trace)
    print(f"\n{result}")
    print(f"Passed: {result.passed}")
    print(f"Tasks found: {result.details['found']}")
    print(f"Tasks completed: {result.details['completed']}")


def example_no_circular_delegation():
    """Example: Verify no circular delegation patterns."""
    print("\n" + "=" * 60)
    print("Example 5: NoCircularDelegationAssertion")
    print("=" * 60)

    trace = create_example_trace()

    # Check for circular delegation (should pass - no cycles)
    assertion = NoCircularDelegationAssertion()
    result = assertion.evaluate(output=None, expected=None, trace=trace)
    print(f"\n{result}")
    print(f"Passed: {result.passed}")
    print(f"Cycles found: {result.details.get('cycles', [])}")


def example_from_config():
    """Example: Create assertions from configuration dicts."""
    print("\n" + "=" * 60)
    print("Example 6: Creating Assertions from Config")
    print("=" * 60)

    trace = create_example_trace()

    # Create assertion from config dict
    config = {
        "agent_name": "manager",
        "min_invocations": 1,
    }
    assertion = AgentUsedAssertion.from_config(config)
    result = assertion.evaluate(output=None, expected=None, trace=trace)
    print(f"\nCreated from config: {config}")
    print(f"{result}")
    print(f"Passed: {result.passed}")


def example_with_eval_framework():
    """Example: Using multi-agent assertions in eval framework."""
    print("\n" + "=" * 60)
    print("Example 7: Integration with Eval Framework")
    print("=" * 60)

    from prela.evals import EvalCase, EvalInput, EvalRunner, EvalSuite

    # Define test cases with multi-agent assertions
    cases = [
        EvalCase(
            id="test_collaboration",
            name="Multi-agent collaboration",
            input=EvalInput(query="Research and summarize AI trends"),
            assertions=[
                AgentCollaborationAssertion(min_agents=2),
                DelegationOccurredAssertion(from_agent="manager"),
                NoCircularDelegationAssertion(),
            ],
        )
    ]

    suite = EvalSuite(name="Multi-Agent Tests", cases=cases)

    print("\nEval Suite Configuration:")
    print(f"  Suite: {suite.name}")
    print(f"  Cases: {len(suite.cases)}")
    print(f"  Assertions per case: {len(cases[0].assertions)}")
    print("\nNote: Run with EvalRunner to execute tests")


if __name__ == "__main__":
    print("\n🤖 Multi-Agent Assertions Examples\n")

    example_agent_used()
    example_agent_collaboration()
    example_delegation()
    example_task_completed()
    example_no_circular_delegation()
    example_from_config()
    example_with_eval_framework()

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - sdk/prela/evals/assertions/multi_agent.py")
    print("  - sdk/prela/evals/assertions/README.md")
