"""
Multi-agent assertions for evaluating agent collaboration and coordination.
"""

from __future__ import annotations

from typing import Any

# Check tier before allowing multi-agent assertions
from prela.license import check_tier

if not check_tier("Multi-agent assertions", "lunch-money", silent=False):
    raise ImportError(
        "Multi-agent assertions require 'lunch-money' subscription or higher. "
        "Upgrade at https://prela.dev/pricing"
    )

from prela.core.span import Span
from prela.evals.assertions.base import AssertionResult, BaseAssertion


class AgentUsedAssertion(BaseAssertion):
    """Assert that a specific agent was used during execution.

    Example:
        >>> assertion = AgentUsedAssertion(agent_name="researcher", min_invocations=2)
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, agent_name: str, min_invocations: int = 1):
        """Initialize agent used assertion.

        Args:
            agent_name: Name of the agent that must be used
            min_invocations: Minimum number of times agent must be invoked
        """
        self.agent_name = agent_name
        self.min_invocations = min_invocations

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if specified agent was used."""
        if not trace:
            return AssertionResult(
                passed=False,
                assertion_type="agent_used",
                message="No trace data available to check agent usage",
                details={"agent_name": self.agent_name},
            )

        agent_spans = [
            s
            for s in trace
            if s.attributes.get("agent.name") == self.agent_name
        ]
        passed = len(agent_spans) >= self.min_invocations

        return AssertionResult(
            passed=passed,
            assertion_type="agent_used",
            message=f"Agent '{self.agent_name}' invoked {len(agent_spans)} times (min: {self.min_invocations})",
            expected=self.min_invocations,
            actual=len(agent_spans),
            details={"agent_name": self.agent_name, "invocations": len(agent_spans)},
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AgentUsedAssertion:
        """Create from configuration.

        Config format:
            {
                "agent_name": "researcher",
                "min_invocations": 2  # optional, default: 1
            }
        """
        if "agent_name" not in config:
            raise ValueError("AgentUsedAssertion requires 'agent_name' in config")

        return cls(
            agent_name=config["agent_name"],
            min_invocations=config.get("min_invocations", 1),
        )

    def __repr__(self) -> str:
        return f"AgentUsedAssertion(agent_name={self.agent_name!r}, min_invocations={self.min_invocations})"


class TaskCompletedAssertion(BaseAssertion):
    """Assert that a task was completed successfully.

    Example:
        >>> assertion = TaskCompletedAssertion(task_description_contains="research")
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, task_description_contains: str, expected_status: str = "completed"):
        """Initialize task completed assertion.

        Args:
            task_description_contains: Text that must be in task description
            expected_status: Expected task status (default: "completed")
        """
        self.task_description_contains = task_description_contains
        self.expected_status = expected_status

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if task was completed."""
        if not trace:
            return AssertionResult(
                passed=False,
                assertion_type="task_completed",
                message="No trace data available to check task completion",
                details={},
            )

        task_spans = [
            s
            for s in trace
            if "task." in s.name
            and self.task_description_contains.lower()
            in s.attributes.get("task.description", "").lower()
        ]

        if not task_spans:
            return AssertionResult(
                passed=False,
                assertion_type="task_completed",
                message=f"No task found containing '{self.task_description_contains}'",
                expected=self.expected_status,
                actual=None,
                details={"task_description_contains": self.task_description_contains},
            )

        completed = [
            s for s in task_spans if s.attributes.get("task.status") == self.expected_status
        ]
        actual_status = task_spans[0].attributes.get("task.status", "unknown")

        return AssertionResult(
            passed=len(completed) > 0,
            assertion_type="task_completed",
            message=f"Task '{self.task_description_contains}' status: {actual_status}",
            expected=self.expected_status,
            actual=actual_status,
            details={"found": len(task_spans), "completed": len(completed)},
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TaskCompletedAssertion:
        """Create from configuration.

        Config format:
            {
                "task_description_contains": "research",
                "expected_status": "completed"  # optional, default: "completed"
            }
        """
        if "task_description_contains" not in config:
            raise ValueError(
                "TaskCompletedAssertion requires 'task_description_contains' in config"
            )

        return cls(
            task_description_contains=config["task_description_contains"],
            expected_status=config.get("expected_status", "completed"),
        )

    def __repr__(self) -> str:
        return f"TaskCompletedAssertion(task_description_contains={self.task_description_contains!r}, expected_status={self.expected_status!r})"


class DelegationOccurredAssertion(BaseAssertion):
    """Assert that delegation occurred between agents.

    Example:
        >>> assertion = DelegationOccurredAssertion(from_agent="manager", to_agent="researcher")
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, from_agent: str | None = None, to_agent: str | None = None):
        """Initialize delegation assertion.

        Args:
            from_agent: Name of delegating agent (optional, matches any if None)
            to_agent: Name of receiving agent (optional, matches any if None)
        """
        self.from_agent = from_agent
        self.to_agent = to_agent

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if delegation occurred."""
        if not trace:
            return AssertionResult(
                passed=False,
                assertion_type="delegation_occurred",
                message="No trace data available to check delegation",
                details={},
            )

        delegations = []
        for span in trace:
            for event in span.events:
                if event.name == "agent.delegation":
                    attrs = event.attributes
                    if self.from_agent and attrs.get("delegation.from") != self.from_agent:
                        continue
                    if self.to_agent and attrs.get("delegation.to") != self.to_agent:
                        continue
                    delegations.append(attrs)

        passed = len(delegations) > 0
        direction = ""
        if self.from_agent and self.to_agent:
            direction = f" from {self.from_agent} to {self.to_agent}"
        elif self.from_agent:
            direction = f" from {self.from_agent}"
        elif self.to_agent:
            direction = f" to {self.to_agent}"

        return AssertionResult(
            passed=passed,
            assertion_type="delegation_occurred",
            message=f"Delegation{direction} {'occurred' if delegations else 'did not occur'}",
            expected=True,
            actual=passed,
            details={"delegations": len(delegations)},
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DelegationOccurredAssertion:
        """Create from configuration.

        Config format:
            {
                "from_agent": "manager",  # optional
                "to_agent": "researcher"   # optional
            }
        """
        return cls(
            from_agent=config.get("from_agent"),
            to_agent=config.get("to_agent"),
        )

    def __repr__(self) -> str:
        return f"DelegationOccurredAssertion(from_agent={self.from_agent!r}, to_agent={self.to_agent!r})"


class HandoffOccurredAssertion(BaseAssertion):
    """Assert that an agent handoff occurred (typically in Swarm pattern).

    Example:
        >>> assertion = HandoffOccurredAssertion(to_agent="specialist")
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, to_agent: str | None = None):
        """Initialize handoff assertion.

        Args:
            to_agent: Name of agent receiving handoff (optional, matches any if None)
        """
        self.to_agent = to_agent

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if handoff occurred."""
        if not trace:
            return AssertionResult(
                passed=False,
                assertion_type="handoff_occurred",
                message="No trace data available to check handoff",
                details={},
            )

        handoffs = []
        for span in trace:
            for event in span.events:
                if event.name == "agent.handoff":
                    attrs = event.attributes
                    if self.to_agent and attrs.get("handoff.to_agent") != self.to_agent:
                        continue
                    handoffs.append(attrs)

        passed = len(handoffs) > 0
        target = self.to_agent or "any agent"

        return AssertionResult(
            passed=passed,
            assertion_type="handoff_occurred",
            message=f"Handoff to {target}: {'found' if handoffs else 'not found'}",
            expected=True,
            actual=passed,
            details={"handoffs": len(handoffs)},
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HandoffOccurredAssertion:
        """Create from configuration.

        Config format:
            {
                "to_agent": "specialist"  # optional
            }
        """
        return cls(to_agent=config.get("to_agent"))

    def __repr__(self) -> str:
        return f"HandoffOccurredAssertion(to_agent={self.to_agent!r})"


class AgentCollaborationAssertion(BaseAssertion):
    """Assert that multiple agents collaborated during execution.

    Example:
        >>> assertion = AgentCollaborationAssertion(min_agents=3, required_agents=["manager", "researcher"])
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(
        self, min_agents: int = 2, required_agents: list[str] | None = None
    ):
        """Initialize collaboration assertion.

        Args:
            min_agents: Minimum number of distinct agents required
            required_agents: List of specific agents that must participate (optional)
        """
        self.min_agents = min_agents
        self.required_agents = required_agents

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check if agents collaborated."""
        if not trace:
            return AssertionResult(
                passed=False,
                assertion_type="agent_collaboration",
                message="No trace data available to check collaboration",
                details={},
            )

        agents_seen = set()
        for span in trace:
            agent_name = span.attributes.get("agent.name")
            if agent_name:
                agents_seen.add(agent_name)

        passed = len(agents_seen) >= self.min_agents
        missing = []

        if self.required_agents:
            missing = list(set(self.required_agents) - agents_seen)
            passed = passed and len(missing) == 0

        message = f"Found {len(agents_seen)} agents: {sorted(agents_seen)}"
        if missing:
            message += f" (missing required: {missing})"

        return AssertionResult(
            passed=passed,
            assertion_type="agent_collaboration",
            message=message,
            expected=self.min_agents,
            actual=len(agents_seen),
            details={
                "agents": sorted(agents_seen),
                "min": self.min_agents,
                "missing": missing,
            },
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AgentCollaborationAssertion:
        """Create from configuration.

        Config format:
            {
                "min_agents": 3,  # optional, default: 2
                "required_agents": ["manager", "researcher"]  # optional
            }
        """
        return cls(
            min_agents=config.get("min_agents", 2),
            required_agents=config.get("required_agents"),
        )

    def __repr__(self) -> str:
        return f"AgentCollaborationAssertion(min_agents={self.min_agents}, required_agents={self.required_agents!r})"


class ConversationTurnsAssertion(BaseAssertion):
    """Assert on the number of conversation turns.

    Example:
        >>> assertion = ConversationTurnsAssertion(min_turns=3, max_turns=10)
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self, min_turns: int | None = None, max_turns: int | None = None):
        """Initialize conversation turns assertion.

        Args:
            min_turns: Minimum number of conversation turns (optional)
            max_turns: Maximum number of conversation turns (optional)
        """
        self.min_turns = min_turns
        self.max_turns = max_turns

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check conversation turn count."""
        if not trace:
            return AssertionResult(
                passed=False,
                assertion_type="conversation_turns",
                message="No trace data available to check conversation turns",
                details={},
            )

        turn_count = 0
        for span in trace:
            if "conversation" in span.name:
                recorded = span.attributes.get("conversation.total_turns")
                if recorded:
                    turn_count = max(turn_count, recorded)

        passed = True
        constraints = []

        if self.min_turns is not None:
            if turn_count < self.min_turns:
                passed = False
            constraints.append(f"min: {self.min_turns}")

        if self.max_turns is not None:
            if turn_count > self.max_turns:
                passed = False
            constraints.append(f"max: {self.max_turns}")

        constraint_str = f" ({', '.join(constraints)})" if constraints else ""

        return AssertionResult(
            passed=passed,
            assertion_type="conversation_turns",
            message=f"Conversation had {turn_count} turns{constraint_str}",
            expected=f"{self.min_turns or 0}-{self.max_turns or '∞'}",
            actual=turn_count,
            details={
                "turns": turn_count,
                "min": self.min_turns,
                "max": self.max_turns,
            },
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConversationTurnsAssertion:
        """Create from configuration.

        Config format:
            {
                "min_turns": 3,  # optional
                "max_turns": 10  # optional
            }
        """
        return cls(
            min_turns=config.get("min_turns"),
            max_turns=config.get("max_turns"),
        )

    def __repr__(self) -> str:
        return f"ConversationTurnsAssertion(min_turns={self.min_turns}, max_turns={self.max_turns})"


class NoCircularDelegationAssertion(BaseAssertion):
    """Assert that no circular delegation patterns exist.

    Detects cycles where agent A delegates to B, B to C, and C back to A.

    Example:
        >>> assertion = NoCircularDelegationAssertion()
        >>> result = assertion.evaluate(output=None, expected=None, trace=spans)
        >>> assert result.passed
    """

    def __init__(self):
        """Initialize no circular delegation assertion."""
        pass

    def evaluate(
        self,
        output: Any,
        expected: Any | None,
        trace: list[Span] | None,
    ) -> AssertionResult:
        """Check for circular delegation patterns."""
        if not trace:
            return AssertionResult(
                passed=True,  # No trace means no cycles
                assertion_type="no_circular_delegation",
                message="No trace data to check for circular delegation",
                details={},
            )

        # Collect all delegations/handoffs
        edges = []
        for span in trace:
            for event in span.events:
                if event.name in ["agent.delegation", "agent.handoff"]:
                    attrs = event.attributes
                    from_a = attrs.get("delegation.from") or attrs.get(
                        "handoff.from_agent"
                    )
                    to_a = attrs.get("delegation.to") or attrs.get("handoff.to_agent")
                    if from_a and to_a:
                        edges.append((from_a, to_a))

        # Detect cycles using DFS
        cycles = self._detect_cycles(edges)

        return AssertionResult(
            passed=len(cycles) == 0,
            assertion_type="no_circular_delegation",
            message=f"{'No cycles found' if not cycles else f'Found {len(cycles)} cycle(s)'}",
            expected=0,
            actual=len(cycles),
            details={"cycles": [" → ".join(cycle) for cycle in cycles]},
        )

    def _detect_cycles(self, edges: list[tuple[str, str]]) -> list[list[str]]:
        """Detect cycles in delegation graph using DFS.

        Args:
            edges: List of (from_agent, to_agent) tuples

        Returns:
            List of cycles, where each cycle is a list of agent names
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {}
        for from_a, to_a in edges:
            if from_a not in graph:
                graph[from_a] = []
            graph[from_a].append(to_a)

        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NoCircularDelegationAssertion:
        """Create from configuration.

        Config format:
            {}  # No parameters required
        """
        return cls()

    def __repr__(self) -> str:
        return "NoCircularDelegationAssertion()"


__all__ = [
    "AgentUsedAssertion",
    "TaskCompletedAssertion",
    "DelegationOccurredAssertion",
    "HandoffOccurredAssertion",
    "AgentCollaborationAssertion",
    "ConversationTurnsAssertion",
    "NoCircularDelegationAssertion",
]
