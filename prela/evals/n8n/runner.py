"""
Evaluation runner for n8n workflows.

This module provides specialized evaluation tools for testing n8n workflows by:
1. Triggering workflows via n8n API/webhook
2. Waiting for execution completion
3. Fetching execution results
4. Running assertions on nodes and workflow outputs
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from prela.core.clock import now
from prela.core.context import get_current_trace_id
from prela.core.span import SpanType
from prela.core.tracer import Tracer
from prela.evals.assertions.base import AssertionResult, BaseAssertion
from prela.evals.suite import EvalSuite


@dataclass
class N8nWorkflowEvalConfig:
    """Configuration for evaluating an n8n workflow.

    Attributes:
        workflow_id: n8n workflow ID to test
        n8n_base_url: Base URL of n8n instance (default: http://localhost:5678)
        n8n_api_key: API key for n8n authentication (optional)
        timeout_seconds: Maximum seconds to wait for workflow completion (default: 120)
        capture_traces: Whether to capture traces during execution (default: True)

    Example:
        >>> config = N8nWorkflowEvalConfig(
        ...     workflow_id="abc123",
        ...     n8n_base_url="https://n8n.example.com",
        ...     n8n_api_key="your-api-key",
        ...     timeout_seconds=60
        ... )
    """

    workflow_id: str
    n8n_base_url: str = "http://localhost:5678"
    n8n_api_key: Optional[str] = None
    timeout_seconds: int = 120
    capture_traces: bool = True


@dataclass
class N8nEvalCase:
    """Test case for an n8n workflow.

    Unlike regular EvalCase which requires EvalInput, N8nEvalCase uses trigger_data
    to start workflows and includes n8n-specific assertion capabilities.

    Attributes:
        id: Unique test case ID
        name: Human-readable test case name
        trigger_data: Data to send when triggering the workflow
        node_assertions: Mapping of node_name -> list of assertions to run on that node's output
        workflow_assertions: List of assertions to run on the complete workflow execution
        expected_output: Expected final output from the workflow (optional)
        tags: Optional tags for filtering/grouping
        timeout_seconds: Maximum execution time for this test case
        metadata: Additional metadata for this test case

    Example:
        >>> from prela.evals.assertions import ContainsAssertion
        >>> case = N8nEvalCase(
        ...     id="test_lead_scoring",
        ...     name="High-intent lead classification",
        ...     trigger_data={
        ...         "email": "I want to buy your product immediately",
        ...         "company": "ACME Corp"
        ...     },
        ...     node_assertions={
        ...         "AI Intent Classifier": [
        ...             ContainsAssertion(text="high_intent")
        ...         ],
        ...         "Lead Scorer": [
        ...             ContainsAssertion(text="score")
        ...         ]
        ...     },
        ...     expected_output={"intent": "high_intent", "score": 90}
        ... )
    """

    id: str
    name: str
    trigger_data: dict = field(default_factory=dict)
    node_assertions: Optional[dict[str, list[BaseAssertion]]] = None
    workflow_assertions: Optional[list[BaseAssertion]] = None
    expected_output: Optional[Any] = None
    tags: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


class N8nWorkflowEvalRunner:
    """
    Runs evaluations against n8n workflows.

    This runner triggers n8n workflows via the API, waits for completion,
    fetches results, and runs assertions on both node-level and workflow-level outputs.

    Example:
        >>> import asyncio
        >>> from prela.evals.n8n import N8nWorkflowEvalConfig, N8nWorkflowEvalRunner, N8nEvalCase
        >>>
        >>> config = N8nWorkflowEvalConfig(workflow_id="abc123")
        >>> runner = N8nWorkflowEvalRunner(config)
        >>>
        >>> case = N8nEvalCase(
        ...     id="test_1",
        ...     name="Test workflow",
        ...     trigger_data={"message": "Hello"}
        ... )
        >>>
        >>> result = asyncio.run(runner.run_case(case))
        >>> print(result["passed"])
    """

    def __init__(self, config: N8nWorkflowEvalConfig, tracer: Optional[Tracer] = None):
        """Initialize the n8n workflow evaluation runner.

        Args:
            config: Configuration for the n8n workflow evaluation
            tracer: Optional tracer for capturing execution traces
        """
        self.config = config
        self.tracer = tracer
        self.client = httpx.AsyncClient(
            base_url=config.n8n_base_url,
            headers=(
                {"X-N8N-API-KEY": config.n8n_api_key} if config.n8n_api_key else {}
            ),
            timeout=config.timeout_seconds,
        )

    async def run_case(self, case: N8nEvalCase) -> dict:
        """Run a single eval case against the n8n workflow.

        Args:
            case: The test case to run

        Returns:
            Dictionary with execution results including:
                - execution_id: n8n execution ID
                - status: Execution status (success, error, crashed)
                - duration_ms: Execution duration in milliseconds
                - node_results: Assertion results per node
                - workflow_results: Workflow-level assertion results
                - passed: Whether all assertions passed
                - output_mismatch: If expected_output provided and doesn't match

        Example:
            >>> result = await runner.run_case(case)
            >>> print(f"Passed: {result['passed']}")
            >>> print(f"Duration: {result['duration_ms']}ms")
        """
        start_time = time.perf_counter()

        # Create span if tracer available
        span = None
        if self.tracer and self.config.capture_traces:
            span = self.tracer.start_span(
                name=f"n8n.eval.{case.name}",
                span_type=SpanType.AGENT,
                attributes={
                    "eval.case_id": case.id,
                    "eval.case_name": case.name,
                    "n8n.workflow_id": self.config.workflow_id,
                },
            )

        try:
            # 1. Trigger the workflow
            execution_id = await self._trigger_workflow(case.trigger_data)

            # 2. Wait for completion
            execution_result = await self._wait_for_completion(execution_id)

            # 3. Build results structure
            duration_ms = (time.perf_counter() - start_time) * 1000

            results = {
                "execution_id": execution_id,
                "status": execution_result["status"],
                "duration_ms": duration_ms,
                "node_results": {},
                "workflow_results": [],
                "passed": True,
                "trace_id": get_current_trace_id() if self.tracer else None,
            }

            # 4. Run node-level assertions
            if case.node_assertions:
                for node_name, assertions in case.node_assertions.items():
                    node_data = self._get_node_data(execution_result, node_name)
                    node_results = []

                    for assertion in assertions:
                        try:
                            result = assertion.evaluate(
                                output=node_data, expected=None, trace=None
                            )
                            node_results.append(result)
                            if not result.passed:
                                results["passed"] = False
                        except Exception as e:
                            # Assertion evaluation failed
                            error_result = AssertionResult(
                                passed=False,
                                assertion_type="error",
                                message=f"Assertion failed: {str(e)}",
                                expected=None,
                                actual=None,
                            )
                            node_results.append(error_result)
                            results["passed"] = False

                    results["node_results"][node_name] = node_results

            # 5. Run workflow-level assertions
            if case.workflow_assertions:
                for assertion in case.workflow_assertions:
                    try:
                        result = assertion.evaluate(
                            output=execution_result, expected=None, trace=None
                        )
                        results["workflow_results"].append(result)
                        if not result.passed:
                            results["passed"] = False
                    except Exception as e:
                        error_result = AssertionResult(
                            passed=False,
                            assertion_type="error",
                            message=f"Assertion failed: {str(e)}",
                            expected=None,
                            actual=None,
                        )
                        results["workflow_results"].append(error_result)
                        results["passed"] = False

            # 6. Check expected output if provided
            if case.expected_output is not None:
                actual_output = execution_result.get("output")
                if actual_output != case.expected_output:
                    results["passed"] = False
                    results["output_mismatch"] = {
                        "expected": case.expected_output,
                        "actual": actual_output,
                    }

            # End span with success
            if span:
                span.set_attribute("eval.passed", results["passed"])
                span.set_attribute("eval.duration_ms", duration_ms)
                span.end()

            return results

        except Exception as e:
            # Execution failed
            duration_ms = (time.perf_counter() - start_time) * 1000

            if span:
                span.set_attribute("eval.passed", False)
                span.set_attribute("eval.error", str(e))
                span.end()

            return {
                "execution_id": None,
                "status": "error",
                "duration_ms": duration_ms,
                "node_results": {},
                "workflow_results": [],
                "passed": False,
                "error": str(e),
                "trace_id": get_current_trace_id() if self.tracer else None,
            }

    async def run_suite(self, suite: EvalSuite) -> dict:
        """Run a full evaluation suite against the n8n workflow.

        Args:
            suite: The evaluation suite containing test cases

        Returns:
            Dictionary with aggregated results:
                - suite_name: Name of the suite
                - total: Total number of test cases
                - passed: Number of passed test cases
                - failed: Number of failed test cases
                - cases: List of individual case results

        Example:
            >>> suite = EvalSuite(name="Lead Scoring Tests", cases=[case1, case2])
            >>> results = await runner.run_suite(suite)
            >>> print(f"Pass rate: {results['passed']}/{results['total']}")
        """
        results = {
            "suite_name": suite.name,
            "total": len(suite.cases),
            "passed": 0,
            "failed": 0,
            "cases": [],
        }

        # Run setup if provided
        if suite.setup:
            try:
                suite.setup()
            except Exception as e:
                # Setup failed, abort suite
                return {
                    **results,
                    "setup_error": str(e),
                    "failed": len(suite.cases),
                }

        # Execute each case
        for case in suite.cases:
            case_result = await self.run_case(case)
            results["cases"].append(case_result)

            if case_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

        # Run teardown if provided
        if suite.teardown:
            try:
                suite.teardown()
            except Exception as e:
                # Teardown failed, include in results
                results["teardown_error"] = str(e)

        return results

    async def _trigger_workflow(self, trigger_data: dict) -> str:
        """Trigger the n8n workflow and return execution ID.

        Args:
            trigger_data: Data to send to the workflow trigger

        Returns:
            Execution ID from n8n

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        # Use n8n API to execute workflow
        # POST /api/v1/workflows/{workflow_id}/execute
        response = await self.client.post(
            f"/api/v1/workflows/{self.config.workflow_id}/execute",
            json={"data": trigger_data},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"]["executionId"]

    async def _wait_for_completion(self, execution_id: str) -> dict:
        """Poll for workflow execution completion.

        Args:
            execution_id: n8n execution ID to poll

        Returns:
            Execution result data from n8n

        Raises:
            TimeoutError: If execution doesn't complete within timeout
            httpx.HTTPStatusError: If the API request fails
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # GET /api/v1/executions/{execution_id}
            response = await self.client.get(f"/api/v1/executions/{execution_id}")
            response.raise_for_status()
            data = response.json()["data"]

            # Check if execution completed
            if data["status"] in ["success", "error", "crashed"]:
                return data

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.config.timeout_seconds:
                raise TimeoutError(
                    f"Workflow execution {execution_id} timed out after {self.config.timeout_seconds}s"
                )

            # Wait before polling again
            await asyncio.sleep(1)

    def _get_node_data(
        self, execution_result: dict, node_name: str
    ) -> Optional[dict]:
        """Extract data for a specific node from execution result.

        Args:
            execution_result: Complete execution result from n8n (already unwrapped from response.json()["data"])
            node_name: Name of the node to extract data for

        Returns:
            Node data dictionary or None if node not found
        """
        # Try different possible structures
        # Structure 1: resultData.runData[node_name]
        if "resultData" in execution_result:
            run_data = execution_result["resultData"].get("runData", {})
            if node_name in run_data:
                return run_data[node_name]

        # Structure 2: nodes array
        for node in execution_result.get("nodes", []):
            if node.get("name") == node_name:
                return node

        return None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def eval_n8n_workflow(
    workflow_id: str,
    test_cases: list[N8nEvalCase],
    n8n_url: str = "http://localhost:5678",
    n8n_api_key: Optional[str] = None,
    timeout_seconds: int = 120,
    tracer: Optional[Tracer] = None,
) -> dict:
    """
    Quick way to run evaluations against an n8n workflow.

    This is a convenience function that sets up the configuration, runner,
    and suite, then executes all test cases.

    Args:
        workflow_id: n8n workflow ID to test
        test_cases: List of N8nEvalCase instances
        n8n_url: Base URL of n8n instance (default: http://localhost:5678)
        n8n_api_key: API key for n8n authentication (optional)
        timeout_seconds: Maximum seconds to wait for each execution (default: 120)
        tracer: Optional tracer for capturing execution traces

    Returns:
        Dictionary with evaluation results (see run_suite for structure)

    Example:
        >>> from prela.evals.n8n import eval_n8n_workflow, N8nEvalCase
        >>> from prela.evals.assertions import ContainsAssertion
        >>>
        >>> results = await eval_n8n_workflow(
        ...     workflow_id="abc123",
        ...     test_cases=[
        ...         N8nEvalCase(
        ...             id="test_1",
        ...             name="High-intent lead",
        ...             trigger_data={"email": "I want to buy..."},
        ...             node_assertions={
        ...                 "Classify Intent": [
        ...                     ContainsAssertion(substring="high")
        ...                 ]
        ...             }
        ...         )
        ...     ],
        ...     n8n_url="https://n8n.example.com",
        ...     n8n_api_key="your-api-key"
        ... )
        >>> print(f"Pass rate: {results['passed']}/{results['total']}")
    """
    config = N8nWorkflowEvalConfig(
        workflow_id=workflow_id,
        n8n_base_url=n8n_url,
        n8n_api_key=n8n_api_key,
        timeout_seconds=timeout_seconds,
    )

    runner = N8nWorkflowEvalRunner(config, tracer=tracer)

    try:
        suite = EvalSuite(name=f"n8n-{workflow_id}", cases=test_cases)
        return await runner.run_suite(suite)
    finally:
        await runner.close()
