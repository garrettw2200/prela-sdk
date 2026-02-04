"""Tests for n8n-specific assertions."""

from __future__ import annotations

import pytest

from prela.evals.n8n.assertions import (
    N8nAINodeTokens,
    N8nNodeCompleted,
    N8nNodeOutput,
    N8nWorkflowDuration,
    N8nWorkflowStatus,
    duration_under,
    node_completed,
    node_output,
    tokens_under,
    workflow_completed,
    workflow_status,
)


class TestN8nNodeCompleted:
    """Tests for N8nNodeCompleted assertion."""

    def test_node_completed_success(self):
        """Test node completed successfully."""
        assertion = N8nNodeCompleted(node_name="Node1")
        execution_result = {
            "nodes": [
                {"name": "Node1", "status": "success"},
                {"name": "Node2", "status": "running"},
            ]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert "Node1" in result.message
        assert "completed successfully" in result.message
        assert result.expected == "success"
        assert result.actual == "success"

    def test_node_completed_failed(self):
        """Test node failed."""
        assertion = N8nNodeCompleted(node_name="Node1")
        execution_result = {
            "nodes": [{"name": "Node1", "status": "error"}]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "failed" in result.message
        assert result.actual == "error"

    def test_node_not_found(self):
        """Test node not found in execution."""
        assertion = N8nNodeCompleted(node_name="NonExistent")
        execution_result = {"nodes": [{"name": "Node1", "status": "success"}]}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "not found" in result.message

    def test_from_config(self):
        """Test creating from config."""
        config = {"node_name": "TestNode"}
        assertion = N8nNodeCompleted.from_config(config)

        assert assertion.node_name == "TestNode"


class TestN8nNodeOutput:
    """Tests for N8nNodeOutput assertion."""

    def test_node_output_matches(self):
        """Test node output matches expected value."""
        assertion = N8nNodeOutput(
            node_name="Node1", path="response.status", expected_value=200
        )
        execution_result = {
            "nodes": [
                {"name": "Node1", "output": {"response": {"status": 200}}}
            ]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert "matches" in result.message
        assert result.expected == 200
        assert result.actual == 200

    def test_node_output_mismatch(self):
        """Test node output doesn't match expected value."""
        assertion = N8nNodeOutput(
            node_name="Node1", path="response.status", expected_value=200
        )
        execution_result = {
            "nodes": [
                {"name": "Node1", "output": {"response": {"status": 404}}}
            ]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "does not match" in result.message
        assert result.expected == 200
        assert result.actual == 404

    def test_nested_path(self):
        """Test nested path extraction."""
        assertion = N8nNodeOutput(
            node_name="Node1", path="data.user.id", expected_value=123
        )
        execution_result = {
            "nodes": [
                {"name": "Node1", "output": {"data": {"user": {"id": 123}}}}
            ]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert result.actual == 123

    def test_path_not_found(self):
        """Test path doesn't exist in output."""
        assertion = N8nNodeOutput(
            node_name="Node1", path="missing.path", expected_value="value"
        )
        execution_result = {
            "nodes": [{"name": "Node1", "output": {"other": "data"}}]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert result.actual is None

    def test_node_not_found(self):
        """Test node not found."""
        assertion = N8nNodeOutput(
            node_name="NonExistent", path="data", expected_value="value"
        )
        execution_result = {"nodes": []}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "not found" in result.message

    def test_from_config(self):
        """Test creating from config."""
        config = {
            "node_name": "TestNode",
            "path": "result.value",
            "expected_value": 42,
        }
        assertion = N8nNodeOutput.from_config(config)

        assert assertion.node_name == "TestNode"
        assert assertion.path == "result.value"
        assert assertion.expected_value == 42


class TestN8nWorkflowDuration:
    """Tests for N8nWorkflowDuration assertion."""

    def test_duration_within_limit(self):
        """Test workflow duration within limit."""
        assertion = N8nWorkflowDuration(max_seconds=5.0)
        execution_result = {"duration_ms": 3000}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert "within" in result.message
        assert "3000" in result.actual

    def test_duration_exceeds_limit(self):
        """Test workflow duration exceeds limit."""
        assertion = N8nWorkflowDuration(max_seconds=2.0)
        execution_result = {"duration_ms": 5000}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "exceeds" in result.message
        assert "5000" in result.actual

    def test_duration_exactly_at_limit(self):
        """Test workflow duration exactly at limit."""
        assertion = N8nWorkflowDuration(max_seconds=3.0)
        execution_result = {"duration_ms": 3000}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True

    def test_missing_duration(self):
        """Test missing duration field defaults to infinity."""
        assertion = N8nWorkflowDuration(max_seconds=5.0)
        execution_result = {}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "inf" in result.actual

    def test_from_config(self):
        """Test creating from config."""
        config = {"max_seconds": 10.5}
        assertion = N8nWorkflowDuration.from_config(config)

        assert assertion.max_seconds == 10.5
        assert assertion.max_ms == 10500


class TestN8nAINodeTokens:
    """Tests for N8nAINodeTokens assertion."""

    def test_tokens_within_budget(self):
        """Test tokens within budget."""
        assertion = N8nAINodeTokens(node_name="GPT-4", max_tokens=1000)
        execution_result = {
            "nodes": [{"name": "GPT-4", "total_tokens": 500}]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert "500" in result.message
        assert "within" in result.message

    def test_tokens_exceed_budget(self):
        """Test tokens exceed budget."""
        assertion = N8nAINodeTokens(node_name="GPT-4", max_tokens=1000)
        execution_result = {
            "nodes": [{"name": "GPT-4", "total_tokens": 1500}]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "1500" in result.message
        assert "exceeds" in result.message

    def test_tokens_exactly_at_budget(self):
        """Test tokens exactly at budget."""
        assertion = N8nAINodeTokens(node_name="GPT-4", max_tokens=1000)
        execution_result = {
            "nodes": [{"name": "GPT-4", "total_tokens": 1000}]
        }

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True

    def test_missing_tokens_field(self):
        """Test missing total_tokens field defaults to 0."""
        assertion = N8nAINodeTokens(node_name="GPT-4", max_tokens=1000)
        execution_result = {"nodes": [{"name": "GPT-4"}]}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert result.actual == "0"

    def test_node_not_found(self):
        """Test AI node not found."""
        assertion = N8nAINodeTokens(node_name="GPT-4", max_tokens=1000)
        execution_result = {"nodes": [{"name": "OtherNode"}]}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "not found" in result.message

    def test_from_config(self):
        """Test creating from config."""
        config = {"node_name": "Claude", "max_tokens": 2000}
        assertion = N8nAINodeTokens.from_config(config)

        assert assertion.node_name == "Claude"
        assert assertion.max_tokens == 2000


class TestN8nWorkflowStatus:
    """Tests for N8nWorkflowStatus assertion."""

    def test_status_matches_success(self):
        """Test workflow status matches expected success."""
        assertion = N8nWorkflowStatus(expected_status="success")
        execution_result = {"status": "success"}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True
        assert "matches" in result.message
        assert result.expected == "success"
        assert result.actual == "success"

    def test_status_does_not_match(self):
        """Test workflow status doesn't match expected."""
        assertion = N8nWorkflowStatus(expected_status="success")
        execution_result = {"status": "error"}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert "does not match" in result.message
        assert result.expected == "success"
        assert result.actual == "error"

    def test_default_status(self):
        """Test default expected status is 'success'."""
        assertion = N8nWorkflowStatus()

        assert assertion.expected_status == "success"

    def test_custom_status(self):
        """Test custom expected status."""
        assertion = N8nWorkflowStatus(expected_status="running")
        execution_result = {"status": "running"}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is True

    def test_missing_status(self):
        """Test missing status field defaults to 'unknown'."""
        assertion = N8nWorkflowStatus(expected_status="success")
        execution_result = {}

        result = assertion.evaluate(execution_result, None, None)

        assert result.passed is False
        assert result.actual == "unknown"

    def test_from_config(self):
        """Test creating from config."""
        config = {"expected_status": "error"}
        assertion = N8nWorkflowStatus.from_config(config)

        assert assertion.expected_status == "error"

    def test_from_config_default(self):
        """Test creating from config with default."""
        config = {}
        assertion = N8nWorkflowStatus.from_config(config)

        assert assertion.expected_status == "success"


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""

    def test_node_completed(self):
        """Test node_completed factory."""
        assertion = node_completed("TestNode")

        assert isinstance(assertion, N8nNodeCompleted)
        assert assertion.node_name == "TestNode"

    def test_node_output(self):
        """Test node_output factory."""
        assertion = node_output("TestNode", "path.to.value", 42)

        assert isinstance(assertion, N8nNodeOutput)
        assert assertion.node_name == "TestNode"
        assert assertion.path == "path.to.value"
        assert assertion.expected_value == 42

    def test_duration_under(self):
        """Test duration_under factory."""
        assertion = duration_under(5.5)

        assert isinstance(assertion, N8nWorkflowDuration)
        assert assertion.max_seconds == 5.5

    def test_tokens_under(self):
        """Test tokens_under factory."""
        assertion = tokens_under("GPT-4", 1500)

        assert isinstance(assertion, N8nAINodeTokens)
        assert assertion.node_name == "GPT-4"
        assert assertion.max_tokens == 1500

    def test_workflow_completed(self):
        """Test workflow_completed factory."""
        assertion = workflow_completed()

        assert isinstance(assertion, N8nWorkflowStatus)
        assert assertion.expected_status == "success"

    def test_workflow_status(self):
        """Test workflow_status factory."""
        assertion = workflow_status("error")

        assert isinstance(assertion, N8nWorkflowStatus)
        assert assertion.expected_status == "error"


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_complete_workflow_evaluation(self):
        """Test evaluating a complete workflow with multiple assertions."""
        execution_result = {
            "status": "success",
            "duration_ms": 2500,
            "nodes": [
                {"name": "Fetch Data", "status": "success"},
                {
                    "name": "Process with AI",
                    "status": "success",
                    "total_tokens": 750,
                    "output": {"result": {"processed": True, "count": 10}},
                },
                {"name": "Send Notification", "status": "success"},
            ],
        }

        assertions = [
            workflow_completed(),
            duration_under(5.0),
            node_completed("Fetch Data"),
            node_completed("Process with AI"),
            tokens_under("Process with AI", 1000),
            node_output("Process with AI", "result.processed", True),
            node_output("Process with AI", "result.count", 10),
        ]

        results = [
            assertion.evaluate(execution_result, None, None)
            for assertion in assertions
        ]

        # All assertions should pass
        assert all(r.passed for r in results)

    def test_failed_workflow_evaluation(self):
        """Test evaluating a failed workflow."""
        execution_result = {
            "status": "error",
            "duration_ms": 8500,
            "nodes": [
                {"name": "Fetch Data", "status": "success"},
                {"name": "Process with AI", "status": "error", "total_tokens": 1500},
            ],
        }

        assertions = [
            workflow_completed(),  # Should fail (status is error)
            duration_under(5.0),  # Should fail (took 8.5s)
            node_completed("Process with AI"),  # Should fail (node errored)
            tokens_under("Process with AI", 1000),  # Should fail (used 1500)
        ]

        results = [
            assertion.evaluate(execution_result, None, None)
            for assertion in assertions
        ]

        # All assertions should fail
        assert all(not r.passed for r in results)
