"""Tests for n8n workflow evaluation runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prela.evals.assertions import ContainsAssertion
from prela.evals.n8n import (
    N8nEvalCase,
    N8nWorkflowEvalConfig,
    N8nWorkflowEvalRunner,
    eval_n8n_workflow,
)
from prela.evals.suite import EvalSuite


class TestN8nWorkflowEvalConfig:
    """Tests for N8nWorkflowEvalConfig."""

    def test_default_config(self):
        """Test config with defaults."""
        config = N8nWorkflowEvalConfig(workflow_id="test-123")

        assert config.workflow_id == "test-123"
        assert config.n8n_base_url == "http://localhost:5678"
        assert config.n8n_api_key is None
        assert config.timeout_seconds == 120
        assert config.capture_traces is True

    def test_custom_config(self):
        """Test config with custom values."""
        config = N8nWorkflowEvalConfig(
            workflow_id="workflow-abc",
            n8n_base_url="https://n8n.example.com",
            n8n_api_key="secret-key",
            timeout_seconds=60,
            capture_traces=False,
        )

        assert config.workflow_id == "workflow-abc"
        assert config.n8n_base_url == "https://n8n.example.com"
        assert config.n8n_api_key == "secret-key"
        assert config.timeout_seconds == 60
        assert config.capture_traces is False


class TestN8nEvalCase:
    """Tests for N8nEvalCase."""

    def test_basic_case(self):
        """Test creating a basic n8n eval case."""
        case = N8nEvalCase(
            id="test_1",
            name="Test workflow",
            trigger_data={"message": "Hello"},
        )

        assert case.id == "test_1"
        assert case.name == "Test workflow"
        assert case.trigger_data == {"message": "Hello"}
        assert case.node_assertions is None
        assert case.workflow_assertions is None
        assert case.expected_output is None

    def test_case_with_assertions(self):
        """Test case with node and workflow assertions."""
        node_assertions = {
            "Node1": [ContainsAssertion(text="test")],
            "Node2": [ContainsAssertion(text="result")],
        }
        workflow_assertions = [ContainsAssertion(text="success")]

        case = N8nEvalCase(
            id="test_2",
            name="Test with assertions",
            trigger_data={"data": "value"},
            node_assertions=node_assertions,
            workflow_assertions=workflow_assertions,
            expected_output={"status": "complete"},
        )

        assert case.node_assertions == node_assertions
        assert case.workflow_assertions == workflow_assertions
        assert case.expected_output == {"status": "complete"}


class TestN8nWorkflowEvalRunner:
    """Tests for N8nWorkflowEvalRunner."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return N8nWorkflowEvalConfig(
            workflow_id="test-workflow",
            n8n_base_url="http://localhost:5678",
            timeout_seconds=30,
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock HTTP client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def runner(self, config, mock_client):
        """Create runner with mocked client."""
        runner = N8nWorkflowEvalRunner(config)
        runner.client = mock_client
        return runner

    def test_init(self, config):
        """Test runner initialization."""
        runner = N8nWorkflowEvalRunner(config)

        assert runner.config == config
        assert runner.tracer is None
        assert runner.client is not None

    def test_init_with_tracer(self, config):
        """Test runner initialization with tracer."""
        tracer = MagicMock()
        runner = N8nWorkflowEvalRunner(config, tracer=tracer)

        assert runner.tracer == tracer

    @pytest.mark.asyncio
    async def test_trigger_workflow(self, runner, mock_client):
        """Test triggering workflow."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"executionId": "exec-123"}}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        execution_id = await runner._trigger_workflow({"test": "data"})

        assert execution_id == "exec-123"
        mock_client.post.assert_called_once_with(
            "/api/v1/workflows/test-workflow/execute",
            json={"data": {"test": "data"}},
        )

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, runner, mock_client):
        """Test waiting for successful completion."""
        # Mock response sequence: running -> running -> success
        responses = [
            {"data": {"status": "running", "id": "exec-123"}},
            {"data": {"status": "running", "id": "exec-123"}},
            {"data": {"status": "success", "id": "exec-123", "output": "result"}},
        ]

        mock_responses = []
        for resp_data in responses:
            mock_resp = MagicMock()
            mock_resp.json.return_value = resp_data
            mock_resp.raise_for_status = MagicMock()
            mock_responses.append(mock_resp)

        mock_client.get = AsyncMock(side_effect=mock_responses)

        result = await runner._wait_for_completion("exec-123")

        assert result["status"] == "success"
        assert result["output"] == "result"
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, runner, mock_client):
        """Test timeout when waiting for completion."""
        # Set very short timeout
        runner.config.timeout_seconds = 0.1

        # Mock response that never completes
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"status": "running"}}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(TimeoutError, match="timed out"):
            await runner._wait_for_completion("exec-123")

    def test_get_node_data_with_run_data(self, runner):
        """Test extracting node data from execution result (runData structure)."""
        execution_result = {
            "resultData": {
                "runData": {
                    "Node1": {"output": "test"},
                    "Node2": {"output": "result"},
                }
            }
        }

        node_data = runner._get_node_data(execution_result, "Node1")

        assert node_data == {"output": "test"}

    def test_get_node_data_with_nodes_array(self, runner):
        """Test extracting node data from nodes array."""
        execution_result = {
            "nodes": [
                {"name": "Node1", "output": "test"},
                {"name": "Node2", "output": "result"},
            ]
        }

        node_data = runner._get_node_data(execution_result, "Node2")

        assert node_data == {"name": "Node2", "output": "result"}

    def test_get_node_data_not_found(self, runner):
        """Test extracting non-existent node data."""
        execution_result = {"nodes": []}

        node_data = runner._get_node_data(execution_result, "NonExistent")

        assert node_data is None

    @pytest.mark.asyncio
    async def test_run_case_success(self, runner, mock_client):
        """Test running a case successfully."""
        case = N8nEvalCase(
            id="test_1",
            name="Test case",
            trigger_data={"input": "data"},
        )

        # Mock trigger response
        trigger_resp = MagicMock()
        trigger_resp.json.return_value = {"data": {"executionId": "exec-123"}}
        trigger_resp.raise_for_status = MagicMock()

        # Mock completion response
        completion_resp = MagicMock()
        completion_resp.json.return_value = {
            "data": {"status": "success", "output": "result"}
        }
        completion_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(return_value=trigger_resp)
        mock_client.get = AsyncMock(return_value=completion_resp)

        result = await runner.run_case(case)

        assert result["passed"] is True
        assert result["execution_id"] == "exec-123"
        assert result["status"] == "success"
        assert result["duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_run_case_with_expected_output(self, runner, mock_client):
        """Test case with expected output validation."""
        case = N8nEvalCase(
            id="test_2",
            name="Output test",
            trigger_data={"input": "data"},
            expected_output={"result": "expected"},
        )

        # Mock responses
        trigger_resp = MagicMock()
        trigger_resp.json.return_value = {"data": {"executionId": "exec-123"}}
        trigger_resp.raise_for_status = MagicMock()

        completion_resp = MagicMock()
        completion_resp.json.return_value = {
            "data": {"status": "success", "output": {"result": "actual"}}
        }
        completion_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(return_value=trigger_resp)
        mock_client.get = AsyncMock(return_value=completion_resp)

        result = await runner.run_case(case)

        assert result["passed"] is False  # Output doesn't match
        assert "output_mismatch" in result
        assert result["output_mismatch"]["expected"] == {"result": "expected"}
        assert result["output_mismatch"]["actual"] == {"result": "actual"}

    @pytest.mark.asyncio
    async def test_run_case_with_node_assertions(self, runner, mock_client):
        """Test case with node-level assertions."""
        case = N8nEvalCase(
            id="test_3",
            name="Node assertion test",
            trigger_data={"input": "data"},
            node_assertions={
                "Node1": [ContainsAssertion(text="test")],
            },
        )

        # Mock responses
        trigger_resp = MagicMock()
        trigger_resp.json.return_value = {"data": {"executionId": "exec-123"}}
        trigger_resp.raise_for_status = MagicMock()

        completion_resp = MagicMock()
        completion_resp.json.return_value = {
            "data": {
                "status": "success",
                "resultData": {
                    "runData": {
                        "Node1": "test result",  # Node data is the actual output string
                    }
                },
            }
        }
        completion_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(return_value=trigger_resp)
        mock_client.get = AsyncMock(return_value=completion_resp)

        result = await runner.run_case(case)

        assert result["passed"] is True
        assert "Node1" in result["node_results"]
        assert len(result["node_results"]["Node1"]) == 1
        assert result["node_results"]["Node1"][0].passed is True

    @pytest.mark.asyncio
    async def test_run_case_error(self, runner, mock_client):
        """Test case that encounters an error."""
        case = N8nEvalCase(
            id="test_4",
            name="Error test",
            trigger_data={"input": "data"},
        )

        # Mock error response
        mock_client.post.side_effect = Exception("API error")

        result = await runner.run_case(case)

        assert result["passed"] is False
        assert result["status"] == "error"
        assert "error" in result
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_run_suite(self, runner, mock_client):
        """Test running a suite of cases."""
        cases = [
            N8nEvalCase(id="test_1", name="Case 1", trigger_data={"data": "1"}),
            N8nEvalCase(id="test_2", name="Case 2", trigger_data={"data": "2"}),
        ]
        suite = EvalSuite(name="Test Suite", cases=cases)

        # Mock responses for both cases
        trigger_resp = MagicMock()
        trigger_resp.json.return_value = {"data": {"executionId": "exec-123"}}
        trigger_resp.raise_for_status = MagicMock()

        completion_resp = MagicMock()
        completion_resp.json.return_value = {"data": {"status": "success"}}
        completion_resp.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(return_value=trigger_resp)
        mock_client.get = AsyncMock(return_value=completion_resp)

        result = await runner.run_suite(suite)

        assert result["suite_name"] == "Test Suite"
        assert result["total"] == 2
        assert result["passed"] == 2
        assert result["failed"] == 0
        assert len(result["cases"]) == 2

    @pytest.mark.asyncio
    async def test_close(self, runner, mock_client):
        """Test closing the runner."""
        await runner.close()
        mock_client.aclose.assert_called_once()


class TestEvalN8nWorkflow:
    """Tests for eval_n8n_workflow convenience function."""

    @pytest.mark.asyncio
    async def test_eval_n8n_workflow(self):
        """Test convenience function."""
        cases = [
            N8nEvalCase(id="test_1", name="Test", trigger_data={"data": "value"}),
        ]

        with patch(
            "prela.evals.n8n.runner.N8nWorkflowEvalRunner"
        ) as mock_runner_class:
            # Mock runner instance
            mock_runner = AsyncMock()
            mock_runner.run_suite.return_value = {
                "suite_name": "n8n-workflow-123",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "cases": [],
            }
            mock_runner_class.return_value = mock_runner

            result = await eval_n8n_workflow(
                workflow_id="workflow-123",
                test_cases=cases,
                n8n_url="http://localhost:5678",
                n8n_api_key="test-key",
            )

            assert result["suite_name"] == "n8n-workflow-123"
            assert result["total"] == 1
            assert result["passed"] == 1

            # Verify runner was created with correct config
            mock_runner_class.assert_called_once()
            config = mock_runner_class.call_args[0][0]
            assert config.workflow_id == "workflow-123"
            assert config.n8n_base_url == "http://localhost:5678"
            assert config.n8n_api_key == "test-key"

            # Verify close was called
            mock_runner.close.assert_called_once()
