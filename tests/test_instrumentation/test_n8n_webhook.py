"""Tests for n8n webhook handler."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from prela.core.span import SpanStatus, SpanType
from prela.core.tracer import Tracer
from prela.exporters.console import ConsoleExporter
from prela.instrumentation.n8n.webhook import (
    N8N_AI_NODE_TYPES,
    N8nWebhookHandler,
    N8nWebhookPayload,
    extract_ai_attributes,
    get_ai_node_category,
    is_ai_node,
    map_n8n_span_type_to_prela,
    parse_n8n_webhook,
)
from prela.instrumentation.n8n.models import N8nSpanType


# Sample n8n webhook payloads


def get_basic_workflow_payload():
    """Basic n8n webhook payload for a non-AI workflow."""
    return {
        "workflow": {
            "id": "wf_123",
            "name": "Customer Onboarding",
            "active": True,
        },
        "execution": {
            "id": "exec_456",
            "mode": "trigger",
            "startedAt": "2025-01-27T10:30:00.000Z",
        },
        "node": {
            "name": "Send Email",
            "type": "n8n-nodes-base.emailSend",
            "parameters": {
                "toEmail": "user@example.com",
                "subject": "Welcome!",
            },
        },
        "data": [
            {
                "json": {
                    "email": "user@example.com",
                    "status": "sent",
                }
            }
        ],
    }


def get_openai_llm_payload():
    """n8n webhook payload for OpenAI LLM node."""
    return {
        "workflow": {
            "id": "wf_ai_001",
            "name": "AI Content Generator",
            "active": True,
        },
        "execution": {
            "id": "exec_ai_789",
            "mode": "manual",
            "startedAt": "2025-01-27T10:35:00.000Z",
        },
        "node": {
            "name": "OpenAI Chat",
            "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
            "parameters": {
                "model": "gpt-4",
                "temperature": 0.7,
                "systemMessage": "You are a helpful assistant.",
            },
        },
        "data": [
            {
                "json": {
                    "response": "Here is the generated content...",
                    "usage": {
                        "prompt_tokens": 150,
                        "completion_tokens": 89,
                        "total_tokens": 239,
                    },
                }
            }
        ],
    }


def get_anthropic_llm_payload():
    """n8n webhook payload for Anthropic Claude node."""
    return {
        "workflow": {
            "id": "wf_ai_002",
            "name": "AI Analysis Pipeline",
            "active": True,
        },
        "execution": {
            "id": "exec_ai_890",
            "mode": "webhook",
            "startedAt": "2025-01-27T10:40:00.000Z",
        },
        "node": {
            "name": "Claude Analyzer",
            "type": "@n8n/n8n-nodes-langchain.lmChatAnthropic",
            "parameters": {
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
                "systemMessage": "Analyze the provided data.",
            },
        },
        "data": [
            {
                "json": {
                    "response": {
                        "content": "Analysis complete. Key findings: ...",
                    },
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 200,
                        "total_tokens": 700,
                    },
                }
            }
        ],
    }


def get_langchain_agent_payload():
    """n8n webhook payload for LangChain agent node."""
    return {
        "workflow": {
            "id": "wf_agent_003",
            "name": "Research Agent Workflow",
            "active": True,
        },
        "execution": {
            "id": "exec_agent_999",
            "mode": "manual",
            "startedAt": "2025-01-27T11:00:00.000Z",
        },
        "node": {
            "name": "Research Agent",
            "type": "n8n-nodes-langchain.agent",
            "parameters": {
                "model": "gpt-4",
                "systemMessage": "You are a research assistant.",
            },
        },
        "data": [
            {
                "json": {
                    "response": "Based on my research...",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "name": "web_search",
                            "arguments": {"query": "AI trends 2025"},
                        },
                        {
                            "id": "call_456",
                            "name": "summarize",
                            "arguments": {"text": "..."},
                        },
                    ],
                    "usage": {
                        "prompt_tokens": 800,
                        "completion_tokens": 350,
                        "total_tokens": 1150,
                    },
                }
            }
        ],
    }


def get_vector_store_payload():
    """n8n webhook payload for vector store retrieval node."""
    return {
        "workflow": {
            "id": "wf_rag_004",
            "name": "RAG Pipeline",
            "active": True,
        },
        "execution": {
            "id": "exec_rag_111",
            "mode": "trigger",
            "startedAt": "2025-01-27T11:15:00.000Z",
        },
        "node": {
            "name": "Qdrant Search",
            "type": "n8n-nodes-langchain.vectorStoreQdrant",
            "parameters": {
                "topK": 5,
            },
        },
        "data": [
            {
                "json": {
                    "query": "What is machine learning?",
                    "documents": [
                        {
                            "content": "Machine learning is...",
                            "score": 0.95,
                            "metadata": {"source": "textbook.pdf"},
                        },
                        {
                            "content": "ML is a subset of AI...",
                            "score": 0.89,
                            "metadata": {"source": "article.md"},
                        },
                    ],
                }
            }
        ],
    }


# Tests


class TestN8nWebhookPayload:
    """Test N8nWebhookPayload model."""

    def test_parse_basic_payload(self):
        """Test parsing basic webhook payload."""
        payload_dict = get_basic_workflow_payload()
        payload = N8nWebhookPayload(**payload_dict)

        assert payload.workflow["id"] == "wf_123"
        assert payload.workflow["name"] == "Customer Onboarding"
        assert payload.execution["id"] == "exec_456"
        assert payload.node["name"] == "Send Email"
        assert len(payload.data) == 1

    def test_parse_ai_payload(self):
        """Test parsing AI node payload."""
        payload_dict = get_openai_llm_payload()
        payload = N8nWebhookPayload(**payload_dict)

        assert payload.node["type"] == "@n8n/n8n-nodes-langchain.lmChatOpenAi"
        assert payload.node["parameters"]["model"] == "gpt-4"
        assert "usage" in payload.data[0]["json"]


class TestAINodeDetection:
    """Test AI node type detection."""

    def test_is_ai_node_openai(self):
        """Test detection of OpenAI node."""
        assert is_ai_node("@n8n/n8n-nodes-langchain.lmChatOpenAi") is True

    def test_is_ai_node_anthropic(self):
        """Test detection of Anthropic node."""
        assert is_ai_node("@n8n/n8n-nodes-langchain.lmChatAnthropic") is True

    def test_is_ai_node_agent(self):
        """Test detection of agent node."""
        assert is_ai_node("n8n-nodes-langchain.agent") is True

    def test_is_ai_node_vector_store(self):
        """Test detection of vector store node."""
        assert is_ai_node("n8n-nodes-langchain.vectorStoreQdrant") is True

    def test_is_not_ai_node(self):
        """Test non-AI node detection."""
        assert is_ai_node("n8n-nodes-base.emailSend") is False

    def test_get_ai_category_llm(self):
        """Test getting LLM category."""
        category = get_ai_node_category("@n8n/n8n-nodes-langchain.lmChatOpenAi")
        assert category == "llm"

    def test_get_ai_category_agent(self):
        """Test getting agent category."""
        category = get_ai_node_category("n8n-nodes-langchain.agent")
        assert category == "ai_agent"

    def test_get_ai_category_retrieval(self):
        """Test getting retrieval category."""
        category = get_ai_node_category("n8n-nodes-langchain.vectorStoreQdrant")
        assert category == "retrieval"

    def test_get_ai_category_none(self):
        """Test getting category for non-AI node."""
        category = get_ai_node_category("n8n-nodes-base.emailSend")
        assert category is None


class TestSpanTypeMapping:
    """Test span type mapping."""

    def test_map_workflow_to_agent(self):
        """Test workflow maps to AGENT."""
        span_type = map_n8n_span_type_to_prela(N8nSpanType.WORKFLOW)
        assert span_type == SpanType.AGENT

    def test_map_llm_to_llm(self):
        """Test LLM maps to LLM."""
        span_type = map_n8n_span_type_to_prela(N8nSpanType.LLM)
        assert span_type == SpanType.LLM

    def test_map_retrieval_to_retrieval(self):
        """Test retrieval maps to RETRIEVAL."""
        span_type = map_n8n_span_type_to_prela(N8nSpanType.RETRIEVAL)
        assert span_type == SpanType.RETRIEVAL

    def test_map_tool_to_tool(self):
        """Test tool maps to TOOL."""
        span_type = map_n8n_span_type_to_prela(N8nSpanType.TOOL)
        assert span_type == SpanType.TOOL


class TestExtractAIAttributes:
    """Test AI attribute extraction."""

    def test_extract_openai_attributes(self):
        """Test extracting OpenAI attributes."""
        node_type = "@n8n/n8n-nodes-langchain.lmChatOpenAi"
        node_params = {
            "model": "gpt-4",
            "temperature": 0.7,
            "systemMessage": "You are helpful.",
        }
        items = [
            {
                "json": {
                    "response": "Hello!",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            }
        ]

        attrs = extract_ai_attributes(node_type, node_params, items)

        assert attrs["model"] == "gpt-4"
        assert attrs["temperature"] == 0.7
        assert attrs["system_prompt"] == "You are helpful."
        assert attrs["provider"] == "openai"
        assert attrs["prompt_tokens"] == 10
        assert attrs["completion_tokens"] == 5
        assert attrs["total_tokens"] == 15
        assert "Hello!" in attrs["response_content"]

    def test_extract_anthropic_attributes(self):
        """Test extracting Anthropic attributes."""
        node_type = "@n8n/n8n-nodes-langchain.lmChatAnthropic"
        node_params = {"model": "claude-3-opus", "temperature": 0.3}
        items = [
            {
                "json": {
                    "response": {"content": "Analysis complete."},
                    "usage": {"total_tokens": 500},
                }
            }
        ]

        attrs = extract_ai_attributes(node_type, node_params, items)

        assert attrs["model"] == "claude-3-opus"
        assert attrs["provider"] == "anthropic"
        assert attrs["total_tokens"] == 500
        assert "Analysis complete." in attrs["response_content"]

    def test_extract_tool_calls(self):
        """Test extracting tool calls."""
        items = [
            {
                "json": {
                    "tool_calls": [
                        {"name": "search", "arguments": {"query": "AI"}},
                    ]
                }
            }
        ]

        attrs = extract_ai_attributes("n8n-nodes-langchain.agent", {}, items)

        assert "tool_calls" in attrs
        assert len(attrs["tool_calls"]) == 1
        assert attrs["tool_calls"][0]["name"] == "search"

    def test_extract_retrieval_attributes(self):
        """Test extracting retrieval attributes."""
        items = [
            {
                "json": {
                    "query": "What is AI?",
                    "documents": [
                        {"content": "AI is...", "score": 0.95},
                        {"content": "Artificial intelligence...", "score": 0.88},
                    ],
                }
            }
        ]

        attrs = extract_ai_attributes(
            "n8n-nodes-langchain.vectorStoreQdrant", {}, items
        )

        assert attrs["retrieval_query"] == "What is AI?"
        assert "retrieved_documents" in attrs
        assert len(attrs["retrieved_documents"]) == 2


class TestParseN8nWebhook:
    """Test webhook payload parsing."""

    def test_parse_basic_workflow(self):
        """Test parsing basic workflow payload."""
        payload = get_basic_workflow_payload()
        spans = parse_n8n_webhook(payload)

        assert len(spans) == 2  # Workflow + node span

        # Check workflow span
        workflow_span = spans[0]
        assert "n8n.workflow.Customer Onboarding" in workflow_span.name
        assert workflow_span.span_type == SpanType.AGENT
        assert workflow_span.parent_span_id is None
        assert workflow_span.attributes["n8n.workflow_id"] == "wf_123"
        assert workflow_span.attributes["n8n.execution_id"] == "exec_456"

        # Check node span
        node_span = spans[1]
        assert "n8n.node.Send Email" in node_span.name
        assert node_span.parent_span_id == workflow_span.span_id
        assert node_span.attributes["n8n.node_type"] == "n8n-nodes-base.emailSend"
        assert node_span.status == SpanStatus.SUCCESS

    def test_parse_openai_llm_payload(self):
        """Test parsing OpenAI LLM payload."""
        payload = get_openai_llm_payload()
        spans = parse_n8n_webhook(payload)

        assert len(spans) == 2

        # Check node span has AI attributes
        node_span = spans[1]
        assert node_span.span_type == SpanType.LLM
        assert node_span.attributes["n8n.ai_category"] == "llm"
        assert node_span.attributes["model"] == "gpt-4"
        assert node_span.attributes["provider"] == "openai"
        assert node_span.attributes["temperature"] == 0.7
        assert node_span.attributes["prompt_tokens"] == 150
        assert node_span.attributes["completion_tokens"] == 89
        assert node_span.attributes["total_tokens"] == 239

    def test_parse_anthropic_llm_payload(self):
        """Test parsing Anthropic LLM payload."""
        payload = get_anthropic_llm_payload()
        spans = parse_n8n_webhook(payload)

        node_span = spans[1]
        assert node_span.span_type == SpanType.LLM
        assert node_span.attributes["model"] == "claude-sonnet-4-20250514"
        assert node_span.attributes["provider"] == "anthropic"

    def test_parse_agent_payload(self):
        """Test parsing agent payload."""
        payload = get_langchain_agent_payload()
        spans = parse_n8n_webhook(payload)

        node_span = spans[1]
        assert node_span.span_type == SpanType.AGENT
        assert node_span.attributes["n8n.ai_category"] == "ai_agent"
        assert "tool_calls" in node_span.attributes

    def test_parse_vector_store_payload(self):
        """Test parsing vector store payload."""
        payload = get_vector_store_payload()
        spans = parse_n8n_webhook(payload)

        node_span = spans[1]
        assert node_span.span_type == SpanType.RETRIEVAL
        assert node_span.attributes["n8n.ai_category"] == "retrieval"
        assert node_span.attributes["retrieval_query"] == "What is machine learning?"

    def test_parse_invalid_payload(self):
        """Test parsing invalid payload."""
        payload = {"invalid": "data"}
        spans = parse_n8n_webhook(payload)

        assert len(spans) == 0  # Should return empty list on error

    def test_trace_id_generation(self):
        """Test trace ID is properly generated."""
        payload = get_basic_workflow_payload()
        spans = parse_n8n_webhook(payload)

        workflow_span = spans[0]
        node_span = spans[1]

        # Both spans should have same trace_id
        assert workflow_span.trace_id == node_span.trace_id
        assert workflow_span.trace_id == "n8n-exec_456"

    def test_span_hierarchy(self):
        """Test span parent-child hierarchy."""
        payload = get_openai_llm_payload()
        spans = parse_n8n_webhook(payload)

        workflow_span = spans[0]
        node_span = spans[1]

        # Node span should be child of workflow span
        assert node_span.parent_span_id == workflow_span.span_id
        assert workflow_span.parent_span_id is None


class TestN8nWebhookHandler:
    """Test N8nWebhookHandler class."""

    def test_handler_initialization(self):
        """Test handler initialization."""
        tracer = Tracer(service_name="test", exporter=ConsoleExporter())
        handler = N8nWebhookHandler(tracer, port=8888)

        assert handler.tracer == tracer
        assert handler.port == 8888
        assert handler.host == "0.0.0.0"

    def test_handler_custom_host(self):
        """Test handler with custom host."""
        tracer = Tracer(service_name="test", exporter=ConsoleExporter())
        handler = N8nWebhookHandler(tracer, port=8888, host="127.0.0.1")

        assert handler.host == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_handle_webhook_success(self):
        """Test successful webhook handling."""
        tracer = Tracer(service_name="test", exporter=ConsoleExporter())
        handler = N8nWebhookHandler(tracer)

        # Mock request object
        class MockRequest:
            async def json(self):
                return get_basic_workflow_payload()

        result = await handler.handle_webhook(MockRequest())

        assert result["status"] == "success"
        assert "Created 2 spans" in result["message"]
        assert result["trace_id"].startswith("n8n-")

    @pytest.mark.asyncio
    async def test_handle_webhook_error(self):
        """Test webhook handling with error."""
        tracer = Tracer(service_name="test", exporter=ConsoleExporter())
        handler = N8nWebhookHandler(tracer)

        # Mock request with invalid payload
        class MockRequest:
            async def json(self):
                return {"invalid": "payload"}

        result = await handler.handle_webhook(MockRequest())

        # Should still return success (parse returns empty list, no crash)
        assert result["status"] == "success"
        assert "Created 0 spans" in result["message"]


class TestN8nAINodeTypes:
    """Test N8N_AI_NODE_TYPES registry."""

    def test_registry_has_openai(self):
        """Test registry includes OpenAI nodes."""
        assert "@n8n/n8n-nodes-langchain.lmChatOpenAi" in N8N_AI_NODE_TYPES

    def test_registry_has_anthropic(self):
        """Test registry includes Anthropic nodes."""
        assert "@n8n/n8n-nodes-langchain.lmChatAnthropic" in N8N_AI_NODE_TYPES

    def test_registry_has_agents(self):
        """Test registry includes agent nodes."""
        assert "n8n-nodes-langchain.agent" in N8N_AI_NODE_TYPES

    def test_registry_has_vector_stores(self):
        """Test registry includes vector store nodes."""
        assert "n8n-nodes-langchain.vectorStoreQdrant" in N8N_AI_NODE_TYPES
        assert "n8n-nodes-langchain.vectorStorePinecone" in N8N_AI_NODE_TYPES

    def test_registry_has_memory(self):
        """Test registry includes memory nodes."""
        assert "n8n-nodes-langchain.memoryBufferWindow" in N8N_AI_NODE_TYPES

    def test_registry_has_tools(self):
        """Test registry includes tool nodes."""
        assert "n8n-nodes-langchain.toolCalculator" in N8N_AI_NODE_TYPES
