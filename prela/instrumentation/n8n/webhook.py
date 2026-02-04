"""
n8n webhook handler for receiving workflow execution traces via HTTP.

This module handles traces sent from n8n workflows via HTTP webhook nodes,
parsing the payload and converting it into Prela spans.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from prela.core.clock import now
from prela.core.span import Span, SpanStatus, SpanType
from prela.core.tracer import Tracer
from prela.instrumentation.n8n.models import (
    N8nAINodeExecution,
    N8nNodeExecution,
    N8nSpanType,
    N8nWorkflowExecution,
)

logger = logging.getLogger(__name__)


# Mapping of n8n node types to AI categories
N8N_AI_NODE_TYPES = {
    # LangChain Agent nodes
    "n8n-nodes-langchain.agent": "ai_agent",
    "n8n-nodes-langchain.agentExecutor": "ai_agent",
    # LangChain Chain nodes
    "n8n-nodes-langchain.chainLlm": "llm_chain",
    "n8n-nodes-langchain.chainSummarization": "llm_chain",
    "n8n-nodes-langchain.chainRetrievalQa": "llm_chain",
    # LLM Chat nodes (OpenAI)
    "@n8n/n8n-nodes-langchain.lmChatOpenAi": "llm",
    "n8n-nodes-langchain.lmChatOpenAi": "llm",
    # LLM Chat nodes (Anthropic)
    "@n8n/n8n-nodes-langchain.lmChatAnthropic": "llm",
    "n8n-nodes-langchain.lmChatAnthropic": "llm",
    # LLM Chat nodes (Ollama)
    "@n8n/n8n-nodes-langchain.lmChatOllama": "llm",
    "n8n-nodes-langchain.lmChatOllama": "llm",
    # LLM Chat nodes (Other providers)
    "n8n-nodes-langchain.lmChatAzureOpenAi": "llm",
    "n8n-nodes-langchain.lmChatMistralCloud": "llm",
    "n8n-nodes-langchain.lmChatGoogleVertex": "llm",
    # Vector Store nodes
    "n8n-nodes-langchain.vectorStoreQdrant": "retrieval",
    "n8n-nodes-langchain.vectorStorePinecone": "retrieval",
    "n8n-nodes-langchain.vectorStoreSupabase": "retrieval",
    "n8n-nodes-langchain.vectorStoreInMemory": "retrieval",
    # Memory nodes
    "n8n-nodes-langchain.memoryBufferWindow": "memory",
    "n8n-nodes-langchain.memoryBuffer": "memory",
    "n8n-nodes-langchain.memoryChatSummary": "memory",
    # Tool nodes
    "n8n-nodes-langchain.toolCalculator": "tool",
    "n8n-nodes-langchain.toolCode": "tool",
    "n8n-nodes-langchain.toolHttpRequest": "tool",
    "n8n-nodes-langchain.toolWorkflow": "tool",
}


class N8nWebhookPayload(BaseModel):
    """
    Represents the payload received from an n8n webhook.

    n8n webhooks send execution data in a specific format with workflow,
    execution, and node metadata, along with the actual data items.
    """

    workflow: dict = Field(..., description="Workflow metadata (id, name, active)")
    execution: dict = Field(
        ..., description="Execution metadata (id, mode, startedAt)"
    )
    node: dict = Field(..., description="Node metadata (name, type, parameters)")
    data: list[dict] = Field(
        default_factory=list,
        description="n8n item array (list of {json: {...}} objects)",
    )
    metadata: Optional[dict] = Field(
        None, description="Additional metadata from n8n"
    )


def is_ai_node(node_type: str) -> bool:
    """Check if a node type is AI-related."""
    return node_type in N8N_AI_NODE_TYPES


def get_ai_node_category(node_type: str) -> Optional[str]:
    """Get the AI category for a node type."""
    return N8N_AI_NODE_TYPES.get(node_type)


def map_n8n_span_type_to_prela(n8n_type: N8nSpanType) -> SpanType:
    """Map n8n span types to Prela SpanType enum."""
    mapping = {
        N8nSpanType.WORKFLOW: SpanType.AGENT,
        N8nSpanType.NODE: SpanType.CUSTOM,
        N8nSpanType.AI_AGENT: SpanType.AGENT,
        N8nSpanType.LLM: SpanType.LLM,
        N8nSpanType.TOOL: SpanType.TOOL,
        N8nSpanType.RETRIEVAL: SpanType.RETRIEVAL,
        N8nSpanType.MEMORY: SpanType.CUSTOM,
    }
    return mapping.get(n8n_type, SpanType.CUSTOM)


def extract_ai_attributes(
    node_type: str, node_params: dict, items: list[dict]
) -> dict[str, Any]:
    """
    Extract AI-specific attributes from node parameters and output items.

    Args:
        node_type: The n8n node type
        node_params: Node parameters/configuration
        items: Output items from the node

    Returns:
        Dictionary of AI-specific attributes (model, tokens, prompts, etc.)
    """
    attrs: dict[str, Any] = {}

    try:
        # Extract model information
        if "model" in node_params:
            attrs["model"] = node_params["model"]
        elif "modelName" in node_params:
            attrs["model"] = node_params["modelName"]

        # Extract temperature
        if "temperature" in node_params:
            attrs["temperature"] = float(node_params["temperature"])

        # Extract system prompt
        if "systemMessage" in node_params:
            attrs["system_prompt"] = str(node_params["systemMessage"])[:500]

        # Determine provider from node type
        node_lower = node_type.lower()
        if "openai" in node_lower:
            attrs["provider"] = "openai"
        elif "anthropic" in node_lower:
            attrs["provider"] = "anthropic"
        elif "ollama" in node_lower:
            attrs["provider"] = "ollama"
        elif "mistral" in node_lower:
            attrs["provider"] = "mistral"
        elif "vertex" in node_lower or "google" in node_lower:
            attrs["provider"] = "google"

        # Extract token usage from items if available
        for item in items:
            json_data = item.get("json", {})

            # OpenAI/Anthropic response format
            if "usage" in json_data:
                usage = json_data["usage"]
                if "prompt_tokens" in usage:
                    attrs["prompt_tokens"] = usage["prompt_tokens"]
                if "completion_tokens" in usage:
                    attrs["completion_tokens"] = usage["completion_tokens"]
                if "total_tokens" in usage:
                    attrs["total_tokens"] = usage["total_tokens"]

            # Extract response content
            if "response" in json_data:
                response = json_data["response"]
                if isinstance(response, str):
                    attrs["response_content"] = response[:500]
                elif isinstance(response, dict):
                    if "text" in response:
                        attrs["response_content"] = str(response["text"])[:500]
                    elif "content" in response:
                        attrs["response_content"] = str(response["content"])[:500]

            # Extract tool calls
            if "tool_calls" in json_data:
                attrs["tool_calls"] = json_data["tool_calls"]
            elif "function_call" in json_data:
                attrs["tool_calls"] = [json_data["function_call"]]

            # Extract retrieval query (for vector store nodes)
            if "query" in json_data:
                attrs["retrieval_query"] = str(json_data["query"])[:200]

            # Extract retrieved documents
            if "documents" in json_data:
                docs = json_data["documents"]
                if isinstance(docs, list):
                    attrs["retrieved_documents"] = docs[:5]  # Limit to 5 docs

    except Exception as e:
        logger.debug(f"Error extracting AI attributes: {e}")

    return attrs


def parse_n8n_webhook(payload: dict) -> list[Span]:
    """
    Convert n8n webhook payload into Prela spans.

    This function creates a hierarchy of spans:
    1. Workflow-level span (parent)
    2. Node-level span(s) (children)

    Args:
        payload: Raw webhook payload from n8n

    Returns:
        List of Span objects representing the execution
    """
    try:
        webhook_data = N8nWebhookPayload(**payload)
    except Exception as e:
        logger.error(f"Failed to parse n8n webhook payload: {e}")
        return []

    spans: list[Span] = []

    # Extract workflow metadata
    workflow_id = webhook_data.workflow.get("id", "unknown")
    workflow_name = webhook_data.workflow.get("name", "Unknown Workflow")
    execution_id = webhook_data.execution.get("id", "unknown")
    execution_mode = webhook_data.execution.get("mode", "manual")

    # Parse timestamps
    started_at_str = webhook_data.execution.get("startedAt")
    started_at = (
        datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        if started_at_str
        else now()
    )

    # Generate trace_id from execution_id
    trace_id = f"n8n-{execution_id}"

    # Create workflow-level span
    workflow_span = Span(
        trace_id=trace_id,
        parent_span_id=None,
        name=f"n8n.workflow.{workflow_name}",
        span_type=SpanType.AGENT,
        started_at=started_at,
        attributes={
            "n8n.workflow_id": workflow_id,
            "n8n.workflow_name": workflow_name,
            "n8n.execution_id": execution_id,
            "n8n.execution_mode": execution_mode,
            "service.name": "n8n",
        },
    )
    spans.append(workflow_span)

    # Extract node metadata
    node_name = webhook_data.node.get("name", "Unknown Node")
    node_type = webhook_data.node.get("type", "unknown")
    node_params = webhook_data.node.get("parameters", {})

    # Determine if this is an AI node
    is_ai = is_ai_node(node_type)
    ai_category = get_ai_node_category(node_type) if is_ai else None

    # Create node-level span
    node_span_name = f"n8n.node.{node_name}"
    node_attributes = {
        "n8n.node_name": node_name,
        "n8n.node_type": node_type,
        "service.name": "n8n",
    }

    # Add AI-specific attributes if applicable
    if is_ai:
        node_attributes["n8n.ai_category"] = ai_category
        ai_attrs = extract_ai_attributes(
            node_type, node_params, webhook_data.data
        )
        node_attributes.update(ai_attrs)

    # Determine span type
    if is_ai:
        if ai_category == "ai_agent":
            span_type = SpanType.AGENT
        elif ai_category in ["llm", "llm_chain"]:
            span_type = SpanType.LLM
        elif ai_category == "tool":
            span_type = SpanType.TOOL
        elif ai_category == "retrieval":
            span_type = SpanType.RETRIEVAL
        else:
            span_type = SpanType.CUSTOM
    else:
        span_type = SpanType.CUSTOM

    node_span = Span(
        trace_id=trace_id,
        parent_span_id=workflow_span.span_id,
        name=node_span_name,
        span_type=span_type,
        started_at=started_at,
        attributes=node_attributes,
    )
    spans.append(node_span)

    # Add input/output data as events
    if webhook_data.data:
        node_span.add_event(
            name="n8n.node.output",
            attributes={
                "item_count": len(webhook_data.data),
                "items": str(webhook_data.data)[:1000],  # Truncate
            },
        )

    # End both spans (since webhook is sent after execution)
    # Note: Span.end() automatically sets status to SUCCESS if still PENDING
    node_span.end()
    workflow_span.end()

    return spans


class N8nWebhookHandler:
    """
    HTTP server for receiving n8n webhook traces locally.

    This handler runs a lightweight HTTP server that receives webhook
    POST requests from n8n workflows and automatically converts them
    into Prela spans.

    Example:
        ```python
        from prela import init
        from prela.instrumentation.n8n.webhook import N8nWebhookHandler

        tracer = init(service_name="n8n-workflows")
        handler = N8nWebhookHandler(tracer, port=8787)
        handler.start()

        # Configure n8n webhook node to POST to http://localhost:8787/webhook
        # Handler will automatically trace all workflow executions
        ```
    """

    def __init__(self, tracer: Tracer, port: int = 8787, host: str = "0.0.0.0"):
        """
        Initialize the webhook handler.

        Args:
            tracer: Prela tracer instance for creating spans
            port: Port to listen on (default: 8787)
            host: Host to bind to (default: 0.0.0.0)
        """
        self.tracer = tracer
        self.port = port
        self.host = host
        self.app = None
        self.runner = None

    async def handle_webhook(self, request) -> Any:
        """
        Handle incoming webhook POST request.

        Args:
            request: aiohttp request object

        Returns:
            JSON response with status
        """
        try:
            # Parse JSON payload
            payload = await request.json()

            # Convert to spans
            spans = parse_n8n_webhook(payload)

            # Export spans via tracer's exporter
            if spans and self.tracer.exporter:
                for span in spans:
                    self.tracer.exporter.export([span])

            logger.info(f"Received n8n webhook, created {len(spans)} spans")

            return {
                "status": "success",
                "message": f"Created {len(spans)} spans",
                "trace_id": spans[0].trace_id if spans else None,
            }

        except Exception as e:
            logger.error(f"Error handling n8n webhook: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def start(self) -> None:
        """
        Start the HTTP server.

        This method starts an aiohttp server on the configured host and port.
        It runs in the current event loop, so it should be called from an
        async context or run in a separate thread.
        """
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for N8nWebhookHandler. "
                "Install with: pip install aiohttp"
            )

        async def _handle_webhook(request):
            result = await self.handle_webhook(request)
            return web.json_response(result)

        async def _start_server():
            self.app = web.Application()
            self.app.router.add_post("/webhook", _handle_webhook)
            self.app.router.add_post("/", _handle_webhook)  # Root endpoint

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, self.host, self.port)
            await site.start()

            logger.info(
                f"n8n webhook handler listening on http://{self.host}:{self.port}"
            )

        # Run the server
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_start_server())
        loop.run_forever()

    def start_background(self) -> None:
        """
        Start the HTTP server in a background thread.

        This method creates a new event loop and runs the server in it.
        Designed to be called from a background thread via threading.Thread.
        """
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for N8nWebhookHandler. "
                "Install with: pip install aiohttp"
            )

        async def _handle_webhook(request):
            result = await self.handle_webhook(request)
            return web.json_response(result)

        async def _start_server():
            self.app = web.Application()
            self.app.router.add_post("/webhook", _handle_webhook)
            self.app.router.add_post("/", _handle_webhook)  # Root endpoint

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, self.host, self.port)
            await site.start()

            logger.info(
                f"n8n webhook handler listening on http://{self.host}:{self.port}"
            )

        # Create new event loop for this thread
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start_server())
        loop.run_forever()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self.runner:
            import asyncio

            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.runner.cleanup())
            logger.info("n8n webhook handler stopped")
