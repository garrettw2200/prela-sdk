"""
Example: n8n Workflow Tracing via Webhooks

This example shows how to receive and trace n8n workflow executions
via HTTP webhooks using the N8nWebhookHandler.

Setup:
1. Start the webhook handler (runs on http://0.0.0.0:8787)
2. In n8n, add an HTTP Request node at the end of your workflow
3. Configure it to POST to http://localhost:8787/webhook with workflow data

The handler automatically:
- Parses n8n webhook payloads
- Creates workflow and node spans
- Detects AI nodes (OpenAI, Anthropic, LangChain, etc.)
- Extracts token usage, costs, and prompts
- Exports traces to your configured exporter
"""

import asyncio
import logging

from prela import init
from prela.instrumentation.n8n.webhook import N8nWebhookHandler

# Enable debug logging to see webhook processing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the n8n webhook handler server."""

    # Initialize Prela with console exporter
    tracer = init(
        service_name="n8n-workflows",
        exporter="console",  # Can also use "file" or custom exporter
        sample_rate=1.0,  # Trace everything
    )

    # Create webhook handler
    handler = N8nWebhookHandler(
        tracer=tracer,
        port=8787,  # Default port
        host="0.0.0.0",  # Listen on all interfaces
    )

    logger.info("Starting n8n webhook handler...")
    logger.info("Configure n8n webhook to POST to: http://localhost:8787/webhook")
    logger.info("Press Ctrl+C to stop")

    try:
        # Start the server (blocking call)
        handler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        handler.stop()


def example_webhook_payload():
    """
    Example n8n webhook payload for testing.

    In n8n, configure an HTTP Request node with:
    - Method: POST
    - URL: http://localhost:8787/webhook
    - Body: JSON
    - Body Parameters:
        workflow: {{ $workflow }}
        execution: {{ $execution }}
        node: {{ $node }}
        data: {{ $json }}
    """
    return {
        "workflow": {
            "id": "wf_123",
            "name": "AI Content Pipeline",
            "active": True,
        },
        "execution": {
            "id": "exec_456",
            "mode": "manual",
            "startedAt": "2025-01-27T10:00:00.000Z",
        },
        "node": {
            "name": "OpenAI GPT-4",
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


async def test_webhook_handler():
    """Test the webhook handler with a sample payload."""
    import json

    from aiohttp import ClientSession

    # Start handler in background
    tracer = init(service_name="n8n-test", exporter="console")
    handler = N8nWebhookHandler(tracer, port=8787)

    # Create server task
    server_task = asyncio.create_task(_run_server(handler))

    # Wait for server to start
    await asyncio.sleep(1)

    # Send test webhook
    async with ClientSession() as session:
        payload = example_webhook_payload()
        async with session.post(
            "http://localhost:8787/webhook", json=payload
        ) as resp:
            result = await resp.json()
            print(f"Response: {json.dumps(result, indent=2)}")

    # Cleanup
    handler.stop()
    server_task.cancel()


async def _run_server(handler):
    """Helper to run server in background."""
    try:
        handler.start()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    # Run the webhook handler server
    main()

    # Or test with a sample payload
    # asyncio.run(test_webhook_handler())
