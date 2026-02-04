"""Example of using the Anthropic instrumentation.

This example demonstrates how to automatically trace Anthropic API calls
using the AnthropicInstrumentor.

Requirements:
    pip install prela[anthropic]
    # or
    pip install anthropic>=0.40.0

Usage:
    export ANTHROPIC_API_KEY=your-api-key
    python examples/anthropic_instrumentation.py
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for importing prela
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prela.instrumentation.anthropic import AnthropicInstrumentor


def example_basic_message():
    """Example: Basic message creation with tracing."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Message Creation")
    print("=" * 60)

    # NOTE: This is a mock example showing the API
    # In a real scenario, you would:
    # 1. Create a tracer instance
    # 2. Instrument the Anthropic SDK
    # 3. Make API calls normally - they'll be automatically traced

    code_example = '''
    from prela.core.tracer import Tracer
    from prela.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    # Initialize tracer
    tracer = Tracer()

    # Instrument Anthropic SDK
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer)

    # Now all Anthropic calls are automatically traced
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude!"}]
    )

    print(response.content[0].text)

    # The span will automatically capture:
    # - Request: model, max_tokens, messages, system prompt
    # - Response: model used, tokens, stop reason, latency
    # - Events: request and response content
    '''

    print(code_example)


def example_streaming():
    """Example: Streaming messages with tracing."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Messages")
    print("=" * 60)

    code_example = '''
    from prela.core.tracer import Tracer
    from prela.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    tracer = Tracer()
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Streaming is automatically traced
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a haiku about observability"}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    # The span captures:
    # - Time to first token
    # - Aggregated response text
    # - Streaming events
    # - Final token usage
    '''

    print(code_example)


def example_tool_use():
    """Example: Tool use with tracing."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Use")
    print("=" * 60)

    code_example = '''
    from prela.core.tracer import Tracer
    from prela.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    tracer = Tracer()
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Define a tool
    tools = [{
        "name": "get_weather",
        "description": "Get the weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}]
    )

    # The span captures tool calls:
    # - Tool IDs, names, and input parameters
    # - Stop reason (tool_use)
    # - Tool execution can be traced separately

    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                print(f"Tool call: {block.name}")
                print(f"Input: {block.input}")
    '''

    print(code_example)


def example_async():
    """Example: Async API with tracing."""
    print("\n" + "=" * 60)
    print("Example 4: Async API")
    print("=" * 60)

    code_example = '''
    import asyncio
    from prela.core.tracer import Tracer
    from prela.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    async def main():
        tracer = Tracer()
        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument(tracer)

        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Async calls are automatically traced
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}]
        )

        print(response.content[0].text)

        # Async streaming also works
        async with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 5"}]
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)

    asyncio.run(main())
    '''

    print(code_example)


def example_uninstrument():
    """Example: Uninstrumenting the SDK."""
    print("\n" + "=" * 60)
    print("Example 5: Uninstrumenting")
    print("=" * 60)

    code_example = '''
    from prela.core.tracer import Tracer
    from prela.instrumentation.anthropic import AnthropicInstrumentor
    import anthropic

    tracer = Tracer()
    instrumentor = AnthropicInstrumentor()

    # Enable instrumentation
    instrumentor.instrument(tracer)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # This call is traced
    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hi"}]
    )

    # Disable instrumentation
    instrumentor.uninstrument()

    # This call is NOT traced
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hi again"}]
    )
    '''

    print(code_example)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Prela Anthropic Instrumentation Examples")
    print("=" * 60)
    print("\nThese examples show how to use the AnthropicInstrumentor")
    print("to automatically trace Anthropic API calls.")
    print("\nNote: These are code examples, not executable demos.")
    print("To run them, you need:")
    print("  1. A Tracer implementation (coming in next phase)")
    print("  2. An Anthropic API key")

    example_basic_message()
    example_streaming()
    example_tool_use()
    example_async()
    example_uninstrument()

    print("\n" + "=" * 60)
    print("Captured Span Attributes")
    print("=" * 60)

    attributes_doc = '''
    The instrumentation automatically captures:

    Request Attributes:
    - llm.vendor: "anthropic"
    - llm.model: Model name (e.g., "claude-sonnet-4-20250514")
    - llm.request.model: Requested model
    - llm.system: System prompt (if provided)
    - llm.temperature: Sampling temperature
    - llm.max_tokens: Maximum output tokens
    - llm.stream: Whether streaming is enabled

    Response Attributes:
    - llm.response.model: Actual model used
    - llm.response.id: Response ID
    - llm.input_tokens: Input token count
    - llm.output_tokens: Output token count
    - llm.stop_reason: Why generation stopped
    - llm.latency_ms: Total latency in milliseconds
    - llm.time_to_first_token_ms: TTFT for streaming (if applicable)

    Events:
    - llm.request: Request messages and parameters
    - llm.response: Response content
    - llm.tool_use: Tool calls (if any)
    - llm.thinking: Extended thinking content (if enabled)
    - error: Error details (if failed)

    Error Attributes (on failure):
    - error.type: Exception type
    - error.message: Error message
    - error.status_code: HTTP status code (if applicable)
    '''

    print(attributes_doc)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
