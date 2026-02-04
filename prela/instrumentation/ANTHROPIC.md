# Anthropic SDK Instrumentation

This document describes the Anthropic SDK instrumentation for automatic tracing of Claude API calls.

## Overview

The `AnthropicInstrumentor` provides automatic tracing for all Anthropic API calls by monkey-patching the `anthropic` Python SDK (version 0.40.0+). Once instrumented, all API calls create spans that capture detailed request/response information, token usage, latency, tool use, and errors.

## Features

### Supported Methods

- ✅ `anthropic.Anthropic.messages.create` (sync)
- ✅ `anthropic.AsyncAnthropic.messages.create` (async)
- ✅ `anthropic.Anthropic.messages.stream` (sync streaming)
- ✅ `anthropic.AsyncAnthropic.messages.stream` (async streaming)

### Captured Data

#### Request Attributes
- `llm.vendor`: Always `"anthropic"`
- `llm.model`: Model identifier (e.g., `"claude-sonnet-4-20250514"`)
- `llm.request.model`: Requested model name
- `llm.system`: System prompt (if provided)
- `llm.temperature`: Sampling temperature
- `llm.max_tokens`: Maximum output tokens
- `llm.stream`: Whether streaming is enabled (for streaming calls)

#### Response Attributes
- `llm.response.model`: Actual model used (may differ from request)
- `llm.response.id`: Response message ID
- `llm.input_tokens`: Input token count from usage
- `llm.output_tokens`: Output token count from usage
- `llm.stop_reason`: Stop reason (`end_turn`, `max_tokens`, `stop_sequence`, `tool_use`)
- `llm.latency_ms`: Total request latency in milliseconds
- `llm.time_to_first_token_ms`: Time to first token (streaming only)

#### Events
- `llm.request`: Captures request messages and system prompt
- `llm.response`: Captures response content blocks
- `llm.tool_use`: Captures tool calls when `stop_reason == "tool_use"`
- `llm.thinking`: Captures extended thinking blocks (if enabled)
- `error`: Captures error details on failure

#### Error Attributes (on failure)
- `error.type`: Exception class name
- `error.message`: Error message
- `error.status_code`: HTTP status code (if applicable)

## Usage

### Basic Example

```python
from prela.core.tracer import Tracer
from prela.instrumentation.anthropic import AnthropicInstrumentor
import anthropic

# Initialize tracer
tracer = Tracer()

# Instrument Anthropic SDK
instrumentor = AnthropicInstrumentor()
instrumentor.instrument(tracer)

# Now all Anthropic calls are automatically traced
client = anthropic.Anthropic(api_key="your-api-key")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Streaming Example

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Streaming automatically captures:
# - Time to first token
# - Aggregated response text
# - Final token usage
```

### Async Example

```python
import asyncio

async def main():
    client = anthropic.AsyncAnthropic(api_key="your-api-key")

    # Async calls are automatically traced
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Async streaming also works
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Count to 5"}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

asyncio.run(main())
```

### Tool Use Example

```python
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
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
    messages=[{"role": "user", "content": "What's the weather in SF?"}]
)

# Tool calls are automatically captured in span events
if response.stop_reason == "tool_use":
    for block in response.content:
        if block.type == "tool_use":
            # Tool execution can be traced separately
            result = execute_tool(block.name, block.input)
```

### Uninstrumenting

```python
# Disable instrumentation and restore original functions
instrumentor.uninstrument()

# Future calls will not be traced
response = client.messages.create(...)
```

## Advanced Features

### Extended Thinking

When extended thinking is enabled in the API, thinking blocks are automatically captured:

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    thinking={
        "type": "enabled",
        "budget_tokens": 1000
    },
    messages=[{"role": "user", "content": "Solve this problem..."}]
)

# Thinking blocks are captured in the llm.thinking event
```

### Error Handling

All errors are automatically captured and re-raised:

```python
try:
    response = client.messages.create(
        model="invalid-model",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}]
    )
except anthropic.APIError as e:
    # Error is captured in span with:
    # - error.type: "APIError"
    # - error.message: Error description
    # - error.status_code: HTTP status code
    # Span status set to ERROR
    pass
```

### Defensive Programming

The instrumentation is designed to never crash user code:

- All attribute extraction is wrapped in try/except
- Malformed responses don't break instrumentation
- Missing attributes are silently skipped
- Logging is used for debugging issues

## Implementation Details

### Monkey Patching

The instrumentor wraps the following methods using `wrap_function`:

1. `Messages.create` (sync and async)
2. `Messages.stream` (sync and async)

The original functions are stored in `module.__prela_originals__` and can be restored via `uninstrument()`.

### Stream Wrapping

Streaming calls return wrapped stream objects:

- `TracedMessageStream` (sync)
- `TracedAsyncMessageStream` (async)

These wrappers:
- Implement context manager protocol (`__enter__`/`__exit__` or async equivalents)
- Implement iterator protocol (`__iter__`/`__next__` or async equivalents)
- Process streaming events to extract metadata
- Aggregate text content for final span
- Calculate time-to-first-token

### Thread Safety

The instrumentation is thread-safe:
- Wrapping/unwrapping uses module-level locks (handled by Python's import system)
- Span creation is handled by the tracer (which manages context per-thread)
- No shared mutable state in the instrumentor

## Testing

The instrumentation has comprehensive test coverage:

- **33 tests** covering all functionality
- **94% code coverage** (remaining 6% is defensive error logging)
- Tests use mocked Anthropic SDK (no API calls required)
- Includes sync, async, streaming, tool use, thinking, and error cases

Run tests:
```bash
pytest tests/test_instrumentation/test_anthropic.py -v
```

Check coverage:
```bash
pytest tests/test_instrumentation/test_anthropic.py --cov=prela.instrumentation.anthropic --cov-report=term-missing
```

## Requirements

- Python 3.9+
- `anthropic>=0.40.0` (optional, only needed if using Anthropic)
- `prela` SDK

Install:
```bash
pip install prela[anthropic]
# or
pip install anthropic>=0.40.0
```

## Limitations

1. **SDK Version**: Requires `anthropic>=0.40.0`. Older versions may have different APIs.

2. **Completion API**: This instrumentor only supports the Messages API. The legacy Completions API is not supported.

3. **Prompt Caching**: Prompt caching metadata is not yet captured (can be added if needed).

4. **Batching**: The SDK doesn't support batching, so this isn't applicable.

## Future Enhancements

Potential improvements:

- [ ] Capture prompt caching hits/misses
- [ ] Capture image inputs (multimodal)
- [ ] Support for future API features
- [ ] Integration with LangChain's Anthropic wrapper
- [ ] Cost calculation based on token usage

## See Also

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Prela SDK Documentation](../../docs/)
- [Base Instrumentor](base.py)
- [Example Usage](../../examples/anthropic_instrumentation.py)
