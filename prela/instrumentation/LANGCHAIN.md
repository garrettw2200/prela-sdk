# LangChain Instrumentation

This document describes how to use Prela's auto-instrumentation for LangChain applications.

## Overview

The LangChain instrumentation provides automatic tracing for all LangChain operations including:

- **LLM Calls**: Traces calls to language models (OpenAI, Anthropic, etc. through LangChain)
- **Chains**: Captures chain executions (LLMChain, SequentialChain, etc.)
- **Tools**: Tracks tool invocations and results
- **Retrievers**: Monitors document retrieval operations
- **Agents**: Traces agent actions, decisions, and final outputs

Unlike the OpenAI and Anthropic instrumentors which use function wrapping, the LangChain instrumentor uses LangChain's built-in callback system for more comprehensive and reliable tracing.

## Installation

Install LangChain alongside Prela:

```bash
pip install prela[langchain]
# or
pip install prela langchain-core>=0.1.0
```

## Quick Start

### Automatic Instrumentation

The simplest way to enable LangChain tracing:

```python
import prela

# Initialize with auto-instrumentation
prela.init(service_name="my-langchain-app")

# Now all LangChain operations are automatically traced!
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)

# This chain execution is automatically traced
result = chain.run("colorful socks")
```

### Manual Instrumentation

For more control, manually instrument LangChain:

```python
from prela.core.tracer import Tracer
from prela.instrumentation.langchain import LangChainInstrumentor
from prela.exporters.console import ConsoleExporter

# Create tracer
tracer = Tracer(
    service_name="langchain-agent",
    exporter=ConsoleExporter()
)

# Instrument LangChain
instrumentor = LangChainInstrumentor()
instrumentor.instrument(tracer)

# Use LangChain as normal - all operations traced
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent execution is fully traced
agent.run("What is the current temperature in San Francisco? What is that in Celsius?")
```

## What Gets Traced

### LLM Calls

**Span Name**: `langchain.llm.{llm_type}` (e.g., `langchain.llm.openai`)

**Span Type**: `LLM`

**Attributes Captured**:
- `llm.vendor`: Always "langchain"
- `llm.type`: LLM type (openai, anthropic, etc.)
- `llm.model`: Model name (gpt-4, claude-3-opus, etc.)
- `llm.prompt_count`: Number of prompts
- `llm.prompt.{N}`: Individual prompts (up to 5, truncated to 500 chars)
- `llm.response.{N}`: Individual responses (up to 5, truncated to 500 chars)
- `llm.usage.prompt_tokens`: Input tokens used
- `llm.usage.completion_tokens`: Output tokens used
- `llm.usage.total_tokens`: Total tokens used
- `langchain.tags`: Tags from LangChain metadata
- `langchain.metadata.*`: Custom metadata fields

**Example Span**:
```json
{
  "name": "langchain.llm.openai",
  "span_type": "llm",
  "attributes": {
    "llm.vendor": "langchain",
    "llm.type": "openai",
    "llm.model": "gpt-4",
    "llm.prompt_count": 1,
    "llm.prompt.0": "What is the capital of France?",
    "llm.response.0": "The capital of France is Paris.",
    "llm.usage.prompt_tokens": 10,
    "llm.usage.completion_tokens": 8,
    "llm.usage.total_tokens": 18
  }
}
```

### Chain Executions

**Span Name**: `langchain.chain.{chain_type}` (e.g., `langchain.chain.LLMChain`)

**Span Type**: `AGENT`

**Attributes Captured**:
- `langchain.type`: Always "chain"
- `langchain.chain_type`: Type of chain
- `chain.input.{key}`: Input values (truncated to 500 chars)
- `chain.output.{key}`: Output values (truncated to 500 chars)
- `langchain.tags`: Tags from metadata
- `langchain.metadata.*`: Custom metadata

**Example Span**:
```json
{
  "name": "langchain.chain.LLMChain",
  "span_type": "agent",
  "attributes": {
    "langchain.type": "chain",
    "langchain.chain_type": "LLMChain",
    "chain.input.product": "eco-friendly water bottles",
    "chain.output.text": "AquaPure - The Sustainable Hydration Company"
  }
}
```

### Tool Invocations

**Span Name**: `langchain.tool.{tool_name}` (e.g., `langchain.tool.Calculator`)

**Span Type**: `TOOL`

**Attributes Captured**:
- `tool.name`: Tool name
- `tool.description`: Tool description
- `tool.input`: Input to the tool (truncated to 500 chars)
- `tool.output`: Output from the tool (truncated to 500 chars)
- `langchain.tags`: Tags from metadata
- `langchain.metadata.*`: Custom metadata

**Example Span**:
```json
{
  "name": "langchain.tool.Calculator",
  "span_type": "tool",
  "attributes": {
    "tool.name": "Calculator",
    "tool.description": "Useful for arithmetic calculations",
    "tool.input": "15 * 7",
    "tool.output": "105"
  }
}
```

### Retriever Operations

**Span Name**: `langchain.retriever.{retriever_type}` (e.g., `langchain.retriever.VectorStoreRetriever`)

**Span Type**: `RETRIEVAL`

**Attributes Captured**:
- `retriever.type`: Type of retriever
- `retriever.query`: Search query (truncated to 500 chars)
- `retriever.document_count`: Number of documents retrieved
- `retriever.doc.{N}.content`: Document content (up to 5 docs, truncated to 200 chars)
- `retriever.doc.{N}.metadata.*`: Document metadata fields
- `langchain.tags`: Tags from metadata
- `langchain.metadata.*`: Custom metadata

**Example Span**:
```json
{
  "name": "langchain.retriever.VectorStoreRetriever",
  "span_type": "retrieval",
  "attributes": {
    "retriever.type": "VectorStoreRetriever",
    "retriever.query": "What are the symptoms of the flu?",
    "retriever.document_count": 3,
    "retriever.doc.0.content": "Common flu symptoms include fever, cough, sore throat...",
    "retriever.doc.0.metadata.source": "medical_guide.pdf",
    "retriever.doc.0.metadata.page": "42"
  }
}
```

### Agent Actions

**Events**: Agent actions are recorded as span events, not separate spans

**Event Name**: `agent.action`

**Event Attributes**:
- `action.tool`: Tool being invoked
- `action.tool_input`: Input to the tool (truncated to 500 chars)
- `action.log`: Agent's reasoning log (truncated to 500 chars)

**Event Name**: `agent.finish`

**Event Attributes**:
- `finish.output`: Final agent output (truncated to 500 chars)
- `finish.log`: Completion log (truncated to 500 chars)

**Example Agent Span with Events**:
```json
{
  "name": "langchain.chain.AgentExecutor",
  "span_type": "agent",
  "events": [
    {
      "name": "agent.action",
      "timestamp": "2026-01-26T12:00:01.000Z",
      "attributes": {
        "action.tool": "Search",
        "action.tool_input": "current weather in Paris",
        "action.log": "I need to search for current weather information"
      }
    },
    {
      "name": "agent.action",
      "timestamp": "2026-01-26T12:00:03.000Z",
      "attributes": {
        "action.tool": "Calculator",
        "action.tool_input": "(18 * 9/5) + 32",
        "action.log": "Convert Celsius to Fahrenheit"
      }
    },
    {
      "name": "agent.finish",
      "timestamp": "2026-01-26T12:00:05.000Z",
      "attributes": {
        "finish.output": "The current temperature in Paris is 18°C (64.4°F)",
        "finish.log": "I now have the complete answer"
      }
    }
  ]
}
```

## Advanced Usage

### Manual Callback Usage

You can also add the callback handler to specific operations instead of global instrumentation:

```python
from prela.instrumentation.langchain import LangChainInstrumentor
from prela.core.tracer import Tracer

tracer = Tracer(service_name="my-app")
instrumentor = LangChainInstrumentor()
instrumentor.instrument(tracer)

# Get the callback handler
handler = instrumentor.get_callback()

# Use it for specific chains
from langchain.chains import LLMChain

chain = LLMChain(...)
result = chain.run(input_text, callbacks=[handler])
```

### Concurrent Operations

The instrumentation correctly handles concurrent LangChain operations by using `run_id` to track individual executions:

```python
import asyncio
from langchain.llms import OpenAI
from langchain.chains import LLMChain

async def run_multiple_chains():
    llm = OpenAI()

    # Multiple chains can run concurrently
    tasks = [
        LLMChain(llm=llm, prompt=prompt1).arun("input1"),
        LLMChain(llm=llm, prompt=prompt2).arun("input2"),
        LLMChain(llm=llm, prompt=prompt3).arun("input3"),
    ]

    results = await asyncio.gather(*tasks)
    # Each chain execution is traced separately with correct parent-child relationships

asyncio.run(run_multiple_chains())
```

### Nested Operations

LangChain operations naturally nest (chains call LLMs, agents use tools, etc.). The instrumentation correctly captures these relationships:

```python
# Example: Agent → Chain → LLM → Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool

# This creates a hierarchy of spans:
# - langchain.chain.AgentExecutor (root)
#   - langchain.llm.openai (agent's reasoning)
#   - langchain.tool.Search (tool execution)
#   - langchain.llm.openai (final answer generation)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run("What's the weather?")
```

## Error Handling

The instrumentation is designed to be defensive and never break user code:

1. **Callback Errors**: If a callback method encounters an error (e.g., malformed data), it logs the error but doesn't raise an exception
2. **Missing Attributes**: Missing or malformed LangChain response attributes are gracefully handled
3. **Import Errors**: If LangChain is not installed, instrumentation fails gracefully with a clear error message
4. **Uninstrumentation**: Safe to call `uninstrument()` multiple times or when not instrumented

```python
# Safe to call even if langchain-core is not installed
try:
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer)
except ImportError as e:
    print(f"LangChain not available: {e}")
    # Application continues without LangChain tracing
```

## Performance Considerations

The callback-based approach has minimal overhead:

1. **No Function Wrapping**: Uses LangChain's native callback system instead of monkey-patching
2. **Lazy Attribute Extraction**: Only extracts attributes when needed
3. **Truncation**: Long strings (prompts, responses, documents) are automatically truncated to prevent memory issues
4. **Sampling**: Works with Prela's sampling system to control trace volume

```python
import prela

# Sample only 10% of traces to reduce overhead in production
prela.init(
    service_name="production-agent",
    sample_rate=0.1
)
```

## Troubleshooting

### Instrumentation Not Working

**Problem**: LangChain operations aren't being traced

**Solutions**:
1. Verify LangChain is installed: `pip install langchain-core>=0.1.0`
2. Check that instrumentation was successful:
   ```python
   from prela.instrumentation.langchain import LangChainInstrumentor
   instrumentor = LangChainInstrumentor()
   print(f"Instrumented: {instrumentor.is_instrumented}")
   ```
3. Enable debug logging to see instrumentation details:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   prela.init(debug=True)
   ```

### Missing Attributes

**Problem**: Some expected attributes are missing from spans

**Cause**: LangChain's response structure varies by LLM provider and version

**Solution**: This is expected behavior. The instrumentation defensively extracts what's available without failing. Missing attributes won't break tracing.

### Duplicate Spans

**Problem**: Seeing duplicate spans for the same operation

**Cause**: Might be instrumenting both at the global level and manually adding callbacks

**Solution**: Choose one approach - either use global auto-instrumentation OR manual callback injection, not both.

## Examples

See [examples/langchain_instrumentation.py](../../examples/langchain_instrumentation.py) for complete working examples including:

- Basic chain tracing
- Agent tracing with tools
- Retrieval-augmented generation (RAG)
- Concurrent chain execution
- Error handling

## API Reference

### `LangChainInstrumentor`

```python
class LangChainInstrumentor(Instrumentor):
    def instrument(self, tracer: Tracer) -> None:
        """Enable LangChain instrumentation."""

    def uninstrument(self) -> None:
        """Disable LangChain instrumentation."""

    @property
    def is_instrumented(self) -> bool:
        """Check if currently instrumented."""

    def get_callback(self) -> PrelaCallbackHandler | None:
        """Get the active callback handler."""
```

### `PrelaCallbackHandler`

The callback handler implements LangChain's callback interface. You typically don't instantiate this directly - use `LangChainInstrumentor` instead.

## Integration with Other Instrumentors

LangChain instrumentation works alongside OpenAI and Anthropic instrumentors:

```python
import prela

# Auto-instruments all detected libraries
prela.init(service_name="multi-library-app")

# Traces will be created for:
# 1. Direct OpenAI/Anthropic API calls (via their instrumentors)
# 2. LangChain operations that call these APIs (via LangChain instrumentor)
# 3. All nested operations correctly linked

from langchain.llms import OpenAI  # Uses LangChain instrumentor
from openai import OpenAI as DirectOpenAI  # Uses OpenAI instrumentor

# Both approaches are traced, with proper span relationships
```

## Version Compatibility

- **Prela**: >= 0.1.0
- **LangChain Core**: >= 0.1.0
- **Python**: >= 3.9

The instrumentation is tested with the latest stable versions of LangChain. Older versions may have different callback signatures but should generally work.

## License

This instrumentation is part of the Prela SDK and follows the same license.
