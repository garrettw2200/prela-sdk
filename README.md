# Prela SDK

[![PyPI version](https://badge.fury.io/py/prela.svg)](https://badge.fury.io/py/prela)
[![Python Support](https://img.shields.io/pypi/pyversions/prela.svg)](https://pypi.org/project/prela/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/garrettw2200/prela-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/garrettw2200/prela-sdk/actions/workflows/test.yml)

**Open-source Python SDK for AI agent observability, testing, and debugging.**

Prela provides comprehensive instrumentation, evaluation frameworks, and replay capabilities for AI agents built with OpenAI, Anthropic, LangChain, LlamaIndex, CrewAI, AutoGen, LangGraph, and more.

## Features

- **Auto-instrumentation** - Zero-code observability for 10+ AI frameworks
- **Evaluation Framework** - 17+ assertion types for testing agent behavior
- **Replay Engine** - Deterministic replay for debugging and testing
- **Multiple Exporters** - Console, file, HTTP, and ClickHouse outputs
- **Multi-Agent Support** - Track complex agent interactions and delegations
- **CLI Tools** - Command-line interface for trace management
- **Production Ready** - 836+ tests with comprehensive coverage

## Installation

```bash
pip install prela
```

### With Optional Integrations

```bash
# Install specific integrations
pip install prela[openai]       # OpenAI SDK
pip install prela[anthropic]    # Anthropic/Claude SDK
pip install prela[langchain]    # LangChain
pip install prela[llamaindex]   # LlamaIndex

# Install all integrations
pip install prela[all]

# Install with development tools
pip install prela[dev]

# Install with CLI tools
pip install prela[cli]
```

## Quick Start

### Basic Usage

```python
import prela

# Initialize with console output
prela.init(service_name="my-agent", exporter="console")

# Your AI agent code here
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Prela automatically traces the OpenAI call
```

### Auto-Instrumentation

Prela automatically instruments supported frameworks:

```python
import prela

# One-line initialization
prela.init(service_name="my-app", exporter="console")

# All these frameworks are automatically traced:
# - OpenAI API calls
# - Anthropic/Claude API calls
# - LangChain chains and agents
# - LlamaIndex queries
# - CrewAI tasks
# - AutoGen conversations
# - LangGraph workflows
# - OpenAI Swarm agents
```

### Manual Instrumentation

```python
from prela import trace

@trace
def my_agent_function(user_input: str):
    # Your agent logic here
    return process_input(user_input)

# Or use context managers
from prela import tracer

with tracer.start_span("complex-operation") as span:
    result = do_something()
    span.set_attribute("result_count", len(result))
```

### Evaluation Framework

```python
from prela.evals import EvalRunner, TestCase, Assertion

# Define test cases
test_cases = [
    TestCase(
        name="greeting-test",
        input={"message": "Hello"},
        assertions=[
            Assertion.contains("hello", case_sensitive=False),
            Assertion.max_length(100)
        ]
    )
]

# Run evaluations
runner = EvalRunner()
results = runner.run(my_agent_function, test_cases)
results.print_summary()
```

### Replay Engine

```python
from prela.replay import ReplayEngine

# Load a trace
engine = ReplayEngine()
trace = engine.load_trace("trace_id_123")

# Replay with comparison
replay_result = engine.replay(trace)
comparison = engine.compare(trace, replay_result)

print(f"Similarity: {comparison.similarity_score}")
```

## Supported Frameworks

| Framework | Auto-Instrumentation | Manual Tracing | Version Support |
|-----------|---------------------|----------------|-----------------|
| OpenAI | ✅ | ✅ | >=1.0.0 |
| Anthropic | ✅ | ✅ | >=0.18.0 |
| LangChain | ✅ | ✅ | >=0.1.0 |
| LlamaIndex | ✅ | ✅ | >=0.9.0 |
| CrewAI | ✅ | ✅ | >=0.1.0 |
| AutoGen | ✅ | ✅ | >=0.2.0 |
| LangGraph | ✅ | ✅ | >=0.0.20 |
| OpenAI Swarm | ✅ | ✅ | >=0.1.0 |
| N8N | ✅ | ✅ | All versions |

## Exporters

### Console Exporter

```python
prela.init(exporter="console")
```

### File Exporter

```python
prela.init(
    exporter="file",
    file_path="traces.jsonl"
)
```

### HTTP Exporter

```python
prela.init(
    exporter="http",
    endpoint="https://api.prela.io",
    api_key="your-api-key"
)
```

### ClickHouse Exporter

```python
prela.init(
    exporter="clickhouse",
    clickhouse_host="localhost",
    clickhouse_port=8123
)
```

## CLI Tools

```bash
# List traces
prela list

# Show trace details
prela show <trace-id>

# Show most recent trace
prela last

# Search traces
prela search "error"

# Show failed traces
prela errors

# Interactive explorer
prela explore
```

## Evaluation Assertions

Prela includes 17+ assertion types:

**Structural Assertions:**
- `contains()` - Check for substring
- `regex()` - Pattern matching
- `json_schema()` - Validate JSON structure
- `length()` - Check output length

**Semantic Assertions:**
- `similarity()` - Embedding-based similarity

**Tool Assertions:**
- `tool_called()` - Verify tool usage
- `tool_args()` - Validate tool arguments
- `tool_sequence()` - Check tool call order

**Multi-Agent Assertions:**
- `agent_used()` - Verify agent participation
- `task_completed()` - Check task completion
- `delegation_pattern()` - Validate delegation
- `no_circular_delegation()` - Prevent cycles
- `collaboration_count()` - Track interactions

## Advanced Features

### Sampling

```python
from prela.core import AlwaysOnSampler, ProbabilitySampler, RateLimitSampler

# Always sample
prela.init(sampler=AlwaysOnSampler())

# Sample 10% of traces
prela.init(sampler=ProbabilitySampler(0.1))

# Sample max 100 traces/second
prela.init(sampler=RateLimitSampler(100))
```

### Context Propagation

```python
from prela import get_current_span

def my_function():
    span = get_current_span()
    span.set_attribute("custom_field", "value")
    span.add_event("checkpoint reached")
```

### Custom Exporters

```python
from prela.exporters import SpanExporter

class MyExporter(SpanExporter):
    def export(self, spans):
        for span in spans:
            # Custom export logic
            pass

prela.init(exporter=MyExporter())
```

## Examples

See the [examples/](examples/) directory for complete working examples:

- [OpenAI Chatbot](examples/openai_basic.py)
- [Anthropic Claude Agent](examples/anthropic_basic.py)
- [LangChain RAG](examples/langchain_rag.py)
- [CrewAI Multi-Agent](examples/crewai_example.py)
- [Evaluation Framework](examples/eval_runner.py)
- [Replay Engine](examples/replay_example.py)

## Documentation

Full documentation is available at [docs.prela.dev](https://docs.prela.dev)

- [Getting Started Guide](https://docs.prela.dev/getting-started/)
- [API Reference](https://docs.prela.dev/api/)
- [Integration Guides](https://docs.prela.dev/integrations/)
- [Evaluation Framework](https://docs.prela.dev/evaluation/)
- [Replay Engine](https://docs.prela.dev/replay/)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/garrettw2200/prela-sdk.git
cd prela-sdk

# Install development dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linters
ruff check .
black .
mypy prela
```

## Community

- [GitHub Discussions](https://github.com/garrettw2200/prela-sdk/discussions)
- [Issue Tracker](https://github.com/garrettw2200/prela-sdk/issues)
- [Discord Community](https://discord.gg/bCMfHnZD)

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with love by the Prela team and contributors.

---

**Star this repo** if you find it useful!
