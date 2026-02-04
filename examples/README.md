# Prela Examples

Working code examples demonstrating all Prela features.

## ✅ Production-Validated Test Scenarios

**NEW:** Complete test scenarios that validate all SDK features with real API calls:

- **[test_scenarios/](test_scenarios/)** - 6 production-validated scenarios (100% validated)
  - Simple success (basic tracing)
  - Multi-step workflow (span hierarchy)
  - Rate limit handling (error capture)
  - Streaming responses
  - Tool calling
  - Evaluation framework

**Status:** ✅ All 21 core features validated, 4/4 performance criteria met. See [test_scenarios/phase4_validation.md](test_scenarios/phase4_validation.md) for complete validation evidence.

---

## Quick Start

Get started with basic tracing:

- **[anthropic_instrumentation.py](anthropic_instrumentation.py)** - Auto-instrument Anthropic Claude SDK
- **[langchain_instrumentation.py](langchain_instrumentation.py)** - Auto-instrument LangChain applications
- **[llamaindex_instrumentation.py](llamaindex_instrumentation.py)** - Auto-instrument LlamaIndex applications

## Exporters

Export traces to different backends:

- **[console_exporter_demo.py](console_exporter_demo.py)** - Pretty-print traces to console with Rich library
- **[file_exporter_demo.py](file_exporter_demo.py)** - Export traces to JSONL files with search and rotation

## CLI Tool

Command-line interface for trace management:

- **[cli_demo.py](cli_demo.py)** - Complete CLI demonstration: init, list, show, search traces

## Evaluation Framework

Test and evaluate AI agent behavior:

### Test Definition
- **[eval_suite_demo.py](eval_suite_demo.py)** - Create test cases and suites with YAML/JSON

### Assertions
- **[assertions_demo.py](assertions_demo.py)** - All 10 assertion types with examples

### Running Tests
- **[eval_runner_demo.py](eval_runner_demo.py)** - Execute eval suites with parallel execution and callbacks

### Reporting
- **[reporters_demo.py](reporters_demo.py)** - Generate reports (Console, JSON, JUnit) for CI/CD

## Installation

Install Prela in development mode with all optional dependencies:

```bash
cd /Users/gw/prela/sdk
pip3 install -e ".[dev,openai,anthropic,langchain,llamaindex]"
```

Or install only the dependencies you need:

```bash
# Basic SDK only
pip3 install -e .

# With specific integrations
pip3 install -e ".[openai]"          # OpenAI only
pip3 install -e ".[anthropic]"       # Anthropic only
pip3 install -e ".[langchain]"       # LangChain only
pip3 install -e ".[llamaindex]"      # LlamaIndex only

# With CLI tools
pip3 install -e ".[cli]"

# Everything
pip3 install -e ".[all]"
```

## Running Examples

Each example is self-contained and can be run directly:

```bash
# Auto-instrumentation examples
python examples/anthropic_instrumentation.py
python examples/langchain_instrumentation.py
python examples/llamaindex_instrumentation.py

# Exporter examples
python examples/console_exporter_demo.py
python examples/file_exporter_demo.py

# CLI example (demonstrates commands)
python examples/cli_demo.py

# Evaluation examples
python examples/eval_suite_demo.py
python examples/assertions_demo.py
python examples/eval_runner_demo.py
python examples/reporters_demo.py
```

## Example Categories

### 🚀 Auto-Instrumentation (3 examples)

Automatically trace LLM SDK calls without code changes:

1. **Anthropic** - Messages API, streaming, tool use, thinking blocks
2. **LangChain** - Chains, agents, tools, retrievers
3. **LlamaIndex** - Query engines, retrievers, embeddings

### 📊 Exporters (2 examples)

Send traces to different backends:

1. **Console** - Pretty terminal output with Rich (tree view, verbosity control)
2. **File** - JSONL export with search, rotation, cleanup

### 💻 CLI (1 example)

Command-line interface for local development:

- Initialize config
- List recent traces
- Show trace details with tree visualization
- Search traces by query

### ✅ Evaluations (4 examples)

Test framework for agent behavior:

1. **Suite Definition** - YAML/JSON test cases with tags and filtering
2. **Assertions** - 10 types: structural, tool, semantic (contains, equals, tool_called, etc.)
3. **Runner** - Sequential/parallel execution with tracer integration
4. **Reporters** - Console (Rich), JSON (data), JUnit (CI/CD)

## API Keys

Some examples require API keys:

```bash
# Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# OpenAI (if using OpenAI examples)
export OPENAI_API_KEY="your-key-here"
```

## Next Steps

- 📚 Read the [documentation](https://prela.readthedocs.io)
- 🔧 Check out the [API reference](https://prela.readthedocs.io/api/core/)
- 🧪 Run the [test suite](../tests/)
- 💡 Browse [integration guides](https://prela.readthedocs.io/integrations/)

## Contributing

Found a bug or have an example to add? See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

All examples are licensed under Apache 2.0 (same as Prela SDK).
