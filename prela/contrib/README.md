# Prela CLI

Command-line interface for the Prela AI Agent Observability Platform.

## Quick Start

### Installation

```bash
pip install prela[cli]
```

### Basic Usage

```bash
# 1. Initialize configuration
prela init

# 2. Run your AI application (generates traces)
python my_agent.py

# 3. View traces
prela trace list
prela trace show <trace-id>
prela trace search <query>
```

## Commands

| Command | Description |
|---------|-------------|
| `prela init` | Initialize configuration with interactive prompts |
| `prela trace list` | List recent traces |
| `prela trace show` | Display full trace tree |
| `prela trace search` | Search traces by name or attributes |
| `prela serve` | Start local web dashboard (coming soon) |
| `prela eval` | Run evaluation suite (coming soon) |

## Documentation

See [CLI.md](CLI.md) for complete documentation.

## Example

```bash
$ prela init
Service name [my-agent]: my-chatbot
Exporter (console/file) [file]: file
Trace directory [./traces]: ./traces
Sample rate (0.0-1.0) [1.0]: 1.0

✓ Configuration saved to .prela.yaml
✓ Created trace directory: ./traces

$ python examples/cli_demo.py
Prela initialized with file exporter
Generating sample traces...
✓ Created trace: example_operation
✓ Created trace: parent_task (with 2 child spans)

$ prela trace list --limit 5

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace ID         ┃ Root Span          ┃ Duration ┃ Status ┃ Spans ┃ Time                ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ trace-abc123...  │ example_operation  │ 123ms    │ ✓      │ 1     │ 2025-01-26 10:30:45 │
│ trace-def456...  │ parent_task        │ 234ms    │ ✓      │ 3     │ 2025-01-26 10:30:46 │
└──────────────────┴────────────────────┴──────────┴────────┴───────┴─────────────────────┘

$ prela trace show trace-abc123

Trace: trace-abc123456789

example_operation (custom) success 123ms

Span Details:

example_operation
  Span ID: span-001
  Type: custom
  Status: success
  Attributes:
    operation.type: demo
    operation.complexity: simple
  Events (2):
    - operation.started @ 2025-01-26T10:30:45.123456Z
    - operation.completed @ 2025-01-26T10:30:45.246789Z
```

## Features

- ✅ **Configuration Management** - Interactive setup with `.prela.yaml`
- ✅ **Trace Listing** - View recent traces with filtering and pagination
- ✅ **Trace Details** - Full trace tree visualization with Rich
- ✅ **Search** - Find traces by span names or attributes
- ✅ **Time Filtering** - View traces from specific time ranges
- ⏳ **Web Dashboard** - Coming in Phase 1
- ⏳ **Eval Suite** - Coming in Phase 1

## Requirements

- Python 3.9+
- typer >= 0.9.0
- pyyaml >= 6.0
- rich >= 13.0.0

## Development

Run tests:
```bash
pytest tests/test_cli.py -v
```

All 39 tests passing ✅

## License

Apache 2.0
