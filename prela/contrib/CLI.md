# Prela CLI Documentation

The Prela CLI provides command-line tools for managing Prela configuration, viewing traces, searching traces, and running evaluations.

## Installation

Install the CLI dependencies:

```bash
pip install prela[cli]
```

This installs:
- `typer` - CLI framework
- `pyyaml` - YAML configuration support
- `rich` - Beautiful terminal output

## Commands

### `prela init`

Initialize Prela configuration with interactive prompts.

**Usage:**
```bash
prela init
```

**Interactive Prompts:**
1. Service name (default: `my-agent`)
2. Exporter type (`console` or `file`, default: `file`)
3. Trace directory (default: `./traces`, only for file exporter)
4. Sample rate (0.0-1.0, default: `1.0`)

**Output:**
- Creates `.prela.yaml` configuration file in current directory
- Creates trace directory (if using file exporter)

**Example:**
```bash
$ prela init
Prela Configuration Setup

Service name [my-agent]: my-chatbot
Exporter (console/file) [file]: file
Trace directory [./traces]: ./my-traces
Sample rate (0.0-1.0) [1.0]: 0.5

✓ Configuration saved to .prela.yaml
✓ Created trace directory: ./my-traces
```

**Configuration File Format:**
```yaml
service_name: my-chatbot
exporter: file
trace_dir: ./my-traces
sample_rate: 0.5
```

---

### `prela trace list`

List recent traces from file exporter.

**Usage:**
```bash
prela trace list [OPTIONS]
```

**Options:**
- `--limit, -n INTEGER` - Maximum number of traces to show (default: 20)
- `--since, -s DURATION` - Show traces since duration ago (e.g., `1h`, `30m`, `2d`)

**Duration Format:**
- `s` - seconds (e.g., `30s`)
- `m` - minutes (e.g., `15m`)
- `h` - hours (e.g., `2h`)
- `d` - days (e.g., `3d`)

**Output:**
Displays table with:
- Trace ID (first 16 chars)
- Root span name
- Duration (ms or seconds)
- Status (success/error/pending)
- Number of spans
- Timestamp

**Example:**
```bash
$ prela trace list --limit 10 --since 1h

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Trace ID         ┃ Root Span          ┃ Duration ┃ Status ┃ Spans ┃ Time                ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ trace-abc123...  │ anthropic.message  │ 1.23s    │ ✓      │ 3     │ 2025-01-26 10:30:45 │
│ trace-def456...  │ langchain.agent    │ 2.45s    │ ✓      │ 8     │ 2025-01-26 10:28:12 │
│ trace-ghi789...  │ custom_operation   │ 567ms    │ ✗      │ 2     │ 2025-01-26 10:25:33 │
└──────────────────┴────────────────────┴──────────┴────────┴───────┴─────────────────────┘
```

---

### `prela trace show`

Display full trace tree with all spans, attributes, and events.

**Usage:**
```bash
prela trace show TRACE_ID
```

**Arguments:**
- `TRACE_ID` - Full or partial trace ID (supports prefix matching)

**Output:**
1. Trace tree visualization (hierarchical spans)
2. Detailed span information:
   - Span ID
   - Type (agent/llm/tool/etc.)
   - Status
   - Attributes (key-value pairs)
   - Events (timestamped events)

**Example:**
```bash
$ prela trace show trace-abc123

Trace: trace-abc123456789

anthropic.messages.create (llm) success 1.23s
└── anthropic.stream.process (custom) success 450ms

Span Details:

anthropic.messages.create
  Span ID: span-001
  Type: llm
  Status: success
  Attributes:
    llm.vendor: anthropic
    llm.model: claude-3-5-sonnet-20241022
    llm.input_tokens: 150
    llm.output_tokens: 89
    llm.temperature: 1.0
  Events (2):
    - llm.request @ 2025-01-26T10:30:45.123456Z
    - llm.response @ 2025-01-26T10:30:46.356789Z

anthropic.stream.process
  Span ID: span-002
  Type: custom
  Status: success
  Attributes:
    stream.chunks: 12
```

---

### `prela trace search`

Search span names and attributes for matching traces.

**Usage:**
```bash
prela trace search QUERY
```

**Arguments:**
- `QUERY` - Search query (case-insensitive)

**Search Scope:**
- Span names
- Attribute keys
- Attribute values

**Output:**
Table showing matching traces with:
- Trace ID
- Root span name
- Number of matching spans
- Status

**Example:**
```bash
$ prela trace search anthropic

Found 3 traces matching 'anthropic'

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Trace ID         ┃ Root Span              ┃ Matching Spans  ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ trace-abc123...  │ anthropic.messages     │ 2               │ ✓      │
│ trace-def456...  │ langchain.agent        │ 1               │ ✓      │
│ trace-ghi789...  │ anthropic.stream       │ 3               │ ✓      │
└──────────────────┴────────────────────────┴─────────────────┴────────┘
```

---

### `prela serve`

Start local web dashboard (placeholder - not yet implemented).

**Usage:**
```bash
prela serve [OPTIONS]
```

**Options:**
- `--port, -p INTEGER` - Port to run server on (default: 8000)

**Status:**
This feature is planned for Phase 1 (Months 4-8). When implemented, it will:
- Start FastAPI server on specified port
- Serve REST API endpoints for trace retrieval
- Serve static React frontend for trace visualization
- Provide real-time trace streaming (WebSocket)

**Example:**
```bash
$ prela serve --port 8000

Web dashboard not yet implemented
Planned: Start server on http://localhost:8000
Will serve API endpoints + static frontend

This feature is planned for Phase 1 (Months 4-8)
```

---

### `prela eval`

Run evaluation suite (placeholder - not yet implemented).

**Usage:**
```bash
prela eval SUITE_PATH
```

**Arguments:**
- `SUITE_PATH` - Path to evaluation suite file (YAML or JSON)

**Status:**
This feature is planned for Phase 1 (Months 4-8). When implemented, it will:
- Load evaluation suite from file
- Run test cases against traced LLM calls
- Compare outputs against expected results
- Generate pass/fail metrics and reports

**Planned Suite Format:**
```yaml
name: my-eval-suite
test_cases:
  - name: factual_accuracy
    input: "What is the capital of France?"
    expected_output: "Paris"
    criteria:
      - type: exact_match
      - type: semantic_similarity
        threshold: 0.9

  - name: hallucination_check
    input: "Explain quantum computing"
    criteria:
      - type: no_hallucination
      - type: coherence
```

**Example:**
```bash
$ prela eval suite.yaml

Eval runner not yet implemented
Planned: Run eval suite from suite.yaml
Will output results with pass/fail metrics

This feature is planned for Phase 1 (Months 4-8)
```

---

## Configuration File

The `.prela.yaml` configuration file is created by `prela init` and used by all commands.

**Location:**
Current working directory (`.prela.yaml`)

**Format:**
```yaml
service_name: my-agent      # Service identifier
exporter: file              # Exporter type (console/file)
trace_dir: ./traces         # Trace directory (file exporter only)
sample_rate: 1.0            # Sampling rate (0.0-1.0)
```

**Default Values:**
- `service_name`: `my-agent`
- `exporter`: `file`
- `trace_dir`: `./traces`
- `sample_rate`: `1.0` (100%)

**Manual Editing:**
You can manually edit `.prela.yaml` instead of using `prela init`.

---

## Examples

### Basic Workflow

```bash
# 1. Initialize configuration
$ prela init
Service name [my-agent]: my-chatbot
Exporter (console/file) [file]: file
Trace directory [./traces]: ./traces
Sample rate (0.0-1.0) [1.0]: 1.0

# 2. Run your application (generates traces)
$ python my_chatbot.py

# 3. List recent traces
$ prela trace list --limit 5

# 4. Show detailed trace
$ prela trace show trace-abc123

# 5. Search for specific traces
$ prela trace search error
```

### Development Workflow

```bash
# Use console exporter for development
$ prela init
Exporter (console/file) [file]: console

# Run your app with live trace output
$ python my_app.py
```

### Production Monitoring

```bash
# Sample 10% of traces in production
$ prela init
Sample rate (0.0-1.0) [1.0]: 0.1

# List traces from last hour
$ prela trace list --since 1h

# Search for errors
$ prela trace search error
```

---

## Troubleshooting

### "CLI dependencies not installed"

Install CLI dependencies:
```bash
pip install prela[cli]
```

Or install individually:
```bash
pip install typer pyyaml rich
```

### "No traces found"

Check:
1. Trace directory exists and contains `.jsonl` files
2. `.prela.yaml` points to correct `trace_dir`
3. Application is actually generating traces

### "Configuration saved" but file not found

Check that you're in the correct directory. `.prela.yaml` is created in the current working directory.

---

## Architecture

The CLI is built with:
- **Typer** - Modern CLI framework with type hints
- **Rich** - Beautiful terminal output (tables, trees, colors)
- **PyYAML** - YAML configuration parsing

**File Structure:**
```
prela/contrib/
├── __init__.py
├── cli.py          # CLI implementation
└── CLI.md          # This documentation

tests/
└── test_cli.py     # CLI tests (28 tests)
```

**Entry Point:**
Registered in `pyproject.toml`:
```toml
[project.scripts]
prela = "prela.contrib.cli:main"
```

---

## Future Enhancements

Planned for Phase 1 (Months 4-8):
- `prela serve` - Local web dashboard
- `prela eval` - Evaluation suite runner
- `prela export` - Export traces to different formats
- `prela stats` - Aggregate statistics and metrics
- `prela compare` - Compare traces side-by-side

Planned for Phase 2:
- `prela cloud push` - Push traces to cloud platform
- `prela cloud sync` - Sync local and cloud traces
- `prela alert` - Configure alerting rules
- `prela team` - Team collaboration features
