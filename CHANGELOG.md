# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-04

### Added

**Core Observability:**
- Complete tracing SDK with span lifecycle management
- Context propagation (async/threading safe)
- Multiple sampling strategies (always on/off, probabilistic, rate limiting)
- Console, file, HTTP, and ClickHouse exporters

**Auto-Instrumentation (10 frameworks):**
- OpenAI SDK instrumentation
- Anthropic/Claude SDK instrumentation
- LangChain chains and agents
- LlamaIndex queries
- CrewAI task-based agents
- AutoGen conversational agents
- LangGraph graph-based workflows
- OpenAI Swarm handoff-based agents
- N8N webhook and code node integration

**Evaluation Framework:**
- 17+ assertion types for testing agent behavior
- Structural assertions (contains, regex, JSON schema, length)
- Semantic assertions (embedding-based similarity with fallback)
- Tool assertions (tool called, args validation, sequence checking)
- Multi-agent assertions (agent usage, task completion, delegation patterns)
- 3 reporter types (Console, JSON, JUnit)
- Parallel test execution with progress callbacks
- N8N-specific evaluation support

**Replay Engine:**
- Deterministic replay for debugging
- API retry with exponential backoff
- Tool re-execution with allowlist/blocklist controls
- Retrieval re-execution with ChromaDB support
- Semantic output comparison with difflib fallback
- Streaming support for OpenAI and Anthropic APIs

**CLI Tools:**
- `prela list` - List traces with filters
- `prela show` - Show trace details
- `prela last` - Show most recent trace
- `prela search` - Full-text search
- `prela errors` - Show failed traces
- `prela tail` - Real-time trace following
- Interactive TUI explorer with Textual

**Testing & Quality:**
- 836+ comprehensive tests
- 95%+ code coverage
- Full type hints (mypy strict mode)
- Linting with ruff and black
- CI/CD with GitHub Actions

### Technical Details

- **Python Support:** 3.9, 3.10, 3.11, 3.12
- **Build System:** Hatchling
- **License:** Apache 2.0
- **Package Name:** `prela`

[0.1.0]: https://github.com/garrettw2200/prela-sdk/releases/tag/v0.1.0
