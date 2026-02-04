# SDK vs Cloud Features - Complete Reference

**Last Updated:** January 2026
**Purpose:** Track feature distribution between free SDK and paid Cloud platform

---

## Free SDK Features (Open Source, Apache 2.0)

### Core Observability

#### ✅ Auto-Instrumentation (10 Frameworks)
**What:** One-line setup with zero configuration
```python
import prela
prela.init()  # Automatically detects and instruments all frameworks
```

**Supported Frameworks:**
- **LLM Providers:** OpenAI, Anthropic
- **Agent Frameworks:** LangChain, LlamaIndex
- **Multi-Agent Systems:** CrewAI, AutoGen, LangGraph, Swarm
- **Workflow Automation:** n8n (webhook + code node helpers)

**Technical Details:**
- Zero code changes required
- Thread-safe, async-safe
- Defensive programming (never crashes user code)
- Auto-detection via package registry

**Value Proposition:** "Works with everything you're already using"

---

#### ✅ Trace Capture & Context Propagation
**What:** Automatic span creation with full context

**Captured Data:**
- Span hierarchy (parent-child relationships)
- Timestamps (microsecond precision)
- Duration (monotonic time for accuracy)
- Status (SUCCESS, ERROR, PENDING)
- Attributes (key-value metadata)
- Events (timestamped log points)
- Stack traces on errors

**Context Features:**
- Thread-safe context propagation
- Async/await support
- Thread pool compatibility (with helper)
- Baggage for cross-cutting concerns

**Value Proposition:** "See every decision your AI agent makes"

---

#### ✅ Deterministic Replay (Industry First)
**What:** Capture LLM executions and replay with modifications

**Replay Capabilities:**
```python
from prela.replay import ReplayEngine

engine = ReplayEngine(trace)
result = engine.replay_with_modifications(
    model="gpt-4o",           # Change model
    temperature=0.7,          # Adjust temperature
    max_tokens=2000,          # Modify limits
    compare=True              # Side-by-side comparison
)
```

**Features:**
- Model switching (GPT-4 → Claude → Llama)
- Parameter modification (temperature, max_tokens)
- Prompt editing
- Tool re-execution (allowlist/blocklist controls)
- Retrieval re-execution (vector DB support)
- API retry logic (exponential backoff)
- Semantic similarity comparison (difflib or sentence-transformers)

**Supported Frameworks:**
- OpenAI, Anthropic (LLM calls)
- LangChain, LlamaIndex (chains, agents, retrievers)
- CrewAI, AutoGen, LangGraph, Swarm (multi-agent)
- n8n code nodes (custom Python)

**Value Proposition:** "Debug production issues in development"

---

#### ✅ Evaluation Framework
**What:** Test non-deterministic AI behavior systematically

**17 Assertion Types:**

**Structural Assertions:**
- `ContainsAssertion` - Text contains substring
- `NotContainsAssertion` - Text doesn't contain substring
- `RegexAssertion` - Regex pattern match
- `LengthAssertion` - Output length constraints
- `JSONValidAssertion` - Valid JSON structure
- `JSONSchemaAssertion` - JSON schema validation

**Tool Assertions:**
- `ToolCalledAssertion` - Tool was invoked
- `ToolNotCalledAssertion` - Tool wasn't invoked
- `ToolArgsAssertion` - Tool called with specific args

**Semantic Assertions:**
- `SemanticSimilarityAssertion` - Embeddings-based similarity
- `SentimentAssertion` - Sentiment analysis (positive/negative/neutral)

**Multi-Agent Assertions:**
- `AgentUsedAssertion` - Agent was invoked
- `TaskCompletedAssertion` - Task completed successfully
- `DelegationOccurredAssertion` - Agent delegated to another
- `HandoffOccurredAssertion` - Swarm-style handoff
- `AgentCollaborationAssertion` - Minimum agents participated
- `ConversationTurnsAssertion` - Conversation length bounds
- `NoCircularDelegationAssertion` - Detect delegation loops

**Execution Modes:**
```python
from prela.evals import EvalRunner, EvalSuite

runner = EvalRunner(
    suite=my_suite,
    agent=my_agent,
    parallel=True,          # Parallel execution
    max_workers=4,          # Configurable workers
    tracer=tracer,          # Optional tracing
    on_case_complete=callback  # Progress callback
)

result = runner.run()
```

**Reporters:**
- `ConsoleReporter` - Beautiful terminal output (rich library)
- `JSONReporter` - Structured data export
- `JUnitReporter` - CI/CD integration (Jenkins, GitHub Actions)

**n8n Workflow Testing:**
- `N8nWorkflowEvalRunner` - Async workflow execution
- `N8nEvalCase` - Workflow-specific test cases
- Node-level and workflow-level assertions

**Value Proposition:** "Test AI agents like you test regular code"

---

#### ✅ Local Exporters
**What:** Export traces to local destinations

**Console Exporter:**
```python
prela.init(exporter="console")
```
- Pretty-printed JSON (configurable indentation)
- Rich library integration (colored output)
- Tree visualization option
- Perfect for development

**File Exporter:**
```python
prela.init(
    exporter="file",
    file_path="traces.jsonl",
    max_file_size_mb=100,    # Rotation
    max_files=10             # Keep last 10
)
```
- JSONL format (one trace per line)
- Automatic file rotation by size
- Thread-safe writes
- Append mode preserves history

**OTLP Exporter:**
```python
prela.init(
    exporter="otlp",
    otlp_endpoint="http://localhost:4317"
)
```
- OpenTelemetry Protocol support
- Send to Jaeger, Tempo, Honeycomb, etc.
- Standard observability integration

**Multi Exporter:**
```python
from prela.exporters import ConsoleExporter, FileExporter, MultiExporter

prela.init(
    exporter=MultiExporter([
        ConsoleExporter(),
        FileExporter("traces.jsonl")
    ])
)
```
- Fanout to multiple destinations
- Combine local + remote

**Value Proposition:** "Works offline, no cloud required"

---

#### ✅ CLI Tool
**What:** Command-line interface for trace management

**Commands:**
```bash
# Initialize project
prela init

# List traces from file
prela list traces.jsonl

# Show detailed trace
prela show <trace_id> --file traces.jsonl

# Search traces
prela search "error" --file traces.jsonl

# Run evaluations
prela eval run tests.yaml

# Replay with modifications
prela replay trace.json --model gpt-4o --temperature 0.7 --compare

# Streaming replay
prela replay trace.json --model claude-sonnet-4 --stream
```

**Features:**
- File-based trace management
- Search functionality
- Evaluation execution
- Replay with parameter overrides
- Streaming output support
- Beautiful terminal output

**Value Proposition:** "Powerful CLI for local workflows"

---

#### ✅ Custom Tracing
**What:** Trace custom functions and code

**@trace Decorator:**
```python
from prela import trace

@trace(name="custom_function", span_type="tool")
def my_function(x, y):
    return x + y
```

**Manual Spans:**
```python
from prela import get_tracer

tracer = get_tracer()

with tracer.span("operation", span_type="custom") as span:
    span.set_attribute("key", "value")
    span.add_event("milestone", {"step": 1})
    # ... work ...
```

**Value Proposition:** "Trace anything, not just LLM calls"

---

#### ✅ Sampling & Filtering
**What:** Control which traces to capture

**Sampling Strategies:**
```python
# Sample everything (development)
prela.init(sample_rate=1.0)

# Sample 10% (production)
prela.init(sample_rate=0.1)

# Sample nothing (disable)
prela.init(sample_rate=0.0)
```

**Samplers:**
- `AlwaysOnSampler` - Capture all traces
- `AlwaysOffSampler` - Disable tracing
- `ProbabilitySampler` - Hash-based deterministic sampling
- `RateLimitingSampler` - Token bucket rate limiting

**Value Proposition:** "Control overhead and costs"

---

#### ✅ Complete Documentation
**What:** 34 comprehensive documentation pages

**Contents:**
- Getting Started (installation, quickstart, configuration)
- Concepts (tracing, spans, context, sampling, exporters)
- Integrations (OpenAI, Anthropic, LangChain, LlamaIndex, multi-agent, n8n)
- Evaluation Framework (writing tests, assertions, running, CI/CD)
- CLI Reference (all commands, options, examples)
- Examples (basic, custom, parallel, production)
- API Reference (auto-generated from docstrings)
- Contributing guide
- Changelog

**Formats:**
- MkDocs Material site (beautiful, searchable)
- Mermaid diagrams (architecture, flows)
- Copy-paste ready code examples
- 150+ working examples

**Value Proposition:** "Everything you need to get started"

---

### SDK Summary

**Total Features:** 12 major components
**Total Tests:** 1,068 (100% passing)
**Test Coverage:** 95%+
**Lines of Code:** ~15,000
**License:** Apache 2.0 (free forever)
**Installation:** `pip install prela`
**Setup Time:** 30 seconds

**Core Value:** Complete local observability with no cloud dependency

---

## Cloud Platform Features (Paid Tiers)

### Infrastructure (Already Built)

#### 🔧 Backend Services
**Architecture:**
```
SDK → Ingest Gateway → Kafka → Trace Service → ClickHouse
                                                     ↓
Frontend ← API Gateway ←────────────────────────────┘
```

**Components:**
- **Ingest Gateway** (FastAPI) - High-throughput HTTP ingestion
- **API Gateway** (FastAPI) - Query API for frontend
- **Trace Service** (Python worker) - Kafka consumer → ClickHouse writer
- **ClickHouse Cloud** - Columnar database (90-day retention)
- **Upstash Kafka** - Serverless message queue
- **Redis** - Real-time pub/sub for WebSocket updates
- **PostgreSQL** - User data, projects, auth

**Hosting:** Railway (production-ready, auto-scaling)

---

### Cloud-Only Features

#### 💰 1. Centralized Storage & Retention
**What:** Cloud storage with automatic retention management

**Features:**
- **Unlimited traces** (within tier limits)
- **90-day automatic retention** (ClickHouse TTL)
- **10x compression** (ClickHouse columnar storage)
- **No manual cleanup** (automatic expiration)
- **Cross-team access** (shared trace repository)

**vs. Free SDK:**
- SDK: Local files, manual deletion, ~10-100 GB limits
- Cloud: Centralized, automatic TTL, unlimited storage

**Tiers:**
- Free: 10,000 traces/month (30-day retention)
- Pro: 100,000 traces/month (90-day retention)
- Team: 1M traces/month (90-day retention)
- Enterprise: Unlimited (custom retention: 1-2 years)

**Value:** "Never lose production traces, automatic compliance"

---

#### 🔍 2. Advanced Search & Filtering
**What:** Fast full-text search across all traces

**Search Capabilities:**
```sql
-- Find all traces matching criteria
WHERE service_name LIKE '%prod%'
  AND status = 'ERROR'
  AND started_at > NOW() - INTERVAL 7 DAYS
  AND attributes LIKE '%timeout%'
```

**Features:**
- **Full-text search** (search within attributes, events)
- **Time-range queries** (last hour, day, week, month)
- **Status filtering** (success, error, pending)
- **Framework filtering** (OpenAI, CrewAI, n8n, etc.)
- **Duration filtering** (>500ms, <100ms)
- **Token filtering** (>1000 tokens, cost >$0.10)
- **Saved searches** (reusable filter combinations)

**Performance:**
- ClickHouse: 50-300ms queries
- vs. Local grep: 10-30 seconds

**Value:** "Find any trace instantly, not after 30 minutes of grepping"

---

#### 📊 3. Analytics & Dashboards
**What:** Pre-built dashboards and custom analytics

**n8n Multi-Tenant Dashboard:** (Already Built)
- Workflow list with execution counts
- Execution timeline visualization (proportional bars)
- AI node detection and metrics
- Real-time updates via WebSocket (<1ms latency)
- Project-based isolation (prod/staging/dev)
- Quick setup card (3-step onboarding)

**Multi-Agent Execution Dashboard:** (Backend Ready)
- Agent communication graphs (nodes + edges)
- Task tracking and completion status
- Delegation flow visualization
- Performance metrics per agent
- Support for CrewAI, AutoGen, LangGraph, Swarm

**Cost Analytics:**
```sql
-- Token usage by project
SELECT
    service_name,
    SUM(JSONExtractInt(attributes, 'total_tokens')) as total_tokens,
    SUM(JSONExtractFloat(attributes, 'cost_usd')) as total_cost
FROM spans
WHERE span_type = 'LLM'
  AND started_at > NOW() - INTERVAL 30 DAYS
GROUP BY service_name
```

**Analytics Features:**
- Token usage trends (daily/weekly/monthly)
- Cost breakdown by model (GPT-4, Claude, etc.)
- Latency percentiles (p50, p95, p99)
- Error rate monitoring
- Agent performance metrics
- Custom time ranges

**Value:** "Executive dashboard showing AI costs and reliability"

---

#### 🚨 4. Real-Time Alerts & Notifications
**What:** Proactive monitoring with instant notifications

**Alert Types:**
- **Error alerts** - Agent failures (status = ERROR)
- **Performance alerts** - Latency >5 seconds
- **Cost alerts** - Token budget exceeded ($100/day)
- **Volume alerts** - Trace volume spike (10x normal)
- **Custom alerts** - User-defined conditions

**Notification Channels:**
- Email (built-in)
- Slack (webhook integration)
- Discord (webhook integration)
- PagerDuty (enterprise)
- Custom webhooks (JSON POST)

**Alert Configuration:**
```yaml
alert:
  name: "Agent Failures"
  condition: "status = ERROR AND framework = 'crewai'"
  threshold: 5 failures in 10 minutes
  channels: ["slack", "email"]
```

**Real-Time Engine:**
- Kafka → Redis Pub/Sub → WebSocket → Frontend
- <1ms latency from event to notification
- Background processing (doesn't block ingestion)

**Value:** "Know when things break, before users complain"

---

#### 👥 5. Team Collaboration
**What:** Multi-user access with project isolation

**Features:**
- **Project-based isolation** (prod-n8n, staging-api, dev-agents)
- **Team member invitations** (email-based)
- **Shared trace access** (entire team sees all traces)
- **Role-based permissions** (future: admin, viewer, developer)
- **Comment on traces** (future: annotate failures)
- **Share trace links** (permalink to specific trace)

**Project Structure:**
```
Organization: ACME Corp
├─ Project: prod-n8n (5 team members)
│  └─ 50,000 traces/month
├─ Project: staging-api (3 team members)
│  └─ 10,000 traces/month
└─ Project: dev-agents (2 team members)
   └─ 5,000 traces/month
```

**Value:** "Stop Slacking trace files back and forth"

---

#### 🎬 6. Replay UI (Backend Ready)
**What:** Visual interface for deterministic replay

**Features:**
- **Browse traces** with replay capability indicator
- **One-click replay** with parameter overrides
- **Side-by-side comparison** (original vs replayed)
- **Execution history** (track all replay attempts)
- **Save configurations** (reusable replay params)
- **Batch replay** (test multiple parameter combinations)

**UI Flow:**
```
1. Browse traces → Filter by "has replay data"
2. Click trace → See original execution
3. Click "Replay" → Modal with parameter overrides
4. Submit → Background execution
5. View results → Side-by-side comparison
```

**Backend API (Complete):**
- `GET /api/v1/replay/traces` - List replayable traces
- `POST /api/v1/replay/execute` - Trigger replay (background task)
- `GET /api/v1/replay/executions/{id}` - Get execution status
- `GET /api/v1/replay/executions/{id}/comparison` - Get comparison results

**Value:** "Click to replay, don't copy 20 lines of code"

---

#### 🔗 7. HTTP Exporter (SDK Feature for Cloud)
**What:** Send traces to cloud platform

**SDK Configuration:**
```python
import prela

prela.init(
    service_name="my-agent",
    exporter="http",
    http_endpoint="https://ingest.prela.app/v1/traces",
    http_api_key="prela_sk_abc123..."
)
```

**Features:**
- Automatic batching (100 spans per request)
- Retry logic (exponential backoff)
- Gzip compression (reduce bandwidth)
- Async sending (non-blocking)
- API key authentication

**Value:** "One line of code to enable cloud sync"

---

#### 📈 8. API Access
**What:** Programmatic access to traces

**Endpoints:**
- `GET /api/v1/traces` - List traces
- `GET /api/v1/traces/{id}` - Get trace details
- `GET /api/v1/search` - Search traces
- `GET /api/v1/analytics/tokens` - Token usage stats
- `GET /api/v1/analytics/costs` - Cost breakdown
- `GET /api/v1/n8n/workflows` - n8n workflows
- `GET /api/v1/multi-agent/executions` - Multi-agent executions

**Authentication:**
```bash
curl https://api.prela.app/v1/traces \
  -H "Authorization: Bearer prela_sk_abc123..."
```

**Use Cases:**
- Custom dashboards (embed in internal tools)
- Data export (compliance, backup)
- Automation (trigger actions on failures)
- Integration (send to data warehouse)

**Value:** "Build custom workflows on top of Prela"

---

#### 🔐 9. Compliance & Governance
**What:** Enterprise-grade compliance features

**Features:**
- **90-day audit trails** (automatic retention)
- **Exportable reports** (JSON, CSV for auditors)
- **SOC 2 compliance** (in progress)
- **Data encryption** (at rest and in transit)
- **GDPR compliance** (data deletion on request)
- **Access logs** (who viewed which traces)
- **Retention policies** (configurable per project)

**Compliance Questions Answered:**
- "Why did the agent approve this loan?" → Show full trace
- "What data did the agent see?" → Attributes in span
- "When did this decision happen?" → Timestamp
- "Who ran this agent?" → Service name, user ID

**Value:** "Pass audits without scrambling"

---

#### ⚡ 10. High-Throughput Ingestion
**What:** Handle production-scale traffic

**Kafka Benefits:**
- **100,000+ traces/second** capacity
- **Zero dropped traces** during bursts
- **Automatic buffering** (7-day retention)
- **Horizontal scaling** (add more workers)
- **Decoupled services** (ingestion vs. storage)

**Real-World Scenario:**
```
9:00 AM: 500 n8n workflows start (scheduled jobs)
         → 5,000 traces in 10 seconds
         → Kafka buffers, Trace Service processes over 2 minutes
         → ClickHouse writes at controlled rate
         → All traces saved, zero dropped
```

**vs. Direct Database:**
```
Same scenario without Kafka:
         → Database overwhelmed
         → Connections exhausted
         → Traces dropped
         → System crashes
```

**Value:** "Production-ready reliability at any scale"

---

### Cloud Summary

**Total Cloud Features:** 10 major capabilities
**Backend Status:** 100% complete (ready for deployment)
**Frontend Status:** 50% complete (n8n dashboard done, multi-agent + replay UI pending)
**Infrastructure:** Railway + ClickHouse + Kafka + Redis + PostgreSQL
**Estimated Hosting Cost:** $25-115/month (scales with usage)

**Core Value:** Team collaboration + analytics + reliability + compliance

---

## Feature Comparison Table

| Feature | Free SDK | Cloud Free | Cloud Pro | Cloud Team | Enterprise |
|---------|----------|------------|-----------|------------|------------|
| **Auto-Instrumentation** | ✅ All 10 | ✅ All 10 | ✅ All 10 | ✅ All 10 | ✅ All 10 |
| **Deterministic Replay** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Evaluation Framework** | ✅ 17 assertions | ✅ 17 assertions | ✅ 17 assertions | ✅ 17 assertions | ✅ 17 assertions |
| **Local Exporters** | ✅ 4 exporters | ✅ 4 exporters | ✅ 4 exporters | ✅ 4 exporters | ✅ 4 exporters |
| **CLI Tool** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Documentation** | ✅ 34 pages | ✅ 34 pages | ✅ 34 pages | ✅ 34 pages | ✅ 34 pages |
| | | | | | |
| **Trace Storage** | Local files | 10k/mo | 100k/mo | 1M/mo | Unlimited |
| **Retention** | Manual | 30 days | 90 days | 90 days | Custom (1-2yr) |
| **Cloud Dashboard** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Search & Filter** | grep | ✅ Basic | ✅ Advanced | ✅ Advanced | ✅ Advanced |
| **Analytics** | ❌ | ❌ | ✅ | ✅ | ✅ Custom |
| **Alerts** | ❌ | ❌ | ✅ Email | ✅ Slack/Discord | ✅ PagerDuty |
| **Team Access** | ❌ | ❌ | ✅ 3 users | ✅ 15 users | ✅ Unlimited |
| **API Access** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Replay UI** | CLI only | ❌ | ✅ | ✅ | ✅ |
| **n8n Dashboard** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Multi-Agent Dashboard** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Support** | Community | Community | Email | Priority Email | Dedicated |
| **SLA** | ❌ | ❌ | ❌ | ❌ | ✅ 99.9% |
| **Self-Hosted** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **SSO/SAML** | ❌ | ❌ | ❌ | ❌ | ✅ |
| | | | | | |
| **Price** | **$0** | **$0** | **$29/mo** | **$99/mo** | **Custom** |

---

## Upgrade Triggers (When Users Convert)

### Free SDK → Cloud Free Tier
**Trigger:** "Try the dashboard, see what we built"
- User installs SDK
- Uses local exporters for 1-2 weeks
- Sees "Upgrade to Cloud" banner in CLI
- Clicks to sign up for free tier

### Cloud Free → Cloud Pro
**Trigger:** "Hit the 10k trace limit" OR "Need team access"
- User approaches 8,000/10,000 traces in month
- Dashboard shows warning: "2,000 traces remaining"
- Email: "You're 80% through your free tier"
- Click to upgrade → enters payment info

### Cloud Pro → Cloud Team
**Trigger:** "Outgrow 100k traces" OR "Need advanced features"
- Team is sending 120k traces/month (over limit)
- Need more than 3 team members
- Want Slack/Discord alerts (not just email)
- Want advanced analytics (cost dashboard)

### Cloud Team → Enterprise
**Trigger:** "Compliance requirements" OR "Scale needs"
- Need self-hosted deployment (security policy)
- Need SSO/SAML (company requirement)
- Need custom retention (2 years for compliance)
- Need SLA (99.9% uptime guarantee)
- Want dedicated support (Slack channel)

---

## Development Priorities

### SDK (100% Complete) ✅
- All 10 framework integrations
- Deterministic replay
- Evaluation framework
- Local exporters
- CLI tool
- Documentation
- **Status:** Ready for PyPI publication

### Cloud Backend (100% Complete) ✅
- Ingest Gateway
- API Gateway
- Trace Service
- ClickHouse schema
- Kafka topics
- Redis pub/sub
- n8n API endpoints (5)
- Multi-agent API endpoints (6)
- Replay API endpoints (6)
- **Status:** Ready for Railway deployment

### Cloud Frontend (50% Complete) ⏳
- ✅ n8n Dashboard (WorkflowList, WorkflowDetail, ExecutionTimeline)
- ✅ Real-time WebSocket updates
- ✅ Quick setup onboarding
- ⏳ Multi-Agent Dashboard (backend ready, frontend pending)
- ⏳ Replay UI (backend ready, frontend pending)
- ⏳ Analytics Dashboard (cost tracking, token usage)
- ⏳ Alerts Configuration UI
- ⏳ Team Management UI

### Priority for Launch
1. **Immediate:** Publish SDK to PyPI (marketing)
2. **Week 1:** Complete multi-agent dashboard frontend
3. **Week 2:** Complete replay UI frontend
4. **Week 3:** Deploy to Railway, open beta
5. **Week 4:** Collect feedback, iterate

---

## Revenue Model

### Customer Journey
```
Day 1:    Install SDK (pip install prela)
Day 7:    See value (debugging with local files)
Day 30:   Hit limits (team needs central storage)
Day 31:   Sign up for Cloud Free (try dashboard)
Day 45:   Hit 10k trace limit
Day 46:   Upgrade to Pro ($29/mo) 💰
Month 3:  Outgrow 100k traces OR need team features
Month 4:  Upgrade to Team ($99/mo) 💰
Year 1:   50+ engineers, compliance needs
Year 2:   Enterprise contract ($2,000/mo) 💰
```

### Financial Projections

**Year 1:**
- 1,000 SDK users (free)
- 100 Cloud Free users (freemium activation)
- 50 Pro customers ($29/mo × 50 = $1,450/mo)
- 10 Team customers ($99/mo × 10 = $990/mo)
- **Total MRR:** $2,440/mo
- **Total ARR:** ~$29k

**Year 2:**
- 10,000 SDK users (viral growth)
- 1,000 Cloud Free users
- 200 Pro customers ($29/mo × 200 = $5,800/mo)
- 50 Team customers ($99/mo × 50 = $4,950/mo)
- 5 Enterprise customers ($2k/mo × 5 = $10,000/mo)
- **Total MRR:** $20,750/mo
- **Total ARR:** ~$249k

**Year 3:**
- 50,000 SDK users
- 5,000 Cloud Free users
- 500 Pro customers ($14,500/mo)
- 100 Team customers ($9,900/mo)
- 20 Enterprise customers ($40,000/mo)
- **Total MRR:** $64,400/mo
- **Total ARR:** ~$773k

---

## Competitive Moat

### Why Prela Wins

**vs. LangSmith:**
- ✅ Framework agnostic (10 vs. 1)
- ✅ Deterministic replay (unique)
- ✅ Open source SDK (trust)
- ✅ Multi-agent native (4 frameworks)

**vs. Weights & Biases:**
- ✅ Built for agents (not ML training)
- ✅ Auto-instrumentation (not manual)
- ✅ Deterministic replay (unique)
- ✅ Lower pricing ($29 vs. $50)

**vs. Arize:**
- ✅ Developer-first (not data science)
- ✅ Multi-framework (not single)
- ✅ Evaluation framework (built-in)
- ✅ Open source (community-owned)

**vs. Custom Logging:**
- ✅ Zero setup time (30 sec vs. weeks)
- ✅ Complete feature set (not DIY)
- ✅ Maintained and updated (not abandoned)
- ✅ Team collaboration (not siloed)

### The Moat
1. **Deterministic Replay** - Industry first, hard to copy
2. **Multi-Framework** - Network effect (more integrations = more value)
3. **Open Source** - Community contributions, trust
4. **Multi-Agent Native** - Future-proof (agents are the trend)

---

## Next Steps

### Pre-Launch
- [ ] Publish SDK to PyPI
- [ ] Deploy MkDocs documentation site
- [ ] Create GitHub repository (public)
- [ ] Set up Discord community
- [ ] Prepare Product Hunt listing

### Launch Day
- [ ] Post on Product Hunt, Hacker News, Reddit
- [ ] Announce on Twitter/X
- [ ] Email dev newsletters
- [ ] Monitor feedback channels

### Post-Launch
- [ ] Deploy cloud backend to Railway
- [ ] Complete multi-agent dashboard frontend
- [ ] Complete replay UI frontend
- [ ] Open cloud beta (free tier)
- [ ] Iterate based on user feedback

---

**Document Version:** 1.0
**Last Updated:** January 29, 2026
**Maintained By:** Product Team
