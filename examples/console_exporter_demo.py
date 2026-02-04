"""Demo of ConsoleExporter tree visualization with verbosity levels."""

import time

from prela.core.clock import now
from prela.core.span import Span, SpanEvent, SpanStatus, SpanType
from prela.exporters.console import ConsoleExporter

# Create a complex span tree simulating an AI agent workflow
print("=" * 80)
print("Console Exporter Demo - Tree Visualization")
print("=" * 80)

# Build trace
trace_id = "demo-trace-001"
spans = []

# Root: Agent orchestration
agent_span = Span(
    span_id="span-1",
    trace_id=trace_id,
    parent_span_id=None,
    name="research_agent",
    span_type=SpanType.AGENT,
    started_at=now(),
    attributes={"agent.task": "research AI developments"},
)
time.sleep(0.1)

# Child 1: LLM call for planning
llm_plan_span = Span(
    span_id="span-2",
    trace_id=trace_id,
    parent_span_id="span-1",
    name="gpt-4-planning",
    span_type=SpanType.LLM,
    started_at=now(),
    attributes={
        "llm.model": "gpt-4",
        "llm.input_tokens": 150,
        "llm.output_tokens": 89,
        "llm.temperature": 0.7,
    },
)
llm_plan_span.add_event(
    SpanEvent(
        timestamp=now(),
        name="llm.request",
        attributes={"prompt": "Plan research steps for AI developments"},
    )
)
time.sleep(0.05)
llm_plan_span.end()
spans.append(llm_plan_span)

# Child 2: Web search tool
search_span = Span(
    span_id="span-3",
    trace_id=trace_id,
    parent_span_id="span-1",
    name="web_search",
    span_type=SpanType.TOOL,
    started_at=now(),
    attributes={
        "tool.name": "web_search",
        "tool.input": "AI developments 2024",
    },
)
time.sleep(0.08)
search_span.end()
spans.append(search_span)

# Child 3: Document retrieval
retrieval_span = Span(
    span_id="span-4",
    trace_id=trace_id,
    parent_span_id="span-1",
    name="vector_retrieval",
    span_type=SpanType.RETRIEVAL,
    started_at=now(),
    attributes={
        "retriever.query": "Recent AI breakthroughs",
        "retriever.document_count": 5,
    },
)
time.sleep(0.06)
retrieval_span.end()
spans.append(retrieval_span)

# Child 4: Embedding generation for retrieved docs
embedding_span = Span(
    span_id="span-5",
    trace_id=trace_id,
    parent_span_id="span-4",
    name="text-embedding",
    span_type=SpanType.EMBEDDING,
    started_at=now(),
    attributes={
        "embedding.model": "text-embedding-ada-002",
        "embedding.dimensions": 1536,
    },
)
time.sleep(0.03)
embedding_span.end()
spans.append(embedding_span)

# Child 5: Final LLM call for synthesis
llm_synth_span = Span(
    span_id="span-6",
    trace_id=trace_id,
    parent_span_id="span-1",
    name="gpt-4-synthesis",
    span_type=SpanType.LLM,
    started_at=now(),
    attributes={
        "llm.model": "gpt-4",
        "llm.input_tokens": 450,
        "llm.output_tokens": 256,
        "llm.temperature": 0.3,
    },
)
llm_synth_span.add_event(
    SpanEvent(
        timestamp=now(),
        name="llm.request",
        attributes={"context": "Synthesize research findings"},
    )
)
time.sleep(0.12)
llm_synth_span.end()
spans.append(llm_synth_span)

# Child 6: Error case - API rate limit
error_span = Span(
    span_id="span-7",
    trace_id=trace_id,
    parent_span_id="span-1",
    name="additional-search",
    span_type=SpanType.TOOL,
    started_at=now(),
    attributes={"tool.name": "web_search", "tool.input": "Additional context"},
)
time.sleep(0.02)
error_span.set_status(SpanStatus.ERROR, "API rate limit exceeded")
error_span.end()
spans.append(error_span)

# End root span
agent_span.end()
spans.insert(0, agent_span)

# Demo 1: Minimal verbosity
print("\n")
print("=" * 80)
print("Demo 1: Minimal Verbosity (name + duration + status only)")
print("=" * 80)
exporter_minimal = ConsoleExporter(
    verbosity="minimal", color=False, show_timestamps=True
)
exporter_minimal.export(spans)

# Demo 2: Normal verbosity (default)
print("\n")
print("=" * 80)
print("Demo 2: Normal Verbosity (+ key attributes)")
print("=" * 80)
exporter_normal = ConsoleExporter(verbosity="normal", color=False, show_timestamps=True)
exporter_normal.export(spans)

# Demo 3: Verbose verbosity
print("\n")
print("=" * 80)
print("Demo 3: Verbose Verbosity (+ all attributes + events)")
print("=" * 80)
exporter_verbose = ConsoleExporter(
    verbosity="verbose", color=False, show_timestamps=True
)
exporter_verbose.export(spans)

# Demo 4: With colors (if rich is available)
print("\n")
print("=" * 80)
print("Demo 4: With Colors (requires 'rich' library)")
print("=" * 80)
exporter_color = ConsoleExporter(verbosity="normal", color=True, show_timestamps=True)
exporter_color.export(spans)

print("\n")
print("=" * 80)
print("Demo Complete!")
print("=" * 80)
print("\nTry running with different verbosity levels:")
print("  verbosity='minimal'  - Compact, overview only")
print("  verbosity='normal'   - Shows key attributes (default)")
print("  verbosity='verbose'  - Shows everything including events")
print("\nInstall 'rich' for colored output: pip install rich")
