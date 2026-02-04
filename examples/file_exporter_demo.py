"""Demo of FileExporter with advanced features."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from prela.core.span import Span, SpanType
from prela.exporters.file import FileExporter

# Create a file exporter with rotation
print("=" * 60)
print("FileExporter Demo - Advanced Features")
print("=" * 60)

# Setup: Create exporter with 1MB rotation
export_dir = Path("./demo_traces")
exporter = FileExporter(
    directory=export_dir,
    format="jsonl",
    max_file_size_mb=1,
    rotate=True
)

print(f"\n✓ Created FileExporter in: {export_dir}")

# 1. Export some traces
print("\n1. Exporting traces...")
traces = []
for i in range(5):
    span = Span(
        name=f"llm-call-{i}",
        span_type=SpanType.LLM,
        trace_id=f"trace-{i}",
        span_id=f"span-{i}",
    )
    span.set_attribute("model", "claude-sonnet-4")
    span.set_attribute("tokens", 100 + i * 10)
    span.end()
    exporter.export([span])
    traces.append(span)

print(f"✓ Exported {len(traces)} spans")

# 2. List files created
files = list(export_dir.glob("traces-*.jsonl"))
print(f"\n2. Files created: {len(files)}")
for f in files:
    print(f"   - {f.name} ({f.stat().st_size} bytes)")

# 3. Find a specific trace
print("\n3. Finding trace-2...")
trace_file = exporter.get_trace_file("trace-2")
if trace_file:
    print(f"✓ Found in: {trace_file.name}")
else:
    print("✗ Not found")

# 4. Read all traces back
print("\n4. Reading all traces...")
read_spans = list(exporter.read_traces())
print(f"✓ Read {len(read_spans)} spans")
for span in read_spans:
    print(f"   - {span.name} (trace: {span.trace_id})")

# 5. Read specific trace
print("\n5. Reading only trace-1...")
trace_1_spans = list(exporter.read_traces(trace_id="trace-1"))
print(f"✓ Found {len(trace_1_spans)} spans for trace-1")

# 6. List traces by date range
print("\n6. Listing traces in last 24 hours...")
now = datetime.now(timezone.utc)
start = now - timedelta(hours=24)
end = now
trace_ids = exporter.list_traces(start, end)
print(f"✓ Found {len(trace_ids)} unique traces")
for tid in trace_ids:
    print(f"   - {tid}")

# 7. Demonstrate cleanup (create old file)
print("\n7. Testing cleanup...")
old_date = now - timedelta(days=10)
old_file = export_dir / f"traces-{old_date.strftime('%Y-%m-%d')}-001.jsonl"
old_file.write_text('{"trace_id": "old-trace"}\n')
print(f"✓ Created old file: {old_file.name}")

deleted = exporter.cleanup_old_traces(days=7)
print(f"✓ Deleted {deleted} files older than 7 days")

if not old_file.exists():
    print("✓ Old file was removed")

# 8. Show rotation with large data
print("\n8. Testing rotation with large spans...")
exporter_rotate = FileExporter(
    directory=export_dir / "rotated",
    max_file_size_mb=0.001,  # Very small to trigger rotation
    rotate=True
)

for i in range(30):
    span = Span(
        name=f"large-span-{i}",
        trace_id=f"large-trace-{i}",
    )
    span.set_attribute("large_data", "x" * 500)  # Make it big
    span.end()
    exporter_rotate.export([span])

rotated_files = list((export_dir / "rotated").glob("traces-*.jsonl"))
print(f"✓ Created {len(rotated_files)} rotated files")
for f in sorted(rotated_files):
    print(f"   - {f.name} ({f.stat().st_size} bytes)")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
print(f"\nCleanup: Delete '{export_dir}' directory to remove demo files")
