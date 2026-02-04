"""Example demonstrating HTTP exporter usage for Railway deployment.

This example shows how to configure Prela to send traces to a remote
backend (like Railway-deployed Ingest Gateway) using the HTTP exporter.
"""

import os

import prela

# Example 1: Basic HTTP exporter with Railway endpoint
print("=" * 60)
print("Example 1: Basic HTTP Exporter")
print("=" * 60)

tracer = prela.init(
    service_name="my-agent",
    exporter="http",
    endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
)

# Create a traced operation
with tracer.span("test_operation", span_type="agent") as span:
    span.set_attribute("test.key", "test.value")
    print("Created span:", span.span_id)

print("✓ Trace exported to Railway backend\n")


# Example 2: HTTP exporter with API key authentication
print("=" * 60)
print("Example 2: HTTP Exporter with API Key")
print("=" * 60)

tracer = prela.init(
    service_name="my-agent",
    exporter="http",
    endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
    api_key="your-api-key-here",
)

with tracer.span("authenticated_operation", span_type="agent") as span:
    span.set_attribute("authenticated", True)
    print("Created authenticated span:", span.span_id)

print("✓ Trace exported with API key authentication\n")


# Example 3: HTTP exporter with gzip compression
print("=" * 60)
print("Example 3: HTTP Exporter with Compression")
print("=" * 60)

tracer = prela.init(
    service_name="my-agent",
    exporter="http",
    endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
    compress=True,  # Enable gzip compression
)

with tracer.span("compressed_operation", span_type="agent") as span:
    span.set_attribute("large.data", "x" * 1000)  # Large attribute
    print("Created span with large data:", span.span_id)

print("✓ Trace exported with gzip compression\n")


# Example 4: HTTP exporter with environment variables
print("=" * 60)
print("Example 4: HTTP Exporter with Environment Variables")
print("=" * 60)

# Set environment variables
os.environ["PRELA_HTTP_ENDPOINT"] = (
    "https://prela-ingest-gateway-xxx.railway.app/v1/traces"
)
os.environ["PRELA_API_KEY"] = "your-api-key-here"

# Initialize without explicit parameters (uses env vars)
tracer = prela.init(
    service_name="my-agent",
    exporter="http",
    # endpoint and api_key are read from environment variables
)

with tracer.span("env_var_operation", span_type="agent") as span:
    span.set_attribute("configured_via", "environment")
    print("Created span:", span.span_id)

print("✓ Trace exported using environment variable configuration\n")


# Example 5: HTTP exporter with custom headers and retry config
print("=" * 60)
print("Example 5: HTTP Exporter with Advanced Configuration")
print("=" * 60)

from prela.exporters.http import HTTPExporter

exporter = HTTPExporter(
    endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
    api_key="your-api-key",
    compress=True,
    headers={"X-Custom-Header": "custom-value"},
    max_retries=5,
    initial_backoff_ms=200.0,
    max_backoff_ms=5000.0,
    timeout_ms=10000.0,
)

tracer = prela.Tracer(service_name="my-agent", exporter=exporter)

with tracer.span("advanced_operation", span_type="agent") as span:
    span.set_attribute("advanced.config", True)
    print("Created span with advanced config:", span.span_id)

print("✓ Trace exported with advanced configuration\n")


# Example 6: Full production setup with auto-instrumentation
print("=" * 60)
print("Example 6: Production Setup with Auto-Instrumentation")
print("=" * 60)

tracer = prela.init(
    service_name="production-agent",
    exporter="http",
    endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
    api_key="your-api-key",
    compress=True,
    sample_rate=1.0,  # Sample 100% of traces
    auto_instrument=True,  # Auto-instrument Anthropic, OpenAI, etc.
)

print("✓ Production configuration complete")
print("✓ All LLM SDK calls will be automatically traced")
print("✓ Traces will be sent to Railway backend with compression\n")

# Now all Anthropic/OpenAI calls are automatically traced!
# Example (requires anthropic package):
# from anthropic import Anthropic
# client = Anthropic()
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     messages=[{"role": "user", "content": "Hello!"}]
# )
# Trace is automatically captured and exported to Railway!


print("=" * 60)
print("All Examples Complete!")
print("=" * 60)
print("\nKey Points:")
print("- Replace 'prela-ingest-gateway-xxx.railway.app' with your actual Railway URL")
print("- Use environment variables (PRELA_HTTP_ENDPOINT, PRELA_API_KEY) for production")
print("- Enable compression for bandwidth savings")
print("- Configure retry logic for reliability")
print("- Use auto-instrumentation for automatic LLM tracing")
