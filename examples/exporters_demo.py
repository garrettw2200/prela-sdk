"""
Demonstration of different exporters (Console, File, OTLP, Multi).

This example shows how to configure Prela with different exporters
for sending traces to various backends.

Run with:
    python examples/exporters_demo.py
"""

import time

import prela


# ============================================================================
# Example 1: Console Exporter (Default)
# ============================================================================


def example_1_console_exporter():
    """Basic console exporter."""
    print("\n" + "=" * 70)
    print("Example 1: Console Exporter (Default)")
    print("=" * 70)

    # Console exporter is the default
    prela.init(service_name="console-demo", exporter="console", verbosity="verbose")

    @prela.trace("process_data")
    def process_data(items: list[int]) -> list[int]:
        time.sleep(0.05)
        return [x * 2 for x in items]

    result = process_data([1, 2, 3])
    print(f"\nResult: {result}")


# ============================================================================
# Example 2: File Exporter
# ============================================================================


def example_2_file_exporter():
    """File exporter with JSONL format."""
    print("\n" + "=" * 70)
    print("Example 2: File Exporter")
    print("=" * 70)

    prela.init(
        service_name="file-demo",
        exporter="file",
        directory="./example_traces",
        max_file_size_mb=10,
        rotate=True,
    )

    @prela.trace("save_data")
    def save_data(data: dict) -> bool:
        time.sleep(0.05)
        return True

    result = save_data({"key": "value", "count": 42})
    print(f"\nResult: {result}")
    print("Traces saved to: ./example_traces/")


# ============================================================================
# Example 3: OTLP Exporter (Jaeger)
# ============================================================================


def example_3_otlp_exporter_jaeger():
    """OTLP exporter for Jaeger."""
    print("\n" + "=" * 70)
    print("Example 3: OTLP Exporter (Jaeger)")
    print("=" * 70)

    try:
        from prela.exporters.otlp import OTLPExporter

        # Send to local Jaeger instance
        prela.init(
            service_name="jaeger-demo",
            exporter=OTLPExporter(endpoint="http://localhost:4318/v1/traces"),
        )

        @prela.trace("query_data", span_type=prela.SpanType.RETRIEVAL)
        def query_data(query: str) -> list[dict]:
            time.sleep(0.05)
            return [{"id": 1, "title": "Result 1"}]

        result = query_data("search term")
        print(f"\nFound {len(result)} results")
        print("Traces sent to Jaeger at http://localhost:16686")

    except ImportError:
        print("\n⚠️  OTLP exporter requires: pip install 'prela[otlp]'")
    except Exception as e:
        print(f"\n⚠️  Could not connect to Jaeger: {e}")
        print("Make sure Jaeger is running: docker run -d -p16686:16686 -p4318:4318 jaegertracing/all-in-one:latest")


# ============================================================================
# Example 4: OTLP Exporter (Grafana Tempo)
# ============================================================================


def example_4_otlp_exporter_tempo():
    """OTLP exporter for Grafana Tempo."""
    print("\n" + "=" * 70)
    print("Example 4: OTLP Exporter (Grafana Tempo)")
    print("=" * 70)

    try:
        from prela.exporters.otlp import OTLPExporter

        # Send to Grafana Tempo with tenant header
        prela.init(
            service_name="tempo-demo",
            exporter=OTLPExporter(
                endpoint="http://tempo:4318/v1/traces",
                headers={"X-Scope-OrgID": "tenant1"},
            ),
        )

        @prela.trace("llm_call", span_type=prela.SpanType.LLM)
        def call_llm(prompt: str) -> str:
            time.sleep(0.1)
            return "LLM response"

        result = call_llm("What is AI?")
        print(f"\nResponse: {result}")
        print("Traces sent to Grafana Tempo")

    except ImportError:
        print("\n⚠️  OTLP exporter requires: pip install 'prela[otlp]'")
    except Exception as e:
        print(f"\n⚠️  Could not connect to Tempo: {e}")


# ============================================================================
# Example 5: OTLP Exporter (Honeycomb)
# ============================================================================


def example_5_otlp_exporter_honeycomb():
    """OTLP exporter for Honeycomb."""
    print("\n" + "=" * 70)
    print("Example 5: OTLP Exporter (Honeycomb)")
    print("=" * 70)

    try:
        from prela.exporters.otlp import OTLPExporter

        # Send to Honeycomb (requires API key)
        API_KEY = "YOUR_HONEYCOMB_API_KEY"  # Replace with real key
        if API_KEY == "YOUR_HONEYCOMB_API_KEY":
            print("\n⚠️  Set your Honeycomb API key to use this example")
            return

        prela.init(
            service_name="honeycomb-demo",
            exporter=OTLPExporter(
                endpoint="https://api.honeycomb.io:443/v1/traces",
                headers={
                    "x-honeycomb-team": API_KEY,
                    "x-honeycomb-dataset": "prela-demo",
                },
            ),
        )

        @prela.trace("agent_task", span_type=prela.SpanType.AGENT)
        def run_agent(task: str) -> str:
            time.sleep(0.1)
            return "Task complete"

        result = run_agent("Process documents")
        print(f"\nResult: {result}")
        print("Traces sent to Honeycomb")

    except ImportError:
        print("\n⚠️  OTLP exporter requires: pip install 'prela[otlp]'")
    except Exception as e:
        print(f"\n⚠️  Could not connect to Honeycomb: {e}")


# ============================================================================
# Example 6: Multiple Exporters
# ============================================================================


def example_6_multi_exporter():
    """Multiple exporters simultaneously."""
    print("\n" + "=" * 70)
    print("Example 6: Multiple Exporters")
    print("=" * 70)

    # Send to console + file simultaneously
    multi = prela.MultiExporter(
        [
            prela.ConsoleExporter(verbosity="normal"),
            prela.FileExporter(directory="./example_traces"),
        ]
    )

    prela.init(service_name="multi-demo", exporter=multi)

    @prela.trace("process_pipeline")
    def process_pipeline(data: str) -> str:
        time.sleep(0.05)
        return data.upper()

    result = process_pipeline("hello world")
    print(f"\nResult: {result}")
    print("Traces sent to console AND file")


# ============================================================================
# Example 7: Multi with OTLP
# ============================================================================


def example_7_multi_with_otlp():
    """Multiple exporters including OTLP."""
    print("\n" + "=" * 70)
    print("Example 7: Multi Exporter with OTLP")
    print("=" * 70)

    try:
        from prela.exporters.otlp import OTLPExporter

        # Send to console + file + Jaeger
        multi = prela.MultiExporter(
            [
                prela.ConsoleExporter(verbosity="minimal"),
                prela.FileExporter(directory="./example_traces"),
                OTLPExporter(endpoint="http://localhost:4318/v1/traces"),
            ]
        )

        prela.init(service_name="multi-otlp-demo", exporter=multi)

        @prela.trace("comprehensive_trace")
        def comprehensive_operation() -> dict:
            time.sleep(0.05)
            return {"status": "success", "items": 42}

        result = comprehensive_operation()
        print(f"\nResult: {result}")
        print("Traces sent to console + file + Jaeger")

    except ImportError:
        print("\n⚠️  OTLP exporter requires: pip install 'prela[otlp]'")
    except Exception as e:
        print(f"\n⚠️  Some exporters may have failed: {e}")


# ============================================================================
# Example 8: Custom Exporter
# ============================================================================


def example_8_custom_exporter():
    """Creating a custom exporter."""
    print("\n" + "=" * 70)
    print("Example 8: Custom Exporter")
    print("=" * 70)

    class CountingExporter(prela.BaseExporter):
        """Custom exporter that counts spans."""

        def __init__(self):
            self.span_count = 0
            self.trace_count = 0
            self.traces_seen = set()

        def export(self, spans):
            self.span_count += len(spans)
            for span in spans:
                self.traces_seen.add(span.trace_id)
            self.trace_count = len(self.traces_seen)
            print(f"  📊 Exported {len(spans)} spans")
            return prela.ExportResult.SUCCESS

        def shutdown(self):
            print(f"\n📈 Final stats:")
            print(f"  Total spans: {self.span_count}")
            print(f"  Total traces: {self.trace_count}")

    exporter = CountingExporter()
    prela.init(service_name="custom-demo", exporter=exporter)

    @prela.trace("operation_1")
    def operation_1():
        time.sleep(0.02)
        return "done"

    @prela.trace("operation_2")
    def operation_2():
        time.sleep(0.02)
        return "done"

    # Generate some traces
    for i in range(5):
        operation_1()
        operation_2()

    exporter.shutdown()


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PRELA EXPORTERS EXAMPLES")
    print("=" * 70)

    example_1_console_exporter()
    example_2_file_exporter()
    example_3_otlp_exporter_jaeger()
    example_4_otlp_exporter_tempo()
    example_5_otlp_exporter_honeycomb()
    example_6_multi_exporter()
    example_7_multi_with_otlp()
    example_8_custom_exporter()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
