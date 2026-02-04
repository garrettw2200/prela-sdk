"""Prela - AI Agent Observability Platform SDK."""

from __future__ import annotations

import logging
import os
from typing import Any

from prela._version import __version__
from prela.core.context import TraceContext, get_current_span, new_trace_context
from prela.core.sampler import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    BaseSampler,
    ProbabilitySampler,
    RateLimitingSampler,
)
from prela.core.span import Span, SpanEvent, SpanStatus, SpanType
from prela.core.tracer import Tracer, get_tracer, set_global_tracer, trace
from prela.exporters.base import BaseExporter, ExportResult
from prela.exporters.console import ConsoleExporter
from prela.exporters.file import FileExporter
from prela.exporters.http import HTTPExporter
from prela.exporters.multi import MultiExporter

# Optional exporters
try:
    from prela.exporters.otlp import OTLPExporter
except ImportError:
    OTLPExporter = None  # type: ignore
from prela.instrumentation.auto import auto_instrument as _auto_instrument

__all__ = [
    # Version
    "__version__",
    # Main API
    "init",
    "Tracer",
    "get_tracer",
    "trace",
    # Span types
    "Span",
    "SpanEvent",
    "SpanType",
    "SpanStatus",
    # Context
    "TraceContext",
    "get_current_span",
    "new_trace_context",
    # Sampling
    "BaseSampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "ProbabilitySampler",
    "RateLimitingSampler",
    # Exporters
    "BaseExporter",
    "ConsoleExporter",
    "FileExporter",
    "HTTPExporter",
    "MultiExporter",
    "OTLPExporter",
    "ExportResult",
    # Auto-instrumentation
    "auto_instrument",
]


def init(
    service_name: str | None = None,
    exporter: str | BaseExporter | None = None,
    auto_instrument: bool = True,
    sample_rate: float | None = None,
    capture_for_replay: bool = False,
    project_id: str | None = None,
    n8n_webhook_port: int | None = None,
    n8n_webhook_host: str = "0.0.0.0",
    **kwargs: Any,
) -> Tracer:
    """
    Initialize Prela tracing with one line of code.

    This is the primary entry point for the Prela SDK. It:
    1. Creates a tracer with the specified configuration
    2. Sets it as the global tracer
    3. Auto-instruments detected LLM SDKs (Anthropic, OpenAI, etc.)
    4. Optionally starts n8n webhook receiver for zero-code workflow tracing
    5. Returns the tracer for manual span creation

    After calling init(), all LLM SDK calls are automatically traced!

    Args:
        service_name: Name of your service (default: $PRELA_SERVICE_NAME or "default")
        exporter: Where to send traces:
            - "console": Pretty-print to console (default)
            - "file": Write to JSONL file
            - "http": Send to HTTP endpoint (Railway, cloud backend)
            - BaseExporter instance: Custom exporter
            - Default: $PRELA_EXPORTER or "console"
        auto_instrument: Whether to auto-instrument detected libraries
            (default: True, disable with $PRELA_AUTO_INSTRUMENT=false)
        sample_rate: Sampling rate 0.0-1.0 (default: $PRELA_SAMPLE_RATE or 1.0)
        capture_for_replay: Enable full replay data capture (default: False)
            When enabled, captures complete request/response data including:
            - LLM: Full prompts, responses, streaming chunks, model info
            - Tools: Input args, output, side effects flag
            - Retrieval: Queries, documents, scores, metadata
            - Agents: System prompts, available tools, memory, config
            Use for debugging, testing, and auditing. Increases storage costs.
        project_id: Project ID for multi-tenant deployments (default: $PRELA_PROJECT_ID or None)
            Used for:
            - Organizing traces in multi-project deployments
            - Filtering dashboards by project
            - n8n webhook routing (?project={project_id})
        n8n_webhook_port: Port for n8n webhook receiver (optional, default: None)
            Set to enable n8n webhook-based tracing (e.g., 8787)
            Example: n8n_webhook_port=8787
        n8n_webhook_host: Host for n8n webhook receiver (default: "0.0.0.0")
            Usually "0.0.0.0" for accepting external connections
        **kwargs: Additional arguments passed to exporter
            For ConsoleExporter: verbosity, color, show_timestamps
            For FileExporter: directory, format, max_file_size_mb, rotate
            For HTTPExporter: endpoint, api_key, bearer_token, compress, headers

    Returns:
        Configured Tracer instance (also set as global tracer)

    Environment Variables:
        PRELA_SERVICE_NAME: Default service name
        PRELA_PROJECT_ID: Default project ID for multi-tenant setups
        PRELA_EXPORTER: Default exporter ("console", "file", or "http")
        PRELA_SAMPLE_RATE: Default sampling rate (0.0-1.0)
        PRELA_CAPTURE_REPLAY: Enable replay capture ("true", "1", or "yes")
        PRELA_AUTO_INSTRUMENT: Enable auto-instrumentation ("true" or "false")
        PRELA_DEBUG: Enable debug logging ("true" or "false")
        PRELA_TRACE_DIR: Directory for file exporter (default: ./traces)
        PRELA_HTTP_ENDPOINT: HTTP endpoint for http exporter
        PRELA_API_KEY: API key for http exporter
        PRELA_N8N_WEBHOOK_PORT: Port for n8n webhook receiver (optional)

    Example:
        ```python
        import prela

        # Simple initialization
        prela.init(service_name="my-agent")

        # All Anthropic/OpenAI calls now auto-traced!
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Trace is automatically captured and exported

        # Manual span creation
        with prela.get_tracer().span("custom_operation") as span:
            span.set_attribute("key", "value")
            # Do work...
        ```

    Example with console exporter (verbose mode):
        ```python
        import prela

        prela.init(
            service_name="my-agent",
            exporter="console",
            verbosity="verbose",  # "minimal", "normal", or "verbose"
            color=True,
            show_timestamps=True
        )
        ```

    Example with file exporter:
        ```python
        import prela

        prela.init(
            service_name="my-agent",
            exporter="file",
            directory="./traces",
            max_file_size_mb=100,  # 100 MB per file
            rotate=True
        )
        ```

    Example with HTTP exporter (Railway deployment):
        ```python
        import prela

        prela.init(
            service_name="my-agent",
            exporter="http",
            endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
            api_key="your-api-key",  # Optional
            compress=True  # Enable gzip compression
        )
        ```

    Example with n8n webhook receiver:
        ```python
        import prela

        # Start webhook receiver on port 8787
        prela.init(
            service_name="n8n-workflows",
            exporter="http",
            endpoint="https://prela-ingest-gateway-xxx.railway.app/v1/traces",
            n8n_webhook_port=8787
        )

        # Now configure n8n HTTP Request node to POST to:
        # http://your-server:8787/webhook
        # Body: {"workflow": "{{ $workflow }}", "execution": "{{ $execution }}", ...}
        ```

    Example with custom exporter:
        ```python
        from prela import init, BaseExporter, ExportResult

        class MyExporter(BaseExporter):
            def export(self, spans):
                # Send to your backend
                return ExportResult.SUCCESS

        init(service_name="my-agent", exporter=MyExporter())
        ```
    """
    # Read from environment variables with fallbacks
    service_name = service_name or os.getenv("PRELA_SERVICE_NAME", "default")
    project_id = project_id or os.getenv("PRELA_PROJECT_ID")  # Optional, defaults to None
    exporter_name = exporter or os.getenv("PRELA_EXPORTER", "console")
    sample_rate = (
        sample_rate
        if sample_rate is not None
        else float(os.getenv("PRELA_SAMPLE_RATE", "1.0"))
    )
    auto_instrument_enabled = auto_instrument and os.getenv(
        "PRELA_AUTO_INSTRUMENT", "true"
    ).lower() not in ("false", "0", "no")
    debug = os.getenv("PRELA_DEBUG", "false").lower() in ("true", "1", "yes")

    # n8n webhook configuration
    n8n_webhook_port = n8n_webhook_port or (
        int(os.getenv("PRELA_N8N_WEBHOOK_PORT"))
        if os.getenv("PRELA_N8N_WEBHOOK_PORT")
        else None
    )

    # Configure debug logging if requested
    if debug:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s"
        )
        logger = logging.getLogger("prela")
        logger.setLevel(logging.DEBUG)

    # Create sampler based on sample rate
    if sample_rate >= 1.0:
        sampler = AlwaysOnSampler()
    elif sample_rate <= 0.0:
        sampler = AlwaysOffSampler()
    else:
        sampler = ProbabilitySampler(rate=sample_rate)

    # Create exporter
    if isinstance(exporter_name, BaseExporter):
        # Custom exporter instance provided
        exporter_instance = exporter_name
    elif exporter_name == "console":
        exporter_instance = ConsoleExporter(**kwargs)
    elif exporter_name == "file":
        directory = kwargs.get("directory", os.getenv("PRELA_TRACE_DIR", "./traces"))
        # Remove directory from kwargs to avoid duplication
        file_kwargs = {k: v for k, v in kwargs.items() if k != "directory"}
        exporter_instance = FileExporter(directory=directory, **file_kwargs)
    elif exporter_name == "http":
        endpoint = kwargs.get("endpoint", os.getenv("PRELA_HTTP_ENDPOINT"))
        if not endpoint:
            raise ValueError(
                "HTTP exporter requires 'endpoint' parameter or PRELA_HTTP_ENDPOINT env var"
            )
        api_key = kwargs.get("api_key", os.getenv("PRELA_API_KEY"))
        # Remove endpoint and api_key from kwargs to avoid duplication
        http_kwargs = {k: v for k, v in kwargs.items() if k not in ("endpoint", "api_key")}
        if api_key:
            http_kwargs["api_key"] = api_key
        exporter_instance = HTTPExporter(endpoint=endpoint, **http_kwargs)
    else:
        raise ValueError(
            f"Unknown exporter: {exporter_name}. "
            f"Use 'console', 'file', 'http', or provide a BaseExporter instance."
        )

    # Read PRELA_CAPTURE_REPLAY environment variable
    capture_replay = capture_for_replay
    if not capture_replay:
        env_val = os.getenv("PRELA_CAPTURE_REPLAY", "").lower()
        capture_replay = env_val in ("true", "1", "yes")

    # Create tracer
    tracer = Tracer(
        service_name=service_name,
        exporter=exporter_instance,
        sampler=sampler,
        capture_for_replay=capture_replay,
    )

    # Set as global tracer
    tracer.set_global()

    # Auto-instrument detected libraries
    if auto_instrument_enabled:
        instrumented = _auto_instrument(tracer)
        if debug and instrumented:
            logger = logging.getLogger("prela")
            logger.debug(f"Auto-instrumented: {', '.join(instrumented)}")

    # Start n8n webhook receiver if configured
    if n8n_webhook_port:
        try:
            from prela.instrumentation.n8n.webhook import N8nWebhookHandler
            import threading

            webhook_handler = N8nWebhookHandler(
                tracer=tracer, port=n8n_webhook_port, host=n8n_webhook_host
            )

            # Start webhook handler in background thread
            webhook_thread = threading.Thread(
                target=webhook_handler.start_background,
                daemon=True,
                name="n8n-webhook-handler",
            )
            webhook_thread.start()

            if debug:
                logger = logging.getLogger("prela")
                logger.debug(
                    f"n8n webhook receiver started on {n8n_webhook_host}:{n8n_webhook_port}"
                )

            # Store references for cleanup (optional future use)
            tracer._n8n_webhook_handler = webhook_handler  # type: ignore
            tracer._n8n_webhook_thread = webhook_thread  # type: ignore

        except Exception as e:
            # Log warning but don't fail initialization
            logger = logging.getLogger("prela")
            logger.warning(f"Failed to start n8n webhook receiver: {e}")

    return tracer


def auto_instrument(tracer: Tracer | None = None) -> list[str]:
    """
    Manually trigger auto-instrumentation.

    This is useful if you want to control when instrumentation happens,
    or if you disabled auto-instrumentation in init() and want to enable
    it later.

    Args:
        tracer: Tracer instance to use (default: global tracer from init())

    Returns:
        List of instrumented library names (e.g., ["anthropic", "openai"])

    Raises:
        RuntimeError: If no tracer is provided and no global tracer is set

    Example:
        ```python
        import prela

        # Initialize without auto-instrumentation
        prela.init(service_name="my-app", auto_instrument=False)

        # Later, manually trigger instrumentation
        instrumented = prela.auto_instrument()
        print(f"Instrumented: {instrumented}")
        ```
    """
    if tracer is None:
        tracer = get_tracer()
        if tracer is None:
            raise RuntimeError(
                "No global tracer set. Call prela.init() first or provide a tracer."
            )

    return _auto_instrument(tracer)
