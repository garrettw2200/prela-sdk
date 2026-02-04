"""Exporters for sending spans to external systems."""

from prela.exporters.base import BaseExporter, BatchExporter, ExportResult
from prela.exporters.console import ConsoleExporter
from prela.exporters.file import FileExporter
from prela.exporters.http import HTTPExporter
from prela.exporters.multi import MultiExporter

# OTLP exporter requires optional dependency
try:
    from prela.exporters.otlp import OTLPExporter

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    OTLPExporter = None  # type: ignore

__all__ = [
    "BaseExporter",
    "BatchExporter",
    "ExportResult",
    "ConsoleExporter",
    "FileExporter",
    "HTTPExporter",
    "MultiExporter",
    "OTLPExporter",
]
