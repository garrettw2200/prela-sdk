"""
Multi exporter for sending traces to multiple backends simultaneously.

This exporter allows you to send the same traces to multiple destinations,
such as console + file, or file + OTLP, etc.
"""

from __future__ import annotations

import logging
from typing import Sequence

from prela.core.span import Span
from prela.exporters.base import BaseExporter, ExportResult

logger = logging.getLogger(__name__)


class MultiExporter(BaseExporter):
    """
    Exporter that fans out to multiple exporters.

    This allows sending traces to multiple backends simultaneously.
    Each exporter is called independently, and failures in one exporter
    don't affect others.

    Example:
        ```python
        from prela import init
        from prela.exporters import ConsoleExporter, FileExporter, MultiExporter
        from prela.exporters.otlp import OTLPExporter

        # Send to console + file + OTLP
        exporter = MultiExporter([
            ConsoleExporter(verbosity="normal"),
            FileExporter(directory="./traces"),
            OTLPExporter(endpoint="http://localhost:4318")
        ])

        init(service_name="my-app", exporter=exporter)
        ```

    Example with mixed results:
        ```python
        # Some exporters may succeed while others fail
        exporter = MultiExporter([
            ConsoleExporter(),  # Always succeeds
            FileExporter(),     # May fail (disk full)
            OTLPExporter()      # May fail (network down)
        ])

        # MultiExporter returns:
        # - SUCCESS if ALL exporters succeed
        # - RETRY if ANY exporter requests retry
        # - FAILURE if all exporters fail
        ```
    """

    def __init__(self, exporters: Sequence[BaseExporter]):
        """
        Initialize multi exporter.

        Args:
            exporters: List of exporters to fan out to

        Raises:
            ValueError: If exporters list is empty
        """
        if not exporters:
            raise ValueError("MultiExporter requires at least one exporter")

        self.exporters = list(exporters)
        logger.debug(f"MultiExporter initialized with {len(self.exporters)} exporters")

    def export(self, spans: list[Span]) -> ExportResult:
        """
        Export spans to all exporters.

        The export result is determined by:
        - SUCCESS: If all exporters succeed
        - RETRY: If any exporter requests retry (even if others succeed)
        - FAILURE: If all exporters fail

        Args:
            spans: List of spans to export

        Returns:
            ExportResult based on combined results
        """
        if not spans:
            return ExportResult.SUCCESS

        results = []

        # Export to each exporter
        for i, exporter in enumerate(self.exporters):
            try:
                result = exporter.export(spans)
                results.append(result)
                logger.debug(
                    f"Exporter {i} ({exporter.__class__.__name__}): {result.name}"
                )
            except Exception as e:
                logger.error(
                    f"Exporter {i} ({exporter.__class__.__name__}) raised exception: {e}",
                    exc_info=True,
                )
                results.append(ExportResult.FAILURE)

        # Determine combined result
        return self._combine_results(results)

    def _combine_results(self, results: list[ExportResult]) -> ExportResult:
        """
        Combine results from multiple exporters.

        Logic:
        - If any exporter requests RETRY, return RETRY
        - Else if all exporters return FAILURE, return FAILURE
        - Else return SUCCESS

        Args:
            results: List of export results

        Returns:
            Combined export result
        """
        if not results:
            return ExportResult.SUCCESS

        # Check for retry first (highest priority)
        if ExportResult.RETRY in results:
            return ExportResult.RETRY

        # Check if all failed
        if all(r == ExportResult.FAILURE for r in results):
            return ExportResult.FAILURE

        # Otherwise success (at least one succeeded)
        return ExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown all exporters."""
        logger.debug(f"Shutting down {len(self.exporters)} exporters")

        for i, exporter in enumerate(self.exporters):
            try:
                exporter.shutdown()
                logger.debug(f"Exporter {i} ({exporter.__class__.__name__}) shutdown")
            except Exception as e:
                logger.error(
                    f"Exporter {i} ({exporter.__class__.__name__}) shutdown error: {e}",
                    exc_info=True,
                )
