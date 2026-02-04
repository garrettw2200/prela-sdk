"""Base classes for span exporters.

This module provides abstract base classes for implementing span exporters.
Exporters are responsible for sending completed spans to external systems
like observability platforms, databases, or files.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from prela.core.span import Span

logger = logging.getLogger(__name__)


class ExportResult(Enum):
    """Result of an export operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"


class BaseExporter(ABC):
    """Abstract base class for span exporters.

    Exporters are responsible for sending spans to external systems.
    Implementations must handle serialization, network requests, and error handling.
    """

    @abstractmethod
    def export(self, spans: list[Span]) -> None:
        """Export a batch of spans.

        Args:
            spans: List of spans to export

        Raises:
            Exception: If export fails and should not be retried
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter and flush any pending data.

        This method should be called before the application exits to ensure
        all spans are properly exported.
        """
        pass


class BatchExporter(BaseExporter):
    """Base class for exporters that batch spans with retry logic.

    This class handles common batching concerns:
    - Retry with exponential backoff
    - Timeout handling
    - Error logging

    Subclasses only need to implement _do_export() to define how spans
    are actually sent to the backend.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_ms: float = 100.0,
        max_backoff_ms: float = 10000.0,
        timeout_ms: float = 30000.0,
    ) -> None:
        """Initialize the batch exporter.

        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff_ms: Initial backoff delay in milliseconds
            max_backoff_ms: Maximum backoff delay in milliseconds
            timeout_ms: Timeout for export operation in milliseconds
        """
        self.max_retries = max_retries
        self.initial_backoff_ms = initial_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self.timeout_ms = timeout_ms
        self._shutdown = False

    @abstractmethod
    def _do_export(self, spans: list[Span]) -> ExportResult:
        """Perform the actual export operation.

        This method should be implemented by subclasses to define how spans
        are sent to the backend system.

        Args:
            spans: List of spans to export

        Returns:
            ExportResult indicating success, failure, or retry needed
        """
        pass

    def export(self, spans: list[Span]) -> None:
        """Export spans with retry logic.

        Args:
            spans: List of spans to export

        Raises:
            RuntimeError: If exporter is shutdown
            Exception: If export fails after all retries
        """
        if self._shutdown:
            raise RuntimeError("Cannot export: exporter is shutdown")

        if not spans:
            return

        start_time = time.perf_counter()
        attempt = 0
        backoff_ms = self.initial_backoff_ms

        while attempt <= self.max_retries:
            # Check timeout
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms >= self.timeout_ms:
                raise TimeoutError(
                    f"Export timeout after {elapsed_ms:.2f}ms " f"(limit: {self.timeout_ms}ms)"
                )

            try:
                result = self._do_export(spans)

                if result == ExportResult.SUCCESS:
                    logger.debug(
                        "Successfully exported %d spans on attempt %d",
                        len(spans),
                        attempt + 1,
                    )
                    return

                if result == ExportResult.FAILURE:
                    raise Exception(f"Export failed permanently on attempt {attempt + 1}")

                # result == ExportResult.RETRY
                if attempt < self.max_retries:
                    logger.warning(
                        "Export needs retry (attempt %d/%d), backing off %.2fms",
                        attempt + 1,
                        self.max_retries + 1,
                        backoff_ms,
                    )
                    time.sleep(backoff_ms / 1000)
                    backoff_ms = min(backoff_ms * 2, self.max_backoff_ms)

            except Exception as e:
                if attempt >= self.max_retries:
                    logger.error(
                        "Export failed after %d attempts: %s",
                        attempt + 1,
                        str(e),
                    )
                    raise

                logger.warning(
                    "Export failed (attempt %d/%d): %s, backing off %.2fms",
                    attempt + 1,
                    self.max_retries + 1,
                    str(e),
                    backoff_ms,
                )
                time.sleep(backoff_ms / 1000)
                backoff_ms = min(backoff_ms * 2, self.max_backoff_ms)

            attempt += 1

        raise Exception(f"Export failed after {self.max_retries + 1} attempts")

    def shutdown(self) -> None:
        """Shutdown the exporter.

        Subclasses can override this to implement custom shutdown logic
        like flushing buffers or closing connections.
        """
        self._shutdown = True
        logger.info("Exporter shutdown")
