"""Sampling strategies for trace collection.

This module provides different sampling strategies to control which traces
are collected and exported. Sampling helps reduce overhead and costs while
still providing useful observability data.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from threading import Lock


class BaseSampler(ABC):
    """Abstract base class for trace samplers.

    Samplers determine whether a trace should be collected based on
    the trace ID and potentially other factors.
    """

    @abstractmethod
    def should_sample(self, trace_id: str) -> bool:
        """Determine if a trace should be sampled.

        Args:
            trace_id: The trace ID to make a sampling decision for

        Returns:
            True if the trace should be sampled, False otherwise
        """
        pass


class AlwaysOnSampler(BaseSampler):
    """Sampler that always samples every trace.

    Use this in development or when you need complete trace coverage.
    Be aware this may generate high data volumes in production.
    """

    def should_sample(self, trace_id: str) -> bool:
        """Always return True.

        Args:
            trace_id: The trace ID (unused)

        Returns:
            Always True
        """
        return True


class AlwaysOffSampler(BaseSampler):
    """Sampler that never samples any traces.

    Use this to completely disable tracing, for example during
    maintenance windows or in testing environments.
    """

    def should_sample(self, trace_id: str) -> bool:
        """Always return False.

        Args:
            trace_id: The trace ID (unused)

        Returns:
            Always False
        """
        return False


class ProbabilitySampler(BaseSampler):
    """Sampler that samples traces with a fixed probability.

    This sampler uses a deterministic hash-based approach to ensure
    consistent sampling decisions for the same trace ID across
    different services and processes.
    """

    def __init__(self, rate: float) -> None:
        """Initialize the probability sampler.

        Args:
            rate: Sampling rate between 0.0 and 1.0 (inclusive)

        Raises:
            ValueError: If rate is not between 0.0 and 1.0
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Sampling rate must be between 0.0 and 1.0, got {rate}")
        self.rate = rate

    def should_sample(self, trace_id: str) -> bool:
        """Sample based on trace ID hash.

        Uses MD5 hash of trace_id to make a deterministic sampling decision.
        This ensures the same trace_id always gets the same sampling decision
        across different processes and services.

        Args:
            trace_id: The trace ID to make a sampling decision for

        Returns:
            True if the trace should be sampled, False otherwise
        """
        if self.rate == 0.0:
            return False
        if self.rate == 1.0:
            return True

        # Use MD5 hash to get a deterministic value between 0 and 1
        hash_bytes = hashlib.md5(trace_id.encode()).digest()
        # Take first 8 bytes and convert to int, then normalize to [0, 1]
        hash_value = int.from_bytes(hash_bytes[:8], byteorder="big")
        probability = hash_value / (2**64 - 1)

        return probability < self.rate


class RateLimitingSampler(BaseSampler):
    """Sampler that limits the number of traces sampled per second.

    This sampler uses a token bucket algorithm to enforce a maximum
    rate of sampled traces per second. Useful for controlling costs
    and backend load.
    """

    def __init__(self, traces_per_second: float) -> None:
        """Initialize the rate limiting sampler.

        Args:
            traces_per_second: Maximum number of traces to sample per second

        Raises:
            ValueError: If traces_per_second is negative
        """
        if traces_per_second < 0:
            raise ValueError(f"traces_per_second must be non-negative, got {traces_per_second}")

        self.traces_per_second = traces_per_second
        self._tokens = traces_per_second
        self._last_update = time.perf_counter()
        self._lock = Lock()

    def should_sample(self, trace_id: str) -> bool:
        """Sample if tokens are available.

        Uses a token bucket algorithm: tokens regenerate at the configured
        rate, and each sampling decision consumes one token.

        Args:
            trace_id: The trace ID (unused)

        Returns:
            True if a token is available, False otherwise
        """
        if self.traces_per_second == 0:
            return False

        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last_update

            # Refill tokens based on elapsed time
            self._tokens = min(
                self.traces_per_second,
                self._tokens + (elapsed * self.traces_per_second),
            )
            self._last_update = now

            # Try to consume a token
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            return False
