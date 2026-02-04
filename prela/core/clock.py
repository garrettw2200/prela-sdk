"""Clock utilities for consistent time handling.

This module provides utilities for working with timestamps and durations
in a consistent way across the SDK. All timestamps are in UTC.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def now() -> datetime:
    """Get the current UTC time with microsecond precision.

    Returns:
        Current datetime in UTC with microsecond precision

    Example:
        >>> timestamp = now()
        >>> timestamp.tzinfo == timezone.utc
        True
    """
    return datetime.now(timezone.utc)


def monotonic_ns() -> int:
    """Get monotonic time in nanoseconds.

    This is useful for measuring durations as it's not affected by
    system clock adjustments.

    Returns:
        Monotonic time in nanoseconds

    Example:
        >>> start = monotonic_ns()
        >>> # ... do work ...
        >>> end = monotonic_ns()
        >>> elapsed_ms = duration_ms(start, end)
    """
    return time.perf_counter_ns()


def duration_ms(start_ns: int, end_ns: int) -> float:
    """Calculate duration in milliseconds from nanosecond timestamps.

    Args:
        start_ns: Start time in nanoseconds (from monotonic_ns)
        end_ns: End time in nanoseconds (from monotonic_ns)

    Returns:
        Duration in milliseconds

    Example:
        >>> start = monotonic_ns()
        >>> end = start + 1_500_000  # 1.5ms later
        >>> duration_ms(start, end)
        1.5
    """
    return (end_ns - start_ns) / 1_000_000


def format_timestamp(dt: datetime) -> str:
    """Format a datetime as ISO 8601 string.

    Args:
        dt: Datetime to format

    Returns:
        ISO 8601 formatted string

    Example:
        >>> dt = datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        >>> format_timestamp(dt)
        '2024-01-15T12:30:45.123456+00:00'
    """
    return dt.isoformat()


def parse_timestamp(s: str) -> datetime:
    """Parse an ISO 8601 timestamp string.

    Args:
        s: ISO 8601 formatted timestamp string

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If the string is not a valid ISO 8601 timestamp

    Example:
        >>> dt = parse_timestamp('2024-01-15T12:30:45.123456+00:00')
        >>> dt.year
        2024
    """
    return datetime.fromisoformat(s)
