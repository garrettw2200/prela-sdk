"""Tests for clock utilities."""

import time
from datetime import datetime, timezone

import pytest

from prela.core.clock import (
    duration_ms,
    format_timestamp,
    monotonic_ns,
    now,
    parse_timestamp,
)


class TestNow:
    """Tests for now() function."""

    def test_returns_datetime(self) -> None:
        """Test that now() returns a datetime object."""
        result = now()
        assert isinstance(result, datetime)

    def test_returns_utc_timezone(self) -> None:
        """Test that now() returns UTC timezone."""
        result = now()
        assert result.tzinfo == timezone.utc

    def test_has_microsecond_precision(self) -> None:
        """Test that now() includes microseconds."""
        result = now()
        # Just verify microseconds field exists and is an integer
        assert isinstance(result.microsecond, int)
        assert 0 <= result.microsecond < 1_000_000

    def test_successive_calls_increase(self) -> None:
        """Test that successive calls return increasing times."""
        time1 = now()
        time.sleep(0.001)  # Sleep 1ms
        time2 = now()
        assert time2 > time1

    def test_returns_recent_time(self) -> None:
        """Test that now() returns a recent timestamp."""
        result = now()
        # Should be within the last second
        expected = datetime.now(timezone.utc)
        delta = abs((expected - result).total_seconds())
        assert delta < 1.0


class TestMonotonicNs:
    """Tests for monotonic_ns() function."""

    def test_returns_int(self) -> None:
        """Test that monotonic_ns() returns an integer."""
        result = monotonic_ns()
        assert isinstance(result, int)

    def test_returns_positive(self) -> None:
        """Test that monotonic_ns() returns a positive number."""
        result = monotonic_ns()
        assert result > 0

    def test_successive_calls_increase(self) -> None:
        """Test that successive calls return increasing values."""
        time1 = monotonic_ns()
        time.sleep(0.001)  # Sleep 1ms
        time2 = monotonic_ns()
        assert time2 > time1

    def test_monotonic_increases_consistently(self) -> None:
        """Test that monotonic time never decreases."""
        times = [monotonic_ns() for _ in range(10)]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]

    def test_nanosecond_precision(self) -> None:
        """Test that monotonic_ns() has nanosecond precision."""
        start = monotonic_ns()
        end = monotonic_ns()
        # Should be measurable in nanoseconds
        assert end >= start


class TestDurationMs:
    """Tests for duration_ms() function."""

    def test_zero_duration(self) -> None:
        """Test duration calculation with same start and end."""
        start = monotonic_ns()
        duration = duration_ms(start, start)
        assert duration == 0.0

    def test_positive_duration(self) -> None:
        """Test duration calculation with positive difference."""
        start = 0
        end = 1_000_000  # 1ms in nanoseconds
        duration = duration_ms(start, end)
        assert duration == 1.0

    def test_multiple_milliseconds(self) -> None:
        """Test duration calculation for multiple milliseconds."""
        start = 0
        end = 5_000_000  # 5ms in nanoseconds
        duration = duration_ms(start, end)
        assert duration == 5.0

    def test_fractional_milliseconds(self) -> None:
        """Test duration calculation with fractional milliseconds."""
        start = 0
        end = 1_500_000  # 1.5ms in nanoseconds
        duration = duration_ms(start, end)
        assert duration == 1.5

    def test_microsecond_precision(self) -> None:
        """Test duration calculation with microsecond precision."""
        start = 0
        end = 1_234  # 0.001234ms in nanoseconds
        duration = duration_ms(start, end)
        assert duration == pytest.approx(0.001234)

    def test_realistic_duration(self) -> None:
        """Test duration with realistic monotonic values."""
        start = monotonic_ns()
        time.sleep(0.01)  # Sleep for ~10ms
        end = monotonic_ns()
        duration = duration_ms(start, end)
        # Should be at least 10ms, but allow some overhead
        assert 10.0 <= duration < 20.0


class TestFormatTimestamp:
    """Tests for format_timestamp() function."""

    def test_formats_datetime(self) -> None:
        """Test formatting a datetime object."""
        dt = datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert isinstance(result, str)

    def test_iso_format(self) -> None:
        """Test that output is in ISO 8601 format."""
        dt = datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert result == "2024-01-15T12:30:45.123456+00:00"

    def test_includes_microseconds(self) -> None:
        """Test that microseconds are included."""
        dt = datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert "123456" in result

    def test_includes_timezone(self) -> None:
        """Test that timezone information is included."""
        dt = datetime(2024, 1, 15, 12, 30, 45, 0, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert "+00:00" in result

    def test_zero_microseconds(self) -> None:
        """Test formatting with zero microseconds."""
        dt = datetime(2024, 1, 15, 12, 30, 45, 0, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert "2024-01-15T12:30:45+00:00" in result

    def test_formats_now(self) -> None:
        """Test formatting the current time."""
        dt = now()
        result = format_timestamp(dt)
        assert isinstance(result, str)
        assert "T" in result  # Contains date-time separator
        assert "+" in result or "Z" in result  # Contains timezone


class TestParseTimestamp:
    """Tests for parse_timestamp() function."""

    def test_parses_iso_string(self) -> None:
        """Test parsing an ISO 8601 string."""
        s = "2024-01-15T12:30:45.123456+00:00"
        result = parse_timestamp(s)
        assert isinstance(result, datetime)

    def test_parses_components_correctly(self) -> None:
        """Test that all datetime components are parsed correctly."""
        s = "2024-01-15T12:30:45.123456+00:00"
        result = parse_timestamp(s)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        assert result.microsecond == 123456

    def test_parses_timezone(self) -> None:
        """Test that timezone is parsed correctly."""
        s = "2024-01-15T12:30:45.123456+00:00"
        result = parse_timestamp(s)
        assert result.tzinfo is not None

    def test_parses_without_microseconds(self) -> None:
        """Test parsing without microseconds."""
        s = "2024-01-15T12:30:45+00:00"
        result = parse_timestamp(s)
        assert result.year == 2024
        assert result.microsecond == 0

    def test_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_timestamp("not a timestamp")

    def test_invalid_date_raises_error(self) -> None:
        """Test that invalid date raises ValueError."""
        with pytest.raises(ValueError):
            parse_timestamp("2024-13-45T12:30:45+00:00")  # Invalid month/day


class TestFormatParseRoundtrip:
    """Tests for format/parse roundtrip operations."""

    def test_roundtrip_preserves_datetime(self) -> None:
        """Test that format->parse roundtrip preserves datetime."""
        original = datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        formatted = format_timestamp(original)
        parsed = parse_timestamp(formatted)
        assert parsed == original

    def test_roundtrip_with_now(self) -> None:
        """Test roundtrip with current time."""
        original = now()
        formatted = format_timestamp(original)
        parsed = parse_timestamp(formatted)
        # Should be equal (timestamp precision is maintained)
        assert parsed == original

    def test_multiple_roundtrips(self) -> None:
        """Test multiple roundtrips maintain consistency."""
        original = now()
        first_formatted = format_timestamp(original)

        # Do multiple roundtrips
        current = original
        for _ in range(3):
            formatted = format_timestamp(current)
            current = parse_timestamp(formatted)

        # After roundtrips, should still equal the original
        assert current == original
        # And formatting should be consistent
        assert format_timestamp(current) == first_formatted


class TestIntegration:
    """Integration tests combining multiple clock functions."""

    def test_measure_duration_with_now(self) -> None:
        """Test measuring duration using now() timestamps."""
        start_dt = now()
        start_ns = monotonic_ns()

        time.sleep(0.01)  # Sleep for ~10ms

        end_dt = now()
        end_ns = monotonic_ns()

        # Both methods should give similar durations
        dt_duration = (end_dt - start_dt).total_seconds() * 1000
        ns_duration = duration_ms(start_ns, end_ns)

        # Should be within 5ms of each other
        assert abs(dt_duration - ns_duration) < 5.0

    def test_format_parse_with_now(self) -> None:
        """Test formatting and parsing current time."""
        original = now()
        formatted = format_timestamp(original)
        parsed = parse_timestamp(formatted)

        assert parsed == original
        assert parsed.tzinfo == timezone.utc

    def test_monotonic_measures_actual_time(self) -> None:
        """Test that monotonic time measures actual elapsed time."""
        sleep_ms = 10
        start = monotonic_ns()
        time.sleep(sleep_ms / 1000)
        end = monotonic_ns()

        measured = duration_ms(start, end)
        # Should be at least the sleep time
        assert measured >= sleep_ms
        # But not too much longer (allow 5ms overhead)
        assert measured < sleep_ms + 5

    def test_all_time_functions_work_together(self) -> None:
        """Test using all time functions together."""
        # Get current time
        timestamp = now()

        # Start monotonic timer
        start = monotonic_ns()

        # Do some work
        time.sleep(0.001)

        # End timer
        end = monotonic_ns()

        # Format timestamp
        formatted = format_timestamp(timestamp)

        # Parse it back
        parsed = parse_timestamp(formatted)

        # Calculate duration
        elapsed = duration_ms(start, end)

        # Verify everything works
        assert parsed == timestamp
        assert elapsed >= 1.0  # At least 1ms
        assert isinstance(formatted, str)
