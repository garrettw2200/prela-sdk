"""Tests for trace samplers."""

import time
from collections import Counter

import pytest

from prela.core.sampler import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    BaseSampler,
    ProbabilitySampler,
    RateLimitingSampler,
)


class TestBaseSampler:
    """Tests for BaseSampler abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseSampler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSampler()  # type: ignore[abstract]

    def test_must_implement_should_sample(self) -> None:
        """Test that subclasses must implement should_sample method."""

        class IncompleteSampler(BaseSampler):
            pass

        with pytest.raises(TypeError):
            IncompleteSampler()  # type: ignore[abstract]

    def test_can_implement_should_sample(self) -> None:
        """Test that implementing should_sample allows instantiation."""

        class CompleteSampler(BaseSampler):
            def should_sample(self, trace_id: str) -> bool:
                return True

        sampler = CompleteSampler()
        assert isinstance(sampler, BaseSampler)


class TestAlwaysOnSampler:
    """Tests for AlwaysOnSampler."""

    def test_always_returns_true(self) -> None:
        """Test that AlwaysOnSampler always returns True."""
        sampler = AlwaysOnSampler()
        assert sampler.should_sample("trace-1") is True
        assert sampler.should_sample("trace-2") is True
        assert sampler.should_sample("trace-3") is True

    def test_with_different_trace_ids(self) -> None:
        """Test with various trace ID formats."""
        sampler = AlwaysOnSampler()
        assert sampler.should_sample("") is True
        assert sampler.should_sample("short") is True
        assert sampler.should_sample("a" * 1000) is True
        assert sampler.should_sample("123-456-789") is True
        assert sampler.should_sample("abc-def-ghi") is True

    def test_consistent_across_calls(self) -> None:
        """Test that same trace ID always returns True."""
        sampler = AlwaysOnSampler()
        trace_id = "test-trace-123"

        for _ in range(100):
            assert sampler.should_sample(trace_id) is True


class TestAlwaysOffSampler:
    """Tests for AlwaysOffSampler."""

    def test_always_returns_false(self) -> None:
        """Test that AlwaysOffSampler always returns False."""
        sampler = AlwaysOffSampler()
        assert sampler.should_sample("trace-1") is False
        assert sampler.should_sample("trace-2") is False
        assert sampler.should_sample("trace-3") is False

    def test_with_different_trace_ids(self) -> None:
        """Test with various trace ID formats."""
        sampler = AlwaysOffSampler()
        assert sampler.should_sample("") is False
        assert sampler.should_sample("short") is False
        assert sampler.should_sample("a" * 1000) is False
        assert sampler.should_sample("123-456-789") is False
        assert sampler.should_sample("abc-def-ghi") is False

    def test_consistent_across_calls(self) -> None:
        """Test that same trace ID always returns False."""
        sampler = AlwaysOffSampler()
        trace_id = "test-trace-123"

        for _ in range(100):
            assert sampler.should_sample(trace_id) is False


class TestProbabilitySampler:
    """Tests for ProbabilitySampler."""

    def test_rate_validation(self) -> None:
        """Test that invalid rates raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ProbabilitySampler(-0.1)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ProbabilitySampler(1.1)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ProbabilitySampler(2.0)

    def test_valid_rates(self) -> None:
        """Test that valid rates are accepted."""
        ProbabilitySampler(0.0)
        ProbabilitySampler(0.5)
        ProbabilitySampler(1.0)

    def test_rate_zero_never_samples(self) -> None:
        """Test that rate=0.0 never samples."""
        sampler = ProbabilitySampler(0.0)
        for i in range(100):
            assert sampler.should_sample(f"trace-{i}") is False

    def test_rate_one_always_samples(self) -> None:
        """Test that rate=1.0 always samples."""
        sampler = ProbabilitySampler(1.0)
        for i in range(100):
            assert sampler.should_sample(f"trace-{i}") is True

    def test_deterministic_sampling(self) -> None:
        """Test that same trace_id gets same result."""
        sampler = ProbabilitySampler(0.5)
        trace_id = "test-trace-123"

        first_result = sampler.should_sample(trace_id)
        for _ in range(10):
            assert sampler.should_sample(trace_id) == first_result

    def test_different_trace_ids_vary(self) -> None:
        """Test that different trace IDs can have different results."""
        sampler = ProbabilitySampler(0.5)

        results = [sampler.should_sample(f"trace-{i}") for i in range(100)]

        # With 100 traces at 50% rate, we should see both True and False
        assert True in results
        assert False in results

    def test_approximate_sampling_rate(self) -> None:
        """Test that actual sampling rate approximates configured rate."""
        rate = 0.3
        sampler = ProbabilitySampler(rate)

        # Test with 10000 different trace IDs
        results = [sampler.should_sample(f"trace-{i}") for i in range(10000)]
        actual_rate = sum(results) / len(results)

        # Should be within 5% of target rate
        assert abs(actual_rate - rate) < 0.05

    def test_different_rates(self) -> None:
        """Test various sampling rates."""
        for rate in [0.1, 0.25, 0.5, 0.75, 0.9]:
            sampler = ProbabilitySampler(rate)
            results = [sampler.should_sample(f"trace-{i}") for i in range(1000)]
            actual_rate = sum(results) / len(results)

            # Should be within 10% for smaller sample sizes
            assert abs(actual_rate - rate) < 0.1

    def test_consistency_across_instances(self) -> None:
        """Test that different sampler instances make same decision."""
        sampler1 = ProbabilitySampler(0.5)
        sampler2 = ProbabilitySampler(0.5)

        trace_id = "test-trace-456"

        assert sampler1.should_sample(trace_id) == sampler2.should_sample(trace_id)

    def test_hash_distribution(self) -> None:
        """Test that hashing provides good distribution."""
        sampler = ProbabilitySampler(0.5)

        # Test sequential trace IDs
        results = [sampler.should_sample(f"trace-{i:06d}") for i in range(1000)]

        # Count True and False
        true_count = sum(results)
        false_count = len(results) - true_count

        # Distribution should be relatively even (within 20%)
        assert 400 < true_count < 600
        assert 400 < false_count < 600


class TestRateLimitingSampler:
    """Tests for RateLimitingSampler."""

    def test_negative_rate_raises_error(self) -> None:
        """Test that negative rate raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            RateLimitingSampler(-1.0)

    def test_zero_rate_never_samples(self) -> None:
        """Test that rate=0 never samples."""
        sampler = RateLimitingSampler(0)
        for i in range(10):
            assert sampler.should_sample(f"trace-{i}") is False

    def test_valid_rates(self) -> None:
        """Test that valid rates are accepted."""
        RateLimitingSampler(0)
        RateLimitingSampler(1.0)
        RateLimitingSampler(10.0)
        RateLimitingSampler(100.0)

    def test_samples_up_to_limit(self) -> None:
        """Test that sampler allows up to the rate limit."""
        sampler = RateLimitingSampler(5.0)  # 5 traces per second

        # Should allow first 5 traces
        results = [sampler.should_sample(f"trace-{i}") for i in range(5)]
        assert sum(results) == 5

    def test_rejects_over_limit(self) -> None:
        """Test that sampler rejects traces over the limit."""
        sampler = RateLimitingSampler(3.0)  # 3 traces per second

        # First 3 should pass
        results = [sampler.should_sample(f"trace-{i}") for i in range(3)]
        assert sum(results) == 3

        # Next few should be rejected (no time has passed)
        results = [sampler.should_sample(f"trace-{i}") for i in range(3, 6)]
        assert sum(results) == 0

    def test_token_regeneration(self) -> None:
        """Test that tokens regenerate over time."""
        sampler = RateLimitingSampler(10.0)  # 10 traces per second

        # Use up initial tokens
        for i in range(10):
            sampler.should_sample(f"trace-{i}")

        # Wait for tokens to regenerate (100ms = 1 token at 10/sec)
        time.sleep(0.15)

        # Should allow at least 1 more trace
        assert sampler.should_sample("trace-new") is True

    def test_fractional_rate(self) -> None:
        """Test fractional traces per second."""
        sampler = RateLimitingSampler(2.5)  # 2.5 traces per second

        # Should allow 2 traces immediately
        assert sampler.should_sample("trace-1") is True
        assert sampler.should_sample("trace-2") is True

        # Third might be allowed (we have 2.5 tokens initially)
        result = sampler.should_sample("trace-3")
        # Don't assert specific value, just verify it's boolean
        assert isinstance(result, bool)

    def test_high_rate_limit(self) -> None:
        """Test with high rate limit."""
        sampler = RateLimitingSampler(1000.0)  # 1000 traces per second

        # Should allow many traces quickly
        results = [sampler.should_sample(f"trace-{i}") for i in range(100)]
        assert sum(results) == 100

    def test_low_rate_limit(self) -> None:
        """Test with low rate limit."""
        sampler = RateLimitingSampler(1.0)  # 1 trace per second

        # Should allow 1 trace
        assert sampler.should_sample("trace-1") is True

        # Should reject next traces immediately
        assert sampler.should_sample("trace-2") is False
        assert sampler.should_sample("trace-3") is False

    def test_token_bucket_max_capacity(self) -> None:
        """Test that token bucket doesn't exceed max capacity."""
        sampler = RateLimitingSampler(5.0)  # 5 traces per second

        # Wait a long time to allow tokens to accumulate
        time.sleep(2.0)  # Should accumulate way more than 5 tokens if no cap

        # Should only allow 5 traces, not more
        results = [sampler.should_sample(f"trace-{i}") for i in range(10)]
        sampled_count = sum(results)

        # Should be around 5, maybe a bit more due to regeneration during loop
        # But definitely not 10+
        assert 5 <= sampled_count <= 7

    def test_concurrent_sampling(self) -> None:
        """Test that sampler is thread-safe."""
        import threading

        sampler = RateLimitingSampler(10.0)
        results: list[bool] = []
        lock = threading.Lock()

        def sample_trace(trace_id: str) -> None:
            result = sampler.should_sample(trace_id)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=sample_trace, args=(f"trace-{i}",)) for i in range(20)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have sampled about 10 traces (the initial capacity)
        sampled_count = sum(results)
        assert 8 <= sampled_count <= 12  # Allow some variance

    def test_regeneration_rate(self) -> None:
        """Test that tokens regenerate at the correct rate."""
        sampler = RateLimitingSampler(10.0)  # 10 traces per second

        # Consume all tokens
        for i in range(10):
            sampler.should_sample(f"trace-{i}")

        # Wait 0.5 seconds (should regenerate ~5 tokens)
        time.sleep(0.5)

        # Count how many traces are sampled
        results = [sampler.should_sample(f"trace-new-{i}") for i in range(10)]
        sampled_count = sum(results)

        # Should be around 5, with some tolerance
        assert 3 <= sampled_count <= 7


class TestSamplerComparison:
    """Comparison tests between different samplers."""

    def test_always_on_vs_probability_100(self) -> None:
        """Test AlwaysOnSampler vs ProbabilitySampler(1.0)."""
        always_on = AlwaysOnSampler()
        prob_100 = ProbabilitySampler(1.0)

        for i in range(100):
            trace_id = f"trace-{i}"
            assert always_on.should_sample(trace_id) == prob_100.should_sample(trace_id)
            assert always_on.should_sample(trace_id) is True

    def test_always_off_vs_probability_0(self) -> None:
        """Test AlwaysOffSampler vs ProbabilitySampler(0.0)."""
        always_off = AlwaysOffSampler()
        prob_0 = ProbabilitySampler(0.0)

        for i in range(100):
            trace_id = f"trace-{i}"
            assert always_off.should_sample(trace_id) == prob_0.should_sample(trace_id)
            assert always_off.should_sample(trace_id) is False

    def test_different_sampler_types_can_coexist(self) -> None:
        """Test that different sampler types can be used together."""
        samplers = [
            AlwaysOnSampler(),
            AlwaysOffSampler(),
            ProbabilitySampler(0.5),
            RateLimitingSampler(10.0),
        ]

        trace_id = "test-trace"

        # All samplers should be able to make decisions
        for sampler in samplers:
            result = sampler.should_sample(trace_id)
            assert isinstance(result, bool)


class TestSamplerEdgeCases:
    """Edge case tests for samplers."""

    def test_empty_trace_id(self) -> None:
        """Test samplers with empty trace ID."""
        samplers = [
            AlwaysOnSampler(),
            AlwaysOffSampler(),
            ProbabilitySampler(0.5),
            RateLimitingSampler(10.0),
        ]

        for sampler in samplers:
            result = sampler.should_sample("")
            assert isinstance(result, bool)

    def test_very_long_trace_id(self) -> None:
        """Test samplers with very long trace ID."""
        trace_id = "x" * 10000
        samplers = [
            AlwaysOnSampler(),
            AlwaysOffSampler(),
            ProbabilitySampler(0.5),
            RateLimitingSampler(10.0),
        ]

        for sampler in samplers:
            result = sampler.should_sample(trace_id)
            assert isinstance(result, bool)

    def test_special_characters_in_trace_id(self) -> None:
        """Test samplers with special characters in trace ID."""
        trace_ids = [
            "trace-with-dashes",
            "trace_with_underscores",
            "trace.with.dots",
            "trace/with/slashes",
            "trace:with:colons",
            "trace@with@at",
            "trace#with#hash",
            "trace$with$dollar",
            "trace with spaces",
            "trace\twith\ttabs",
            "trace\nwith\nnewlines",
        ]

        sampler = ProbabilitySampler(0.5)

        for trace_id in trace_ids:
            result = sampler.should_sample(trace_id)
            assert isinstance(result, bool)

    def test_unicode_trace_id(self) -> None:
        """Test samplers with unicode characters."""
        trace_ids = [
            "trace-🚀",
            "trace-日本語",
            "trace-🎉🎊🎈",
            "trace-αβγ",
            "trace-עברית",
        ]

        sampler = ProbabilitySampler(0.5)

        for trace_id in trace_ids:
            result = sampler.should_sample(trace_id)
            assert isinstance(result, bool)
