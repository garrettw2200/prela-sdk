"""Tests for license and tier-based feature gating."""

import pytest
import os

from prela.license import (
    set_tier,
    get_tier,
    has_access,
    require_tier,
    check_tier,
    SubscriptionError,
    TIER_HIERARCHY,
)


class TestTierManagement:
    """Test tier management functions."""

    def test_tier_hierarchy(self):
        """Test tier hierarchy is correctly defined."""
        assert TIER_HIERARCHY == ["free", "lunch-money", "pro", "enterprise"]

    def test_set_and_get_tier(self):
        """Test setting and getting tier."""
        set_tier("pro")
        assert get_tier() == "pro"

        set_tier("free")
        assert get_tier() == "free"

    def test_set_unknown_tier_defaults_to_free(self):
        """Test unknown tier defaults to free."""
        set_tier("unknown-tier")
        assert get_tier() == "free"

    def test_get_tier_from_environment(self):
        """Test getting tier from environment variable."""
        # Clear current tier
        import prela.license
        prela.license._current_tier = None

        os.environ["PRELA_TIER"] = "lunch-money"
        assert get_tier() == "lunch-money"

        # Clean up
        del os.environ["PRELA_TIER"]
        prela.license._current_tier = None

    def test_get_tier_defaults_to_free(self):
        """Test tier defaults to free if not set."""
        import prela.license
        prela.license._current_tier = None

        # Make sure PRELA_TIER is not set
        os.environ.pop("PRELA_TIER", None)

        assert get_tier() == "free"


class TestHasAccess:
    """Test has_access function."""

    def test_free_tier_access(self):
        """Test free tier access permissions."""
        assert has_access("free", "free") is True
        assert has_access("free", "lunch-money") is False
        assert has_access("free", "pro") is False
        assert has_access("free", "enterprise") is False

    def test_lunch_money_tier_access(self):
        """Test lunch-money tier access permissions."""
        assert has_access("lunch-money", "free") is True
        assert has_access("lunch-money", "lunch-money") is True
        assert has_access("lunch-money", "pro") is False
        assert has_access("lunch-money", "enterprise") is False

    def test_pro_tier_access(self):
        """Test pro tier access permissions."""
        assert has_access("pro", "free") is True
        assert has_access("pro", "lunch-money") is True
        assert has_access("pro", "pro") is True
        assert has_access("pro", "enterprise") is False

    def test_enterprise_tier_access(self):
        """Test enterprise tier access permissions."""
        assert has_access("enterprise", "free") is True
        assert has_access("enterprise", "lunch-money") is True
        assert has_access("enterprise", "pro") is True
        assert has_access("enterprise", "enterprise") is True

    def test_unknown_tier_denies_access(self):
        """Test unknown tier denies access."""
        assert has_access("unknown", "free") is False
        assert has_access("free", "unknown") is False


class TestRequireTierDecorator:
    """Test require_tier decorator."""

    def test_decorator_allows_access_with_correct_tier(self):
        """Test decorator allows access with correct tier."""
        set_tier("lunch-money")

        @require_tier("Test feature", "lunch-money")
        def test_function():
            return "success"

        assert test_function() == "success"

    def test_decorator_allows_access_with_higher_tier(self):
        """Test decorator allows access with higher tier."""
        set_tier("pro")

        @require_tier("Test feature", "lunch-money")
        def test_function():
            return "success"

        assert test_function() == "success"

    def test_decorator_blocks_access_with_lower_tier(self):
        """Test decorator blocks access with lower tier."""
        set_tier("free")

        @require_tier("Test feature", "lunch-money")
        def test_function():
            return "success"

        with pytest.raises(SubscriptionError) as exc_info:
            test_function()

        assert "Test feature" in str(exc_info.value)
        assert "lunch-money" in str(exc_info.value)
        assert "free" in str(exc_info.value)
        assert "https://prela.dev/pricing" in str(exc_info.value)

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""
        set_tier("lunch-money")

        @require_tier("Test feature", "lunch-money")
        def test_function():
            """Test docstring."""
            return "success"

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."


class TestCheckTier:
    """Test check_tier function."""

    def test_check_tier_returns_true_with_access(self):
        """Test check_tier returns True with access."""
        set_tier("lunch-money")
        assert check_tier("Test feature", "lunch-money", silent=True) is True

    def test_check_tier_returns_false_without_access(self):
        """Test check_tier returns False without access."""
        set_tier("free")
        assert check_tier("Test feature", "lunch-money", silent=True) is False

    def test_check_tier_logs_warning_when_not_silent(self):
        """Test check_tier logs warning when not silent."""
        set_tier("free")

        # This should not raise, just log
        result = check_tier("Test feature", "lunch-money", silent=False)
        assert result is False


class TestFeatureGatingScenarios:
    """Test real-world feature gating scenarios."""

    def test_free_tier_user_cannot_use_crewai(self):
        """Test free tier user cannot use CrewAI."""
        set_tier("free")

        @require_tier("CrewAI instrumentation", "lunch-money")
        def instrument_crewai():
            return "instrumented"

        with pytest.raises(SubscriptionError) as exc_info:
            instrument_crewai()

        assert "CrewAI instrumentation" in str(exc_info.value)
        assert "lunch-money" in str(exc_info.value)

    def test_lunch_money_user_can_use_crewai(self):
        """Test lunch-money user can use CrewAI."""
        set_tier("lunch-money")

        @require_tier("CrewAI instrumentation", "lunch-money")
        def instrument_crewai():
            return "instrumented"

        assert instrument_crewai() == "instrumented"

    def test_lunch_money_user_cannot_use_hallucination_detection(self):
        """Test lunch-money user cannot use hallucination detection."""
        set_tier("lunch-money")

        @require_tier("Hallucination detection", "pro")
        def detect_hallucinations():
            return "detected"

        with pytest.raises(SubscriptionError):
            detect_hallucinations()

    def test_pro_user_can_use_all_features(self):
        """Test pro user can use all lunch-money and pro features."""
        set_tier("pro")

        @require_tier("CrewAI instrumentation", "lunch-money")
        def instrument_crewai():
            return "instrumented"

        @require_tier("Hallucination detection", "pro")
        def detect_hallucinations():
            return "detected"

        assert instrument_crewai() == "instrumented"
        assert detect_hallucinations() == "detected"

    def test_enterprise_user_can_use_everything(self):
        """Test enterprise user can use all features."""
        set_tier("enterprise")

        @require_tier("EU AI Act compliance", "enterprise")
        def compliance_check():
            return "compliant"

        assert compliance_check() == "compliant"
