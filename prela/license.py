"""Tier-based feature gating for Prela SDK.

This module provides decorators and utilities for restricting features
to specific subscription tiers.
"""

import functools
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SubscriptionError(Exception):
    """Raised when a feature requires a higher subscription tier."""

    pass


# Tier hierarchy (lower index = lower tier)
TIER_HIERARCHY = ["free", "lunch-money", "pro", "enterprise"]

# Current tier (detected from environment or API key)
_current_tier: Optional[str] = None


def set_tier(tier: str):
    """Set the current subscription tier.

    This is called automatically when initializing the HTTP exporter
    with an API key.

    Args:
        tier: Subscription tier (free, lunch-money, pro, enterprise).
    """
    global _current_tier

    if tier not in TIER_HIERARCHY:
        logger.warning(f"Unknown tier: {tier}, defaulting to free")
        tier = "free"

    _current_tier = tier
    logger.debug(f"Subscription tier set to: {tier}")


def get_tier() -> str:
    """Get the current subscription tier.

    Returns:
        Current tier string. Defaults to "free" if not set.
    """
    global _current_tier

    # If not set, try to get from environment
    if _current_tier is None:
        env_tier = os.environ.get("PRELA_TIER", "free")
        set_tier(env_tier)

    return _current_tier or "free"


def has_access(current_tier: str, required_tier: str) -> bool:
    """Check if current tier has access to a feature requiring a specific tier.

    Args:
        current_tier: User's current subscription tier.
        required_tier: Tier required for the feature.

    Returns:
        True if user has access, False otherwise.
    """
    try:
        current_idx = TIER_HIERARCHY.index(current_tier)
        required_idx = TIER_HIERARCHY.index(required_tier)
        return current_idx >= required_idx
    except ValueError:
        # Unknown tier, deny access
        return False


def require_tier(feature_name: str, required_tier: str):
    """Decorator to restrict a feature to a specific subscription tier.

    Usage:
        @require_tier("CrewAI instrumentation", "lunch-money")
        def instrument_crewai():
            pass

    Args:
        feature_name: Human-readable name of the feature.
        required_tier: Minimum tier required (free, lunch-money, pro, enterprise).

    Returns:
        Decorator function.

    Raises:
        SubscriptionError: If user doesn't have required tier.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_tier = get_tier()

            if not has_access(current_tier, required_tier):
                raise SubscriptionError(
                    f"\n\n"
                    f"🔒 {feature_name} requires '{required_tier}' subscription or higher.\n"
                    f"   Current tier: '{current_tier}'\n\n"
                    f"   Upgrade at: https://prela.dev/pricing\n\n"
                    f"   Features by tier:\n"
                    f"   • free: OpenAI, Anthropic, LangChain, LlamaIndex (basic)\n"
                    f"   • lunch-money: + CrewAI, AutoGen, LangGraph, Swarm, n8n, replay, semantic assertions\n"
                    f"   • pro: + hallucination detection, drift detection, natural language search\n"
                    f"   • enterprise: + compliance features, dedicated support\n"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_tier_async(feature_name: str, required_tier: str):
    """Async version of require_tier decorator.

    Usage:
        @require_tier_async("CrewAI instrumentation", "lunch-money")
        async def instrument_crewai():
            pass

    Args:
        feature_name: Human-readable name of the feature.
        required_tier: Minimum tier required.

    Returns:
        Async decorator function.

    Raises:
        SubscriptionError: If user doesn't have required tier.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_tier = get_tier()

            if not has_access(current_tier, required_tier):
                raise SubscriptionError(
                    f"\n\n"
                    f"🔒 {feature_name} requires '{required_tier}' subscription or higher.\n"
                    f"   Current tier: '{current_tier}'\n\n"
                    f"   Upgrade at: https://prela.dev/pricing\n"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def check_tier(feature_name: str, required_tier: str, silent: bool = False) -> bool:
    """Check if current tier has access to a feature.

    This is a non-decorator version for conditional logic.

    Usage:
        if check_tier("Replay engine", "lunch-money"):
            # Enable replay features
            pass
        else:
            print("Upgrade to use replay")

    Args:
        feature_name: Human-readable name of the feature.
        required_tier: Minimum tier required.
        silent: If True, don't log warnings.

    Returns:
        True if user has access, False otherwise.
    """
    current_tier = get_tier()
    has_feature = has_access(current_tier, required_tier)

    if not has_feature and not silent:
        logger.warning(
            f"{feature_name} requires '{required_tier}' tier. "
            f"Current tier: '{current_tier}'. "
            f"Upgrade at https://prela.dev/pricing"
        )

    return has_feature


def get_tier_features() -> dict[str, list[str]]:
    """Get a dictionary of features available in each tier.

    Returns:
        Dictionary mapping tier names to lists of features.
    """
    return {
        "free": [
            "Basic tracing (traces, spans, context)",
            "OpenAI & Anthropic instrumentation",
            "LangChain & LlamaIndex (basic)",
            "Console & File exporters",
            "Basic CLI commands",
            "Local storage",
        ],
        "lunch-money": [
            "All free features",
            "CrewAI, AutoGen, LangGraph, Swarm, n8n",
            "All 17+ assertion types",
            "Semantic similarity assertions",
            "Multi-agent assertions",
            "Replay engine (100/month)",
            "HTTP exporter (cloud sync)",
            "100k traces/month",
            "30-day retention",
        ],
        "pro": [
            "All lunch-money features",
            "Hallucination detection",
            "Drift detection with alerts",
            "Natural language search",
            "One-click debug flow",
            "Cost optimization",
            "Batch replay (50 traces)",
            "1M traces/month",
            "90-day retention",
        ],
        "enterprise": [
            "All pro features",
            "EU AI Act compliance",
            "Data lineage tracking",
            "Custom model cards",
            "SSO/SAML",
            "Dedicated infrastructure",
            "Unlimited traces",
            "Custom retention",
        ],
    }
