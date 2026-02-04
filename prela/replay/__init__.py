"""Replay execution engine for deterministic re-execution of AI agent traces."""

from __future__ import annotations

# Check tier on module import
from prela.license import check_tier

if not check_tier("Replay engine", "lunch-money", silent=False):
    raise ImportError(
        "Replay engine requires 'lunch-money' subscription or higher. "
        "Upgrade at https://prela.dev/pricing"
    )

from prela.replay.comparison import ReplayComparator, compare_replays
from prela.replay.engine import ReplayEngine
from prela.replay.result import (
    ReplayComparison,
    ReplayResult,
    ReplayedSpan,
    SpanDifference,
)

__all__ = [
    "ReplayEngine",
    "ReplayResult",
    "ReplayedSpan",
    "ReplayComparison",
    "SpanDifference",
    "ReplayComparator",
    "compare_replays",
]
