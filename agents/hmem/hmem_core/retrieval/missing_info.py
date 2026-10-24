"""Agentic follow-up when evidence is insufficient (community ``missing_info.py``).

Paper: adaptive retrieval may issue a secondary query when ranked evidence is weak.
Phase 3 optional; stub documents intended hook.
"""

from __future__ import annotations


def needs_follow_up(*, top_score: float, threshold: float = 0.35) -> bool:
    """Return whether retrieval should run a missing-info query.

    Args:
        top_score: Best F(m,Q,t) from ranking.
        threshold: Minimum acceptable score.

    Returns:
        True when follow-up retrieval is warranted.
    """
    return top_score < threshold
