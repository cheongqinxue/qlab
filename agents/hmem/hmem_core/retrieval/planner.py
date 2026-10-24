"""Retrieval planning — sub-query decomposition (paper §3.3 step 1).

Community reference: ``retrieval/planner.py`` (JSON sub-queries + scope labels).
"""

from __future__ import annotations

from hmem_core.agents.fixture_agents import FixturePlanner
from hmem_core.agents.llm_agents import LlmRetrievalPlanner
from hmem_core.agents.protocols import RetrievalPlannerAgent
from hmem_core.models import SubQuery


def build_planner(*, agents_mode: str) -> RetrievalPlannerAgent:
    """Fixture or LLM planner factory."""
    if agents_mode == "llm":
        return LlmRetrievalPlanner()
    return FixturePlanner()


class RetrievalPlanner:
    """Thin wrapper around ``RetrievalPlannerAgent``."""

    def __init__(self, agent: RetrievalPlannerAgent) -> None:
        """Store agent implementation."""
        self._agent = agent

    def decompose(self, query: str) -> list[SubQuery]:
        """Split query into scoped sub-queries."""
        return self._agent.plan(query)
