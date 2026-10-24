"""Protocols for LLM-backed H-MEM steps (fixture implementations in ``fixture_agents``).

Each protocol maps to a paper stage; LLM wiring uses ``agents_utils.llm.openai_chat``.
"""

from __future__ import annotations

from typing import Protocol

from hmem_core.models import GraphEdge, MemoryEvent, MemoryFragment, SubQuery


class EventExtractor(Protocol):
    """Extract leaf memory events and graph triples from a fragment (§3.2)."""

    def extract(self, fragment: MemoryFragment) -> tuple[list[MemoryEvent], list[GraphEdge]]:
        """Return events and relations for one fragment."""


class SummarizerAgent(Protocol):
    """Consolidate child texts into a parent summary node (§3.2)."""

    def summarize(self, child_texts: list[str]) -> str:
        """Produce summary text for a new or updated internal node."""


class RetrievalPlannerAgent(Protocol):
    """Decompose query into sub-queries with scope labels (§3.3 step 1)."""

    def plan(self, query: str) -> list[SubQuery]:
        """Return ordered sub-queries with SHORT/LONG/MIXED scopes."""


class SynthesizerAgent(Protocol):
    """Generate sub-answers and final answer from evidence (§3.3 step 3)."""

    def synthesize(self, query: str, evidence_texts: list[str]) -> str:
        """Return natural-language answer."""
