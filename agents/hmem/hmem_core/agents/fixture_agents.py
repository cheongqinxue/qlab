"""Fixture implementations for tests and ``HMEM_AGENTS=fixture``."""

from __future__ import annotations

import json
import re

from hmem_core.models import GraphEdge, MemoryEvent, MemoryFragment, MemoryScope, SubQuery


class FixtureEventExtractor:
    """One event per fragment; optional capitalized entity heuristics."""

    def extract(self, fragment: MemoryFragment) -> tuple[list[MemoryEvent], list[GraphEdge]]:
        """Mirror fragment text as a single leaf event."""
        event = MemoryEvent(
            event_id=f"ev_{fragment.id}",
            fragment_id=fragment.id,
            text=fragment.text,
            timestamp=fragment.timestamp,
        )
        edges: list[GraphEdge] = []
        names = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", fragment.text)
        for index, name in enumerate(names):
            if index + 1 < len(names):
                continue
            edges.append(GraphEdge(head=name, relation="mentioned_in", tail=fragment.id))
        return [event], edges


class FixtureSummarizer:
    """Concatenate child texts (placeholder for LLM summary)."""

    def summarize(self, child_texts: list[str]) -> str:
        """Join child snippets."""
        return " | ".join(child_texts)


class FixturePlanner:
    """Single sub-query with MIXED scope."""

    def plan(self, query: str) -> list[SubQuery]:
        """Pass through full query."""
        return [SubQuery(text=query, scope=MemoryScope.MIXED)]


class FixtureSynthesizer:
    """Heuristic answer from ranked evidence (no LLM)."""

    def synthesize(self, query: str, evidence_texts: list[str]) -> str:
        """Return a short answer extracted from evidence."""
        if not evidence_texts:
            return ""
        corpus = " ".join(evidence_texts)
        if "name" in query.lower():
            for text in evidence_texts:
                named = re.search(r"cat named (\w+)", text, flags=re.IGNORECASE)
                if named:
                    return named.group(1)
                if "mochi" in text.lower():
                    return "Mochi"
            match = re.search(r"\b([A-Z][a-z]+)\b", corpus)
            if match:
                return match.group(1)
        return evidence_texts[0][:240]
