"""Graph-side evidence seeding and multi-hop expansion (paper §3.3)."""

from __future__ import annotations

import re

from hmem_core.config import HMemConfig
from hmem_core.graph.knowledge_graph import KnowledgeGraph


class GraphSearcher:
    """Locate seed entities and expand along relations."""

    def __init__(self, graph: KnowledgeGraph, config: HMemConfig) -> None:
        """Configure graph and hop limit."""
        self._graph = graph
        self._max_hops = config.graph_max_hops

    def seed_entities_from_text(self, text: str) -> set[str]:
        """Lexical + vocabulary match against known entities."""
        seeds: set[str] = set()
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9']+", text))
        lowered = text.lower()
        for entity in self._graph.entities:
            if entity.lower() in lowered or entity in tokens:
                seeds.add(entity)
        for token in tokens:
            seeds.add(self._graph.resolve_entity(token))
        return seeds

    def expand_from_query_entities(self, entity_names: list[str]) -> set[str]:
        """Multi-hop entity set from seed mentions."""
        found: set[str] = set()
        for name in entity_names:
            canonical = self._graph.resolve_entity(name)
            found.update(self._graph.neighbors(canonical, max_hops=self._max_hops))
        return found

    def fragment_ids_for_entities(self, entities: set[str]) -> set[str]:
        """Map entities back to source fragments."""
        result: set[str] = set()
        for fragment_id, names in self._graph.fragment_entities.items():
            if names & entities:
                result.add(fragment_id)
        return result
