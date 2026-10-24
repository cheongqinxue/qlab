"""Entity knowledge graph for multi-hop memory access (paper §3.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from hmem_core.models import EntityNode, GraphEdge


def _normalize_entity(name: str) -> str:
    """Lowercase alphanumeric tokens for matching."""
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


@dataclass
class KnowledgeGraph:
    """In-memory directed multigraph with entity resolution."""

    entities: dict[str, EntityNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    fragment_entities: dict[str, set[str]] = field(default_factory=dict)
    _alias_map: dict[str, str] = field(default_factory=dict)

    def resolve_entity(self, name: str, *, entity_type: str | None = None) -> str:
        """Normalize and deduplicate entity names (paper disambiguation lite).

        Args:
            name: Mention string from extraction.
            entity_type: Optional type hint for merge compatibility.

        Returns:
            Canonical entity key stored in ``entities``.
        """
        norm = _normalize_entity(name)
        if norm in self._alias_map:
            return self._alias_map[norm]
        for canonical, node in self.entities.items():
            if _normalize_entity(canonical) == norm:
                self._alias_map[norm] = canonical
                return canonical
            if SequenceMatcher(None, norm, _normalize_entity(canonical)).ratio() >= 0.92:
                self._alias_map[norm] = canonical
                return canonical
        self.entities[name] = EntityNode(name=name, entity_type=entity_type)
        self._alias_map[norm] = name
        return name

    def merge_entity(self, alias: str, canonical: str) -> None:
        """Force alias to canonical entity."""
        resolved = self.resolve_entity(canonical)
        norm = _normalize_entity(alias)
        self._alias_map[norm] = resolved

    def add_edge(self, edge: GraphEdge, *, fragment_id: str | None = None) -> None:
        """Insert a relation between resolved entities."""
        head = self.resolve_entity(edge.head)
        tail = self.resolve_entity(edge.tail)
        canonical = GraphEdge(head=head, relation=edge.relation, tail=tail)
        if canonical not in self.edges:
            self.edges.append(canonical)
        if fragment_id:
            self.fragment_entities.setdefault(fragment_id, set()).update({head, tail})

    def neighbors(self, entity: str, *, max_hops: int = 2) -> set[str]:
        """Multi-hop entity expansion."""
        canonical = self.resolve_entity(entity)
        frontier = {canonical}
        visited = {canonical}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for edge in self.edges:
                if edge.head in frontier:
                    next_frontier.add(edge.tail)
                if edge.tail in frontier:
                    next_frontier.add(edge.head)
            next_frontier -= visited
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        return visited
