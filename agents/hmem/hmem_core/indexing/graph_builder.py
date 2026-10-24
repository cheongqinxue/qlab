"""Knowledge graph construction from extracted triples (paper §3.2).

Community reference: ``indexing/graph_builder.py`` (entity resolution, profiles).

Uses ``KnowledgeGraph.add_edge`` today; ``merge_entity`` when disambiguation lands.
"""

from __future__ import annotations

from hmem_core.graph.knowledge_graph import KnowledgeGraph
from hmem_core.models import GraphEdge, MemoryFragment


class GraphBuilder:
    """Append relations from fragments into the hybrid graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Attach to shared graph instance."""
        self._graph = graph

    def ingest_edges(self, fragment: MemoryFragment, edges: list[GraphEdge]) -> int:
        """Add triples and back-link fragment id.

        Args:
            fragment: Source fragment.
            edges: Extracted relations.

        Returns:
            Number of edges added.
        """
        for edge in edges:
            self._graph.add_edge(edge, fragment_id=fragment.id)
        return len(edges)
