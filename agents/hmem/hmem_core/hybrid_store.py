"""In-memory hybrid tree + graph store.

Owns the paper's dual index: ``TemporalSemanticTree`` + ``KnowledgeGraph``.
Legacy flat ``events`` / ``edges`` lists remain for fixture bootstrap counts until
Phase 2 routes all ingest through ``HybridIndexer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hmem_core.config import HMemConfig
from hmem_core.graph.knowledge_graph import KnowledgeGraph
from hmem_core.models import GraphEdge, MemoryEvent, MemoryFragment
from hmem_core.tree.temporal_semantic_tree import TemporalSemanticTree


@dataclass
class InMemoryHybridStore:
    """Mutable hybrid index."""

    config: HMemConfig
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    tree: TemporalSemanticTree | None = None
    fragments: dict[str, MemoryFragment] = field(default_factory=dict)
    events: list[MemoryEvent] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    summary_nodes: int = 0
    indexed: bool = False

    def __post_init__(self) -> None:
        """Ensure tree exists for configured levels."""
        if self.tree is None:
            self.tree = TemporalSemanticTree(config=self.config)

    @property
    def fragment_count(self) -> int:
        """Number of stored original fragments."""
        return len(self.fragments)

    @property
    def leaf_count(self) -> int:
        """Number of leaf memory events."""
        return len(self.events)

    @property
    def entity_count(self) -> int:
        """Distinct entity nodes in the knowledge graph."""
        return len(self.graph.entities)

    def clear(self) -> None:
        """Remove all indexed memory."""
        self.fragments.clear()
        self.events.clear()
        self.edges.clear()
        self.graph = KnowledgeGraph()
        self.tree = TemporalSemanticTree(config=self.config)
        self.summary_nodes = 0
        self.indexed = False
