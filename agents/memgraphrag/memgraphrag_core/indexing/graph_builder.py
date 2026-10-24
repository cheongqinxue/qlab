"""Build the hierarchical indexing graph from three-layer memory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hipporag_core.phrase_encoder import PhraseEncoder
from memgraphrag_core.memory.three_layer import ThreeLayerMemory


@dataclass(frozen=True)
class GraphNodeKey:
    """Stable node identity in the indexing graph."""

    kind: str
    record_id: str


@dataclass
class IndexingGraph:
    """In-memory hierarchical graph for memory-guided PPR."""

    node_keys: list[GraphNodeKey] = field(default_factory=list)
    node_texts: list[str] = field(default_factory=list)
    node_vectors: np.ndarray | None = None
    edges: list[tuple[int, int, float]] = field(default_factory=list)
    passage_indices: list[int] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return len(self.node_keys)

    @property
    def edge_count(self) -> int:
        """Number of undirected edges."""
        return len(self.edges)

    def key_to_index(self) -> dict[tuple[str, str], int]:
        """Map ``(kind, record_id)`` → node index."""
        return {(k.kind, k.record_id): i for i, k in enumerate(self.node_keys)}


def build_from_memory(memory: ThreeLayerMemory, encoder: PhraseEncoder) -> IndexingGraph:
    """Project M_ont/M_fac/M_pas into an ``IndexingGraph``."""
    graph = IndexingGraph()
    index = graph.key_to_index()

    def ensure_node(kind: str, record_id: str, text: str) -> int:
        key = (kind, record_id)
        if key in index:
            return index[key]
        idx = len(graph.node_keys)
        graph.node_keys.append(GraphNodeKey(kind=kind, record_id=record_id))
        graph.node_texts.append(text)
        index[key] = idx
        if kind == "passage":
            graph.passage_indices.append(idx)
        return idx

    for schema in memory.iter_ontology():
        ensure_node("schema", schema.id, schema.label)
    for fact in memory.iter_facts():
        ensure_node("fact", fact.id, fact.text)
    for passage in memory.iter_passages():
        ensure_node("passage", passage.id, passage.text)

    def add_edge(a: int, b: int, weight: float = 1.0) -> None:
        if a == b:
            return
        graph.edges.append((a, b, weight))

    idx_map = graph.key_to_index()
    for fact in memory.iter_facts():
        fact_idx = idx_map[("fact", fact.id)]
        for sid in fact.schema_ids:
            key = ("schema", sid)
            if key in idx_map:
                add_edge(idx_map[key], fact_idx, 1.0)
        for pid in fact.passage_ids:
            key = ("passage", pid)
            if key in idx_map:
                add_edge(fact_idx, idx_map[key], 1.0)

    if graph.node_texts:
        vectors = np.stack([encoder.encode(t) for t in graph.node_texts], axis=0)
        graph.node_vectors = vectors
    else:
        graph.node_vectors = np.zeros((0, 8), dtype=np.float64)

    return graph
