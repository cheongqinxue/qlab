"""Filter M_ont, M_fac, and M_pas by query–memory similarity."""

from __future__ import annotations

from dataclasses import dataclass

from agents_utils.memory.base_encoder import BaseEncoder
from memgraphrag_core.memory.three_layer import ThreeLayerMemory


@dataclass(frozen=True)
class FilteredNode:
    """One memory node above the similarity threshold."""

    kind: str
    record_id: str
    similarity: float


def filter_layers_by_query(
    memory: ThreeLayerMemory,
    encoder: BaseEncoder,
    query: str,
    *,
    sim_threshold: float,
) -> list[FilteredNode]:
    """Return graph node ids per layer that exceed similarity threshold."""
    query_vec = encoder.encode_query_numpy(query)
    hits: list[FilteredNode] = []

    def consider(kind: str, record_id: str, text: str) -> None:
        sim = float(query_vec @ encoder.encode_document_numpy(text))
        if sim >= sim_threshold:
            hits.append(FilteredNode(kind=kind, record_id=record_id, similarity=sim))

    for schema in memory.iter_ontology():
        consider("schema", schema.id, schema.label)
    for fact in memory.iter_facts():
        consider("fact", fact.id, fact.text)
    for passage in memory.iter_passages():
        consider("passage", passage.id, passage.text)
    return hits
