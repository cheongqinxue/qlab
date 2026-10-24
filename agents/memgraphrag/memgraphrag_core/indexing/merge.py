"""Merge extraction payloads into ``ThreeLayerMemory``."""

from __future__ import annotations

from memgraphrag_core.memory.three_layer import ThreeLayerMemory
from memgraphrag_core.models import FactRecord


def apply_extraction(memory: ThreeLayerMemory, passage_id: str, payload: dict) -> list[str]:
    """Apply A_ext output for one passage.

    Args:
        memory: Global memory tiers.
        passage_id: Source passage id.
        payload: Dict with ``schemas`` and ``facts`` lists.

    Returns:
        New fact ids added in this call.
    """
    schema_ids: list[str] = []
    for schema in payload.get("schemas", []):
        sid = str(schema["id"])
        label = str(schema.get("label", sid))
        memory.merge_ontology(sid, label)
        schema_ids.append(sid)

    new_fact_ids: list[str] = []
    for fact in payload.get("facts", []):
        fid = str(fact["id"])
        memory.add_fact(
            FactRecord(
                id=fid,
                text=str(fact["text"]),
                passage_ids=list(fact.get("passage_ids", [passage_id])),
                schema_ids=list(fact.get("schema_ids", schema_ids)),
            )
        )
        new_fact_ids.append(fid)
    return new_fact_ids
