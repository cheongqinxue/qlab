"""Prompt templates for MemGraphRAG indexing agents."""

EXTRACTION_SYSTEM = """You extract structured global memory from one passage.

Return ontology schemas (reusable types) and concrete facts grounded in this passage.
Use stable lowercase ids: schemas like s_person_birth_year, facts like f_entity_attr_value.
Each fact must include passage_ids containing the source passage id provided.
Link each fact to relevant schema_ids from the schemas you emit."""

EXTRACTION_HUMAN = """Passage id: {passage_id}

Passage text:
{text}
"""

CONFLICT_DETECTION_SYSTEM = """You detect HARD conflicts among facts in global memory.

Hard conflicts: mutually exclusive values for the same entity+attribute, incompatible dates,
or contradictory statements that cannot all be true.

Only report groups where the target_id is in the NEW_FACT_IDS list and at least one
related_id is another fact in the corpus."""

CONFLICT_DETECTION_HUMAN = """NEW_FACT_IDS: {new_fact_ids}

ALL_FACTS_JSON:
{facts_json}
"""

CONFLICT_RESOLUTION_SYSTEM = """You resolve a hard conflict using passage evidence.

Prefer facts with clearer primary-source support. Return fact ids to drop and to keep.
Only drop facts that are clearly contradicted by stronger evidence."""

CONFLICT_RESOLUTION_HUMAN = """CONFLICT_GROUP_JSON:
{group_json}

PASSAGE_EVIDENCE_JSON:
{passages_json}
"""
