"""In-memory three-layer global memory (M_ont, M_fac, M_pas)."""

from __future__ import annotations

from collections.abc import Iterator

from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord


class ThreeLayerMemory:
    """Ontology, fact, and passage tiers with corpus-wide consistency."""

    def __init__(self) -> None:
        """Initialize empty memory tiers."""
        self._ontology: dict[str, OntologySchemaRecord] = {}
        self._facts: dict[str, FactRecord] = {}
        self._passages: dict[str, PassageRecord] = {}

    def add_passage(self, record: PassageRecord) -> None:
        """Register or replace a passage in M_pas."""
        self._passages[record.id] = record

    def add_fact(self, record: FactRecord) -> None:
        """Register or replace a fact in M_fac."""
        self._facts[record.id] = record

    def add_ontology(self, record: OntologySchemaRecord) -> None:
        """Register or replace a schema entry in M_ont."""
        self._ontology[record.id] = record

    def merge_ontology(self, schema_id: str, label: str, *, increment: int = 1) -> None:
        """Insert or bump extraction frequency for a schema."""
        existing = self._ontology.get(schema_id)
        if existing is None:
            self.add_ontology(
                OntologySchemaRecord(id=schema_id, label=label, extraction_count=increment)
            )
            return
        self._ontology[schema_id] = existing.model_copy(
            update={"extraction_count": existing.extraction_count + increment}
        )

    def get_passage(self, passage_id: str) -> PassageRecord | None:
        """Return passage by id."""
        return self._passages.get(passage_id)

    def get_fact(self, fact_id: str) -> FactRecord | None:
        """Return fact by id."""
        return self._facts.get(fact_id)

    def delete_fact(self, fact_id: str) -> bool:
        """Remove one fact from M_fac after conflict resolution."""
        return self._facts.pop(fact_id, None) is not None

    def delete_passage(self, passage_id: str) -> bool:
        """Remove a passage from M_pas."""
        return self._passages.pop(passage_id, None) is not None

    def iter_passages(self) -> Iterator[PassageRecord]:
        """Iterate passages in stable id order."""
        for pid in sorted(self._passages.keys()):
            yield self._passages[pid]

    def iter_facts(self) -> Iterator[FactRecord]:
        """Iterate facts in stable id order."""
        for fid in sorted(self._facts.keys()):
            yield self._facts[fid]

    def iter_ontology(self) -> Iterator[OntologySchemaRecord]:
        """Iterate ontology entries in stable id order."""
        for sid in sorted(self._ontology.keys()):
            yield self._ontology[sid]

    def clear(self) -> None:
        """Remove all tiers."""
        self._ontology.clear()
        self._facts.clear()
        self._passages.clear()

    @property
    def ontology_count(self) -> int:
        """Number of ontology schema entries."""
        return len(self._ontology)

    @property
    def fact_count(self) -> int:
        """Number of stored facts."""
        return len(self._facts)

    @property
    def passage_count(self) -> int:
        """Number of stored passages."""
        return len(self._passages)
