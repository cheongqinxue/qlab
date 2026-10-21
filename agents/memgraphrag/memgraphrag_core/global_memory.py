"""Three-tier global memory store (stub for Phase 1)."""

from __future__ import annotations

from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord


class GlobalMemory:
    """In-memory placeholder for ontology, fact, and passage layers."""

    def __init__(self) -> None:
        """Initialize empty memory tiers."""
        self._ontology: dict[str, OntologySchemaRecord] = {}
        self._facts: dict[str, FactRecord] = {}
        self._passages: dict[str, PassageRecord] = {}

    def add_passage(self, record: PassageRecord) -> None:
        """Store a passage in M_pas."""
        self._passages[record.id] = record

    def add_fact(self, record: FactRecord) -> None:
        """Store a fact in M_fac."""
        self._facts[record.id] = record

    def add_ontology(self, record: OntologySchemaRecord) -> None:
        """Store or update an ontology entry in M_ont."""
        self._ontology[record.id] = record

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
