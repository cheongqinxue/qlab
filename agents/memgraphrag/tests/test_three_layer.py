"""Tests for ``ThreeLayerMemory``."""

from __future__ import annotations

from memgraphrag_core.memory.three_layer import ThreeLayerMemory
from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord


class TestThreeLayerMemory:
    """M_ont / M_fac / M_pas operations."""

    def test_add_and_counts(self) -> None:
        memory = ThreeLayerMemory()
        memory.add_passage(PassageRecord(id="p1", text="hello"))
        memory.add_fact(FactRecord(id="f1", text="fact", passage_ids=["p1"]))
        memory.add_ontology(OntologySchemaRecord(id="s1", label="Type", extraction_count=1))
        assert memory.passage_count == 1
        assert memory.fact_count == 1
        assert memory.ontology_count == 1

    def test_delete_fact(self) -> None:
        memory = ThreeLayerMemory()
        memory.add_fact(FactRecord(id="f1", text="x", passage_ids=[]))
        assert memory.delete_fact("f1") is True
        assert memory.fact_count == 0
        assert memory.delete_fact("missing") is False

    def test_merge_ontology_increments(self) -> None:
        memory = ThreeLayerMemory()
        memory.merge_ontology("s1", "Type")
        memory.merge_ontology("s1", "Type", increment=2)
        for schema in memory.iter_ontology():
            assert schema.extraction_count == 3

    def test_delete_passage(self) -> None:
        memory = ThreeLayerMemory()
        memory.add_passage(PassageRecord(id="p1", text="a"))
        assert memory.delete_passage("p1") is True
        assert memory.passage_count == 0

    def test_clear(self) -> None:
        memory = ThreeLayerMemory()
        memory.add_passage(PassageRecord(id="p1", text="a"))
        memory.add_fact(FactRecord(id="f1", text="b", passage_ids=["p1"]))
        memory.clear()
        assert memory.passage_count == 0
        assert memory.fact_count == 0
        assert memory.ontology_count == 0
