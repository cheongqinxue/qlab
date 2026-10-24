"""Tests for Pydantic memory records."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord


class TestPassageRecord:
    """M_pas record."""

    def test_valid(self) -> None:
        rec = PassageRecord(id="p1", text="body")
        assert rec.id == "p1"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            PassageRecord(id="p1", text="")


class TestFactRecord:
    """M_fac record."""

    def test_default_passage_ids(self) -> None:
        rec = FactRecord(id="f1", text="fact")
        assert rec.passage_ids == []


class TestOntologySchemaRecord:
    """M_ont record."""

    def test_extraction_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            OntologySchemaRecord(id="s1", label="T", extraction_count=-1)
