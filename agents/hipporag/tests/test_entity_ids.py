"""Tests for phrase normalization and stable entity ids."""

from __future__ import annotations

from hipporag_core.entity_ids import entity_id, normalize_phrase


class TestNormalizePhrase:
    """Normalization rules for graph nodes."""

    def test_lowercase_and_strip(self) -> None:
        assert normalize_phrase("  Stanford University  ") == "stanford university"

    def test_collapse_whitespace(self) -> None:
        assert normalize_phrase("Prof   Thomas") == "prof thomas"


class TestEntityId:
    """Stable hashed ids for MERGE keys."""

    def test_same_phrase_same_id(self) -> None:
        a = entity_id("Professor Thomas")
        b = entity_id("  professor thomas ")
        assert a == b
        assert a.startswith("entity-")

    def test_different_phrases_different_ids(self) -> None:
        assert entity_id("Stanford") != entity_id("Harvard")
