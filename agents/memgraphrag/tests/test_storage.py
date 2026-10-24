"""Tests for persistence and Neo4j projection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memgraphrag_core.memory.three_layer import ThreeLayerMemory
from memgraphrag_core.models import FactRecord, PassageRecord
from memgraphrag_core.storage.memory_store import load_memory, memory_json_path, save_memory
from memgraphrag_core.storage.neo4j_store import Neo4jMemoryProjection


class TestMemoryStore:
    """JSON snapshot helpers."""

    def test_memory_json_path(self, tmp_path) -> None:
        path = memory_json_path(tmp_path / "data")
        assert path.name == "memory.json"

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        memory = ThreeLayerMemory()
        memory.add_passage(PassageRecord(id="p1", text="hello"))
        memory.add_fact(FactRecord(id="f1", text="fact", passage_ids=["p1"]))
        data_dir = tmp_path / "mem"
        written = save_memory(memory, data_dir)
        assert written.is_file()
        restored = load_memory(data_dir)
        assert restored.passage_count == 1
        assert restored.fact_count == 1


class TestNeo4jMemoryProjection:
    """Neo4j dual-write."""

    def test_close_no_op(self) -> None:
        proj = Neo4jMemoryProjection("bolt://localhost:7687", "neo4j", "pass")
        proj.close()

    def test_sync_layers_with_mock_driver(self) -> None:
        memory = ThreeLayerMemory()
        memory.add_passage(PassageRecord(id="p1", text="text"))
        memory.add_fact(
            FactRecord(id="f1", text="fact", passage_ids=["p1"], schema_ids=["s1"])
        )
        proj = Neo4jMemoryProjection("bolt://localhost:7687", "neo4j", "pass")
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        with patch.object(proj, "_get_driver", return_value=mock_driver):
            proj.sync_layers(memory, None)
        assert mock_session.run.call_count >= 1
