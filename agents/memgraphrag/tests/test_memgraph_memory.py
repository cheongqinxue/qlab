"""Tests for ``MemGraphGlobalMemory``."""

from __future__ import annotations

import asyncio

import pytest
from hipporag_core.phrase_encoder import PhraseEncoder

from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.models import FactRecord, PassageRecord
from memgraphrag_core.storage.memory_store import save_memory


class TestMemGraphGlobalMemory:
    """BaseMemory adapter."""

    def test_build_or_restore(self) -> None:
        inst = MemGraphGlobalMemory.build_or_restore(MemGraphMemoryConfig())
        assert isinstance(inst, MemGraphGlobalMemory)
        assert inst.passage_count == 0

    def test_build_or_restore_loads_disk(self, tmp_path) -> None:
        cfg = MemGraphMemoryConfig(memory_data_dir=str(tmp_path))
        memory = MemGraphGlobalMemory(cfg)
        memory.layers.add_passage(PassageRecord(id="p1", text="saved"))
        save_memory(memory.layers, tmp_path)
        restored = MemGraphGlobalMemory.build_or_restore(cfg, encoder=PhraseEncoder())
        assert restored.passage_count == 1

    def test_add_and_afind(self) -> None:
        memory = MemGraphGlobalMemory(
            MemGraphMemoryConfig(retrieval_sim_threshold=0.0),
            encoder=PhraseEncoder(),
        )

        async def _run() -> None:
            await memory.add({"id": "p1", "text": "Einstein developed relativity"})
            memory.layers.merge_ontology("s1", "Science")
            memory.layers.add_fact(
                FactRecord(
                    id="f1",
                    text="Einstein relativity",
                    passage_ids=["p1"],
                    schema_ids=["s1"],
                )
            )
            memory.rebuild_index_graph()
            hits = await memory.afind(query="Einstein relativity", metadata={"top_k": 1})
            assert len(hits) == 1

        asyncio.run(_run())

    def test_adelete_fact_and_passage(self) -> None:
        memory = MemGraphGlobalMemory(MemGraphMemoryConfig(), encoder=PhraseEncoder())
        memory.layers.add_passage(PassageRecord(id="p1", text="x"))
        memory.layers.add_fact(
            FactRecord(id="f1", text="f", passage_ids=["p1"]),
        )
        memory.rebuild_index_graph()

        async def _run() -> None:
            assert await memory.adelete("f1") is True
            assert await memory.adelete("p1") is True
            assert await memory.adelete("missing") is False

        asyncio.run(_run())

    def test_aclear(self) -> None:
        memory = MemGraphGlobalMemory(MemGraphMemoryConfig())
        memory.layers.add_passage(PassageRecord(id="p1", text="x"))

        async def _run() -> None:
            await memory.aclear()

        asyncio.run(_run())
        assert memory.passage_count == 0
        assert memory.index_graph is None

    def test_add_type_error(self) -> None:
        memory = MemGraphGlobalMemory(MemGraphMemoryConfig())

        async def _run() -> None:
            with pytest.raises(TypeError):
                await memory.add(123)

        asyncio.run(_run())
