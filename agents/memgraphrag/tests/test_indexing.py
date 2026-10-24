"""Tests for indexing orchestrator and graph builder."""

from __future__ import annotations

import asyncio

import pytest
from hipporag_core.phrase_encoder import PhraseEncoder

from memgraphrag_core.agents.conflict_detect import FixtureConflictDetectionAgent
from memgraphrag_core.agents.conflict_resolve import FixtureConflictResolutionAgent
from memgraphrag_core.agents.extraction import FixtureExtractionAgent
from memgraphrag_core.eval_loader import load_toy_conflicts, load_toy_extraction, load_toy_resolutions
from memgraphrag_core.indexing.graph_builder import IndexingGraph, build_from_memory
from memgraphrag_core.indexing.merge import apply_extraction
from memgraphrag_core.indexing.orchestrator import IndexOrchestrator
from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.models import PassageRecord


class TestIndexingGraph:
    """Graph projection."""

    def test_build_from_memory(self) -> None:
        encoder = PhraseEncoder()
        memory = MemGraphGlobalMemory(MemGraphMemoryConfig())
        memory.layers.add_passage(PassageRecord(id="p1", text="hello world"))
        apply_extraction(
            memory.layers,
            "p1",
            {
                "schemas": [{"id": "s1", "label": "Greeting"}],
                "facts": [{"id": "f1", "text": "hello", "passage_ids": ["p1"]}],
            },
        )
        graph = build_from_memory(memory.layers, encoder)
        assert isinstance(graph, IndexingGraph)
        assert graph.node_count >= 3
        assert graph.edge_count >= 1


class TestIndexOrchestrator:
    """Offline index pipeline entry."""

    def test_run_full_index_toy(self, tmp_path) -> None:
        cfg = MemGraphMemoryConfig(memory_data_dir=str(tmp_path / "mem"))
        memory = MemGraphGlobalMemory(cfg)
        encoder = PhraseEncoder()
        memory.set_encoder(encoder)
        memory.layers.add_passage(
            PassageRecord(id="p1", text="Isaac Newton was born in 1642.")
        )
        memory.layers.add_passage(
            PassageRecord(id="p2", text="Some cite Newton's birth as 1643.")
        )
        orchestrator = IndexOrchestrator(
            memory,
            encoder,
            extraction=FixtureExtractionAgent(load_toy_extraction()),
            detection=FixtureConflictDetectionAgent(load_toy_conflicts()),
            resolution=FixtureConflictResolutionAgent(load_toy_resolutions()),
        )

        async def _run() -> dict[str, int]:
            return await orchestrator.run_full_index()

        summary = asyncio.run(_run())
        assert summary["facts"] == 1
        assert summary["conflicts_resolved"] == 1
        assert memory.index_graph is not None
