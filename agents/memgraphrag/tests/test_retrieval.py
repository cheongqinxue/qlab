"""Tests for memory-guided retrieval."""

from __future__ import annotations

import asyncio

import pytest
from hipporag_core.phrase_encoder import PhraseEncoder

from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.models import FactRecord, OntologySchemaRecord, PassageRecord
from memgraphrag_core.retrieval.layer_filter import filter_layers_by_query
from memgraphrag_core.retrieval.ppr_seeds import build_ppr_seeds
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever


class TestRetrieval:
    """Filter, seeds, and retriever."""

    @pytest.fixture
    def memory_with_graph(self) -> MemGraphGlobalMemory:
        encoder = PhraseEncoder()
        memory = MemGraphGlobalMemory(MemGraphMemoryConfig(retrieval_sim_threshold=0.0))
        memory.set_encoder(encoder)
        memory.layers.add_passage(PassageRecord(id="p1", text="Isaac Newton born 1642"))
        memory.layers.merge_ontology("s1", "BirthYear")
        memory.layers.add_fact(
            FactRecord(
                id="f1",
                text="Newton born 1642",
                passage_ids=["p1"],
                schema_ids=["s1"],
            )
        )
        memory.rebuild_index_graph(encoder)
        return memory

    def test_filter_layers_returns_nodes(self, memory_with_graph: MemGraphGlobalMemory) -> None:
        encoder = memory_with_graph._encoder
        assert encoder is not None
        hits = filter_layers_by_query(
            memory_with_graph.layers,
            encoder,
            "Newton birth year",
            sim_threshold=0.0,
        )
        assert len(hits) >= 1

    def test_build_ppr_seeds(self, memory_with_graph: MemGraphGlobalMemory) -> None:
        graph = memory_with_graph.index_graph
        assert graph is not None
        encoder = memory_with_graph._encoder
        assert encoder is not None
        filtered = filter_layers_by_query(
            memory_with_graph.layers,
            encoder,
            "Newton",
            sim_threshold=0.0,
        )
        reset = build_ppr_seeds(
            filtered,
            num_nodes=graph.node_count,
            key_to_index=graph.key_to_index(),
        )
        assert reset.sum() > 0

    def test_retriever_returns_passage(self, memory_with_graph: MemGraphGlobalMemory) -> None:
        encoder = memory_with_graph._encoder
        assert encoder is not None
        retriever = MemoryGuidedRetriever(
            memory_with_graph,
            encoder,
            memory_with_graph.config,
        )

        async def _run():
            return await retriever.retrieve("Newton birth", top_k=1)

        hits = asyncio.run(_run())
        assert len(hits) == 1
        assert hits[0].id == "p1"
