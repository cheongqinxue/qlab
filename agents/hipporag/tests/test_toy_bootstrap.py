"""Toy corpus bootstrap (fixture OpenIE, no LLM)."""

from __future__ import annotations

from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.phrase_encoder import PhraseEncoder
from hipporag_core.retriever import Retriever
from hipporag_core.runtime import HipporagRuntime
from hipporag_core.toy_bootstrap import index_toy_graph, load_toy_passages


class TestToyBootstrap:
    """Load eval corpus into graph."""

    def test_load_and_index_toy(self, phrase_encoder: PhraseEncoder) -> None:
        from hipporag_core.indexer import Indexer

        store = MemoryGraphStore()
        indexer = Indexer(store=store, encoder=phrase_encoder, synonymy_threshold=0.8)
        runtime = HipporagRuntime(
            store=store,
            encoder=phrase_encoder,
            indexer=indexer,
            retriever=Retriever(store=store, encoder=phrase_encoder, damping=0.5),
        )
        count = load_toy_passages(runtime)
        assert count == 5
        summary = index_toy_graph(runtime)
        assert summary["indexed_passages"] == 5
        assert summary["entities"] > 0
        assert summary["triples"] == 5
