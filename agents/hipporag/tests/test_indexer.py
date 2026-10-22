"""Indexing pipeline with staged passages and OpenIE."""

from __future__ import annotations

from hipporag_core.indexer import Indexer
from hipporag_core.memory_graph_store import MemoryGraphStore


class TestIndexer:
    """Corpus staging and graph build."""

    def test_index_increments_counts(
        self,
        toy_passages,
        toy_openie_by_passage,
        phrase_encoder,
    ) -> None:
        store = MemoryGraphStore()
        indexer = Indexer(store=store, encoder=phrase_encoder, synonymy_threshold=0.8)
        for passage in toy_passages:
            store.stage_passage(passage)
        indexer.index_passages(openie_by_passage=toy_openie_by_passage)
        assert store.passage_count == len(toy_passages)
        assert store.triple_count >= len(toy_passages)
        assert store.entity_count > 0

    def test_skips_unchanged_passage_on_reindex(
        self,
        toy_passages,
        toy_openie_by_passage,
        phrase_encoder,
    ) -> None:
        store = MemoryGraphStore()
        indexer = Indexer(store=store, encoder=phrase_encoder, synonymy_threshold=0.8)
        store.stage_passage(toy_passages[0])
        indexer.index_passages(openie_by_passage=toy_openie_by_passage)
        first_triples = store.triple_count
        indexer.index_passages(openie_by_passage={})
        assert store.triple_count == first_triples
