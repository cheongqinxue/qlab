"""Shared runtime wiring (encoder, indexer, retriever)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hipporag_core.graph_store import create_graph_store
from hipporag_core.indexer import Indexer
from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.phrase_encoder import PhraseEncoder
from hipporag_core.retriever import Retriever


@dataclass
class HipporagRuntime:
    """Handles for indexing and retrieval."""

    store: MemoryGraphStore
    encoder: PhraseEncoder
    indexer: Indexer
    retriever: Retriever

    def close(self) -> None:
        """Release external resources."""
        from hipporag_core.neo4j_graph_store import Neo4jGraphStore

        if isinstance(self.store, Neo4jGraphStore):
            self.store.close()


def build_runtime(*, damping: float, synonymy_threshold: float) -> HipporagRuntime:
    """Construct runtime from environment.

    Args:
        damping: PPR damping factor.
        synonymy_threshold: Synonymy τ.

    Returns:
        Wired runtime bundle.
    """
    store = create_graph_store()
    pytest_run = os.environ.get("PYTEST_CURRENT_TEST") is not None
    if os.environ.get("HIPPORAG_TEST_ENCODER") == "hash" or pytest_run:
        encoder = PhraseEncoder()
    else:
        model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        encoder = PhraseEncoder.from_env_model(model)
    indexer = Indexer(
        store=store,
        encoder=encoder,
        synonymy_threshold=synonymy_threshold,
    )
    retriever = Retriever(store=store, encoder=encoder, damping=damping)
    return HipporagRuntime(
        store=store,
        encoder=encoder,
        indexer=indexer,
        retriever=retriever,
    )
