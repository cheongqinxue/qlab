"""Shared runtime wiring (encoder, indexer, retriever)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from agents_utils.memory.base_encoder import BaseEncoder
from agents_utils.memory.sentence_transformers_encoder import SentenceTransformersEncoder

from hipporag_core.graph_store import create_graph_store
from hipporag_core.indexer import Indexer
from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.retriever import Retriever


@dataclass
class HipporagRuntime:
    """Handles for indexing and retrieval."""

    store: MemoryGraphStore
    encoder: BaseEncoder
    indexer: Indexer
    retriever: Retriever

    def close(self) -> None:
        """Release external resources."""
        from hipporag_core.neo4j_graph_store import Neo4jGraphStore

        if isinstance(self.store, Neo4jGraphStore):
            self.store.close()


def build_runtime(
    *,
    damping: float,
    synonymy_threshold: float,
    encoder: BaseEncoder | None = None,
) -> HipporagRuntime:
    """Construct runtime from environment.

    Args:
        damping: PPR damping factor.
        synonymy_threshold: Synonymy τ.
        encoder: Dense encoder; defaults to ``EMBEDDING_MODEL`` when omitted.

    Returns:
        Wired runtime bundle.
    """
    store = create_graph_store()
    resolved = encoder
    if resolved is None:
        model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        resolved = SentenceTransformersEncoder(model)
    indexer = Indexer(
        store=store,
        encoder=resolved,
        synonymy_threshold=synonymy_threshold,
    )
    retriever = Retriever(store=store, encoder=resolved, damping=damping)
    return HipporagRuntime(
        store=store,
        encoder=resolved,
        indexer=indexer,
        retriever=retriever,
    )
