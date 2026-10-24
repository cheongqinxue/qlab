"""Indexing helpers: batch ingest and consolidation orchestration."""

from __future__ import annotations

from agents_utils.memory.base_encoder import BaseEncoder

from hmem_core.config import HMemConfig
from hmem_core.hybrid_store import InMemoryHybridStore
from hmem_core.indexing.event_extractor import build_event_extractor
from hmem_core.indexing.indexer import HybridIndexer
from hmem_core.models import GraphEdge, MemoryFragment


def build_indexer(
    store: InMemoryHybridStore,
    encoder: BaseEncoder,
    *,
    agents_mode: str,
) -> HybridIndexer:
    """Create a hybrid indexer for the store."""
    extractor = build_event_extractor(agents_mode=agents_mode)
    return HybridIndexer(store, encoder, extractor, agents_mode=agents_mode)


def index_fragments(
    store: InMemoryHybridStore,
    fragments: list[MemoryFragment],
    edges: list[GraphEdge],
    *,
    config: HMemConfig,
    encoder: BaseEncoder,
    agents_mode: str = "fixture",
) -> int:
    """Insert fragments via ``HybridIndexer`` and attach graph edges.

    Args:
        store: Hybrid store.
        fragments: Memory fragments.
        edges: Pre-extracted graph edges (toy fixtures).
        config: Config (unused when store carries config).
        encoder: Embedding model.
        agents_mode: ``fixture`` or ``llm``.

    Returns:
        Leaf event count.
    """
    _ = config
    indexer = build_indexer(store, encoder, agents_mode=agents_mode)
    leaves = 0
    for fragment in fragments:
        leaves += indexer.index_fragment(fragment)
    for edge in edges:
        store.graph.add_edge(edge, fragment_id=fragments[0].id if fragments else None)
        if edge not in store.edges:
            store.edges.append(edge)
    return leaves


def run_consolidation_pass(
    store: InMemoryHybridStore,
    *,
    config: HMemConfig,
    encoder: BaseEncoder,
    agents_mode: str = "fixture",
) -> int:
    """Run tree consolidation after ingest.

    Args:
        store: Hybrid store with leaves.
        config: Hyperparameters.
        encoder: Embedding model.
        agents_mode: Agent mode for summarizer selection.

    Returns:
        Summary node count.
    """
    _ = config, encoder, agents_mode
    indexer = build_indexer(store, encoder, agents_mode=agents_mode)
    summaries = indexer.finalize()
    store.summary_nodes = summaries
    store.indexed = True
    return summaries
