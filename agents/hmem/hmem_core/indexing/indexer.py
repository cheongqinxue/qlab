"""Offline indexing orchestrator (paper §3.2 — hybrid structure build)."""

from __future__ import annotations

from agents_utils.memory.base_encoder import BaseEncoder

from hmem_core.agents.protocols import EventExtractor, SummarizerAgent
from hmem_core.hybrid_store import InMemoryHybridStore
from hmem_core.indexing.agent_factory import build_summarizer
from hmem_core.indexing.graph_builder import GraphBuilder
from hmem_core.indexing.tree_builder import TreeBuilder
from hmem_core.models import MemoryFragment


class HybridIndexer:
    """Index one fragment into tree + graph."""

    def __init__(
        self,
        store: InMemoryHybridStore,
        encoder: BaseEncoder,
        extractor: EventExtractor,
        *,
        agents_mode: str = "fixture",
        summarizer: SummarizerAgent | None = None,
    ) -> None:
        """Wire store and agents."""
        self._store = store
        self._encoder = encoder
        self._extractor = extractor
        self._summarizer = summarizer or build_summarizer(agents_mode=agents_mode)
        assert store.tree is not None
        self._tree_builder = TreeBuilder(store.config, store.tree, encoder, self._summarizer)
        self._graph_builder = GraphBuilder(store.graph)

    def index_fragment(self, fragment: MemoryFragment) -> int:
        """Extract, insert events, and update graph."""
        self._store.fragments[fragment.id] = fragment
        events, edges = self._extractor.extract(fragment)
        for event in events:
            self._store.events.append(event)
            self._tree_builder.insert_event(event)
        self._graph_builder.ingest_edges(fragment, edges)
        self._store.edges.extend(edges)
        self._store.indexed = False
        return len(events)

    def finalize(self) -> int:
        """Consolidation pass after batch ingest."""
        summaries = self._tree_builder.consolidate()
        self._store.summary_nodes = summaries
        self._store.indexed = True
        return summaries
