"""Unit tests for hybrid indexing."""

from __future__ import annotations

from datetime import datetime, timezone

from agents_utils.memory.hash_dense_encoder import HashDenseEncoder

from hmem_core.config import HMemConfig
from hmem_core.hybrid_store import InMemoryHybridStore
from hmem_core.indexing.fixture_indexer import index_fragments, run_consolidation_pass
from hmem_core.models import GraphEdge, MemoryFragment


class TestFixtureIndexer:
    """Offline indexing."""

    def test_index_fragments_creates_leaves(self) -> None:
        """Each fragment becomes one leaf event."""
        config = HMemConfig()
        store = InMemoryHybridStore(config=config)
        encoder = HashDenseEncoder()
        fragments = [
            MemoryFragment(
                id="a",
                text="Hello",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        ]
        added = index_fragments(
            store,
            fragments,
            [GraphEdge(head="Hello", relation="mentions", tail="world")],
            config=config,
            encoder=encoder,
        )
        assert added == 1
        assert store.leaf_count == 1
        assert store.entity_count >= 1

    def test_consolidation_sets_summary_nodes(self) -> None:
        """Consolidation pass creates summary nodes when embeddings align."""
        config = HMemConfig(similarity_thresholds=[0.5, 0.5, 0.5])
        store = InMemoryHybridStore(config=config)
        encoder = HashDenseEncoder()
        text = "Mochi the cat sat on the mat"
        index_fragments(
            store,
            [
                MemoryFragment(
                    id="a",
                    text=text,
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                ),
                MemoryFragment(
                    id="b",
                    text=text,
                    timestamp=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                ),
            ],
            [],
            config=config,
            encoder=encoder,
        )
        summaries = run_consolidation_pass(
            store,
            config=config,
            encoder=encoder,
        )
        assert summaries >= 1
        assert store.indexed is True
