"""Incremental temporal–semantic tree construction (paper §3.2).

Orchestrates ``TemporalSemanticTree`` + ``SummarizerAgent`` + ``BaseEncoder``.
Community reference: ``indexing/tree_builder.py`` (embed-on-insert, consolidate triggers).

Phase 2 will replace ``fixture_indexer`` leaf list with calls here.
"""

from __future__ import annotations

from agents_utils.memory.base_encoder import BaseEncoder

from hmem_core.agents.protocols import SummarizerAgent
from hmem_core.config import HMemConfig
from hmem_core.models import MemoryEvent
from hmem_core.tree.temporal_semantic_tree import TemporalSemanticTree


class TreeBuilder:
    """Maintains the temporal–semantic tree during offline indexing."""

    def __init__(
        self,
        config: HMemConfig,
        tree: TemporalSemanticTree,
        encoder: BaseEncoder,
        summarizer: SummarizerAgent,
    ) -> None:
        """Bind config, tree, encoder, and summarizer."""
        self._config = config
        self._tree = tree
        self._encoder = encoder
        self._summarizer = summarizer

    def insert_event(self, event: MemoryEvent) -> str:
        """Embed event text and insert into tree (partial: leaf only today).

        Args:
            event: Leaf memory event.

        Returns:
            Node id.
        """
        vector = self._encoder.encode_query_numpy(event.text)
        return self._tree.insert_event(event, embedding=vector.tolist())

    def consolidate(self) -> int:
        """Run upward merge pass using configured summarizer.

        Returns:
            Summary nodes created or updated.
        """
        return self._tree.consolidate(self._summarizer.summarize)
