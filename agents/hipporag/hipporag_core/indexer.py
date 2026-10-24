"""Offline indexing: OpenIE → graph + optional synonymy."""

from __future__ import annotations

from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.models import OpenIEExtraction
from agents_utils.memory.base_encoder import BaseEncoder
from hipporag_core.synonymy import pairs_above_threshold


class Indexer:
    """Build the hippocampal index from OpenIE extractions."""

    def __init__(
        self,
        *,
        store: MemoryGraphStore,
        encoder: BaseEncoder,
        synonymy_threshold: float = 0.8,
    ) -> None:
        """Configure indexer.

        Args:
            store: Graph backing store.
            encoder: Phrase embedding model.
            synonymy_threshold: τ for synonymy edges.
        """
        self._store = store
        self._encoder = encoder
        self._synonymy_threshold = synonymy_threshold

    def index_passages(
        self,
        *,
        openie_by_passage: dict[str, OpenIEExtraction],
    ) -> int:
        """Index pending passages using supplied OpenIE results.

        Args:
            openie_by_passage: Map passage id → extraction (from LLM or cache).

        Returns:
            Number of passages indexed in this call.
        """
        indexed = 0
        for passage_id in self._store.pending_passage_ids():
            extraction = openie_by_passage.get(passage_id)
            if extraction is None:
                continue
            self._index_one(extraction)
            self._store.mark_passage_indexed(passage_id)
            indexed += 1
        return indexed

    def _index_one(self, extraction: OpenIEExtraction) -> None:
        passage_id = extraction.passage_id
        entity_eids: dict[str, str] = {}
        for phrase in extraction.entities:
            eid = self._store.merge_entity(phrase, self._encoder)
            entity_eids[phrase] = eid
            self._store.add_mention(eid, passage_id, 1)

        for triple in extraction.triples:
            sub_eid = self._store.merge_entity(triple.subject, self._encoder)
            obj_eid = self._store.merge_entity(triple.object, self._encoder)
            self._store.add_mention(sub_eid, passage_id, 1)
            self._store.add_mention(obj_eid, passage_id, 1)
            self._store.add_rel(sub_eid, obj_eid)
            self._store.add_triple(triple)

    def add_synonymy_edges(self) -> int:
        """Add SYNONYM edges for entity pairs above τ.

        Returns:
            Number of new synonym pairs added.
        """
        snapshot = self._store.export_ppr_snapshot()
        if len(snapshot.entity_ids) < 2:
            return 0
        pairs = pairs_above_threshold(
            snapshot.entity_vectors,
            threshold=self._synonymy_threshold,
        )
        added = 0
        for i, j in pairs:
            a = snapshot.entity_ids[i]
            b = snapshot.entity_ids[j]
            before = self._store.synonym_edge_count
            self._store.add_synonym(a, b)
            if self._store.synonym_edge_count > before:
                added += 1
        return added
