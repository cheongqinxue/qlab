"""Online retrieval: link query entities → PPR → ranked passages."""

from __future__ import annotations

from dataclasses import dataclass

from hipporag_core.linking import link_query_entities
from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.models import RetrievalHit
from hipporag_core.passage_scoring import score_passages_from_ppr
from agents_utils.memory.base_encoder import BaseEncoder
from agents_utils.graph.ppr import run_personalized_pagerank


@dataclass(frozen=True)
class RetrievalDebug:
    """Optional debug payload for API responses."""

    seed_entities: list[str]
    top_entities: list[tuple[str, float]]


class Retriever:
    """HippoRAG 1 retrieval loop (graph PPR)."""

    def __init__(
        self,
        *,
        store: MemoryGraphStore,
        encoder: BaseEncoder,
        damping: float = 0.5,
    ) -> None:
        """Create retriever.

        Args:
            store: Indexed graph store.
            encoder: Query linking encoder.
            damping: PPR damping factor.
        """
        self._store = store
        self._encoder = encoder
        self._damping = damping

    def retrieve(
        self,
        *,
        query: str,
        query_entities: list[str],
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Rank passages for a query.

        Args:
            query: Raw question (for future logging; entities drive retrieval).
            query_entities: NER phrases from the question.
            top_k: Number of passages to return.

        Returns:
            Passages sorted by score descending.
        """
        _ = query
        snapshot = self._store.export_ppr_snapshot()
        if not snapshot.passage_ids:
            return []

        links = link_query_entities(
            query_entities,
            entity_texts=snapshot.entity_texts,
            entity_vectors=snapshot.entity_vectors,
            encoder=self._encoder,
        )
        if not links:
            return []

        reset = self._store.build_reset_weights(
            [link.entity_index for link in links],
            node_specificity=True,
        )
        node_scores = run_personalized_pagerank(
            len(snapshot.entity_ids),
            snapshot.edges,
            reset,
            damping=self._damping,
        )
        passage_scores = score_passages_from_ppr(node_scores, snapshot.mentions)
        order = passage_scores.argsort()[::-1]

        hits: list[RetrievalHit] = []
        for rank_idx in order[:top_k]:
            pid = snapshot.passage_ids[int(rank_idx)]
            hits.append(
                RetrievalHit(
                    passage_id=pid,
                    score=float(passage_scores[int(rank_idx)]),
                    text=self._store.get_passage_text(pid),
                )
            )
        return hits

    def retrieve_with_debug(
        self,
        *,
        query: str,
        query_entities: list[str],
        top_k: int = 5,
    ) -> tuple[list[RetrievalHit], RetrievalDebug]:
        """Like ``retrieve`` but include seed entity labels."""
        snapshot = self._store.export_ppr_snapshot()
        links = link_query_entities(
            query_entities,
            entity_texts=snapshot.entity_texts,
            entity_vectors=snapshot.entity_vectors,
            encoder=self._encoder,
        )
        hits = self.retrieve(
            query=query,
            query_entities=query_entities,
            top_k=top_k,
        )
        debug = RetrievalDebug(
            seed_entities=[link.entity_text for link in links],
            top_entities=[],
        )
        return hits, debug
