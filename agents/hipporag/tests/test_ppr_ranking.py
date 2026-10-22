"""Personalized PageRank and multi-hop passage ranking (M1)."""

from __future__ import annotations

from hipporag_core.retriever import Retriever


class TestPprRanking:
    """Bridge passage should beat single-hop distractors on the toy corpus."""

    def test_multihop_question_ranks_bridge_passage_first(
        self,
        indexed_toy_graph,
        phrase_encoder,
    ) -> None:
        retriever = Retriever(
            store=indexed_toy_graph,
            encoder=phrase_encoder,
            damping=0.5,
        )
        hits = retriever.retrieve(
            query="Which Stanford professor works on the neuroscience of Alzheimer's?",
            query_entities=["Stanford", "Professor Thomas", "Alzheimer's disease"],
            top_k=5,
        )
        assert hits, "expected at least one hit"
        top_ids = [h.passage_id for h in hits[:3]]
        single_hop = {"p_stanford", "p_alz", "p_harvard"}
        bridge = {"p_prof_alz", "p_prof_stanford"}
        assert top_ids[0] not in single_hop
        assert any(pid in top_ids for pid in bridge)
        assert "p_harvard" not in top_ids[:1]

    def test_harvard_passage_not_top_for_stanford_query(
        self,
        indexed_toy_graph,
        phrase_encoder,
    ) -> None:
        retriever = Retriever(
            store=indexed_toy_graph,
            encoder=phrase_encoder,
            damping=0.5,
        )
        hits = retriever.retrieve(
            query="Stanford professor Alzheimer's",
            query_entities=["Stanford", "Alzheimer's disease"],
            top_k=3,
        )
        assert hits[0].passage_id != "p_harvard"
