"""Query entity → graph node linking."""

from __future__ import annotations

from hipporag_core.linking import link_query_entities


class TestLinkQueryEntities:
    """Argmax cosine linking against indexed phrase vectors."""

    def test_links_to_best_matching_entity(
        self,
        indexed_toy_graph,
        phrase_encoder,
    ) -> None:
        snapshot = indexed_toy_graph.export_ppr_snapshot()
        links = link_query_entities(
            ["Stanford University"],
            entity_texts=snapshot.entity_texts,
            entity_vectors=snapshot.entity_vectors,
            encoder=phrase_encoder,
        )
        assert len(links) == 1
        assert links[0].entity_text in {"stanford", "stanford university"}

    def test_builds_reset_weights_with_node_specificity(
        self,
        indexed_toy_graph,
        phrase_encoder,
    ) -> None:
        snapshot = indexed_toy_graph.export_ppr_snapshot()
        links = link_query_entities(
            ["Professor Thomas"],
            entity_texts=snapshot.entity_texts,
            entity_vectors=snapshot.entity_vectors,
            encoder=phrase_encoder,
        )
        reset = indexed_toy_graph.build_reset_weights(
            [link.entity_index for link in links],
            node_specificity=True,
        )
        assert reset.sum() > 0
        assert len(reset) == len(snapshot.entity_texts)
