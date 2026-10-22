"""Passage scores from PPR node mass (n′ᵀ P spirit)."""

from __future__ import annotations

import numpy as np

from hipporag_core.passage_scoring import score_passages_from_ppr


class TestPassageScoring:
    """Matrix aggregation from node scores to passages."""

    def test_aggregates_mention_counts(self) -> None:
        ppr = np.array([1.0, 0.5, 0.0])
        # entities 0,1 mention passage 0; entity 1 mentions passage 1
        mentions = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        scores = score_passages_from_ppr(ppr, mentions)
        assert scores[0] > scores[1]
