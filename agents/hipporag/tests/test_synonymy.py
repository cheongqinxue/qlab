"""Synonymy edge selection."""

from __future__ import annotations

import numpy as np

from hipporag_core.synonymy import pairs_above_threshold


class TestSynonymy:
    """Cosine threshold τ = 0.8."""

    def test_pairs_above_threshold(self) -> None:
        vectors = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pairs = pairs_above_threshold(vectors, threshold=0.8)
        assert (0, 1) in pairs or (1, 0) in pairs
        assert all(i != j for i, j in pairs)
