"""Synonymy edge discovery (cosine ≥ τ)."""

from __future__ import annotations

import numpy as np


def pairs_above_threshold(
    vectors: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[int, int]]:
    """All unordered entity pairs with cosine similarity ≥ threshold.

    Args:
        vectors: ``(n, dim)`` row-normalized embeddings.
        threshold: Paper τ (default 0.8).

    Returns:
        List of ``(i, j)`` with ``i < j``.
    """
    mat = np.asarray(vectors, dtype=np.float64)
    n = mat.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        sims = mat[i + 1 :] @ mat[i]
        for offset, score in enumerate(sims):
            j = i + 1 + offset
            if score >= threshold:
                pairs.append((i, j))
    return pairs
