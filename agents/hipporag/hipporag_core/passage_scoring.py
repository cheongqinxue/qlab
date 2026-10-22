"""Map node-level PPR scores to passage rankings."""

from __future__ import annotations

import numpy as np


def score_passages_from_ppr(
    node_scores: np.ndarray,
    mention_matrix: np.ndarray,
) -> np.ndarray:
    """Compute passage scores as ``node_scores @ mention_matrix``.

    Args:
        node_scores: PPR mass per entity node (length ``|N|``).
        mention_matrix: ``(|N|, |P|)`` mention counts (paper matrix **P** transpose layout).

    Returns:
        Score per passage column.
    """
    node = np.asarray(node_scores, dtype=np.float64)
    mentions = np.asarray(mention_matrix, dtype=np.float64)
    return node @ mentions
