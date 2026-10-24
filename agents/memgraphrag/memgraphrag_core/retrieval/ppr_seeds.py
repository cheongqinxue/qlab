"""Structure-aware PPR seed weights from filtered memory nodes."""

from __future__ import annotations

import numpy as np

from memgraphrag_core.retrieval.layer_filter import FilteredNode

_STRUCTURE_WEIGHT: dict[str, float] = {
    "schema": 0.6,
    "fact": 1.0,
    "passage": 0.5,
}


def build_ppr_seeds(
    filtered_nodes: list[FilteredNode],
    *,
    num_nodes: int,
    key_to_index: dict[tuple[str, str], int],
) -> np.ndarray:
    """Map filtered nodes to a personalization vector for PPR."""
    reset = np.zeros(num_nodes, dtype=np.float64)
    for node in filtered_nodes:
        idx = key_to_index.get((node.kind, node.record_id))
        if idx is None:
            continue
        weight = _STRUCTURE_WEIGHT.get(node.kind, 0.5)
        reset[idx] += weight * node.similarity
    if reset.sum() <= 0 and num_nodes > 0:
        reset[:] = 1.0 / num_nodes
    return reset
