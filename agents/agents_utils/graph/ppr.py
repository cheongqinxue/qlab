"""Personalized PageRank via igraph (shared by graph-RAG harnesses)."""

from __future__ import annotations

import numpy as np


def run_personalized_pagerank(
    num_nodes: int,
    edges: list[tuple[int, int, float]],
    reset_weights: np.ndarray,
    *,
    damping: float = 0.5,
) -> np.ndarray:
    """Run undirected weighted PPR on entity nodes.

    Args:
        num_nodes: Number of graph nodes.
        edges: Undirected edges as ``(source, target, weight)``.
        reset_weights: Teleport / reset mass per node (same length as ``num_nodes``).
        damping: PageRank damping factor (paper default 0.5).

    Returns:
        PageRank score per node.
    """
    import igraph as ig

    reset = np.asarray(reset_weights, dtype=np.float64).copy()
    reset = np.where(np.isnan(reset) | (reset < 0), 0.0, reset)
    if reset.sum() <= 0:
        reset = np.ones(num_nodes, dtype=np.float64)
    graph = ig.Graph(n=num_nodes, edges=[], directed=False)
    if edges:
        pairs = [(i, j) for i, j, _ in edges]
        weights = [w for _, _, w in edges]
        graph.add_edges(pairs)
        graph.es["weight"] = weights
    scores = graph.personalized_pagerank(
        vertices=range(num_nodes),
        damping=damping,
        directed=False,
        weights="weight" if edges else None,
        reset=reset.tolist(),
        implementation="prpack",
    )
    return np.asarray(scores, dtype=np.float64)
