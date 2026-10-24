"""Evidence ranking scores (paper §3.3 — Eq. 1–3).

Semantic term ``S`` uses ``agents_utils.memory.dense_numpy.dense_cosine_similarity_numpy``.
Temporal ``T`` and robustness ``R`` follow paper definitions; weights ``θ`` from config.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from agents_utils.memory.dense_numpy import dense_cosine_similarity_numpy

from hmem_core.config import HMemConfig
from hmem_core.models import Evidence


def semantic_similarity(query_vec: np.ndarray, memory_vec: np.ndarray) -> float:
    """Cosine similarity S(m, Q) between embeddings."""
    return float(dense_cosine_similarity_numpy(query_vec, memory_vec))


def temporal_relevance(
    memory_start: datetime,
    memory_end: datetime,
    query_start: datetime | None,
    query_end: datetime | None,
    *,
    lam: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """Temporal score T(m, Q) — overlap plus normalized center distance (paper Eq. 1).

    Args:
        memory_start: Evidence interval start.
        memory_end: Evidence interval end.
        query_start: Inferred query window start (optional).
        query_end: Inferred query window end (optional).
        lam: Balance between overlap and distance.
        epsilon: Stability constant.

    Returns:
        Score in [0, 1] when query interval is set; 0.5 if query time unknown.
    """
    if query_start is None or query_end is None:
        return 0.5
    mem_len = (memory_end - memory_start).total_seconds()
    q_len = (query_end - query_start).total_seconds()
    union = max(mem_len + q_len, epsilon)
    overlap_start = max(memory_start, query_start)
    overlap_end = min(memory_end, query_end)
    overlap = max(0.0, (overlap_end - overlap_start).total_seconds())
    overlap_term = overlap / (mem_len + q_len - overlap + epsilon)
    cm = memory_start + (memory_end - memory_start) / 2
    ck = query_start + (query_end - query_start) / 2
    dist_term = 1.0 - abs((cm - ck).total_seconds()) / union
    return lam * overlap_term + (1 - lam) * dist_term


def memory_robustness(
    query_time: datetime,
    last_consolidated: datetime,
    *,
    consolidation_count: int,
    tau: float = 86400.0 * 30,
    eta: float = 0.1,
) -> float:
    """Robustness R(m, t) — Ebbinghaus-style decay (paper Eq. 2)."""
    delta = max(0.0, (query_time - last_consolidated).total_seconds())
    denom = tau * (1 + eta * np.log1p(consolidation_count))
    return float(np.exp(-delta / max(denom, 1e-6)))


def combined_score(evidence: Evidence, config: HMemConfig) -> float:
    """F(m, Q, t) = θ₁S + θ₂T + θ₃R using fields on ``Evidence``."""
    weights = config.score_weights
    return (
        weights.semantic * evidence.semantic
        + weights.temporal * evidence.temporal
        + weights.robustness * evidence.robustness
    )
